# # model.py
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # -----------------------------
# # 工具：矩形区域掩码
# # -----------------------------
# def generate_mask(H, W, top_left, bottom_right):
#     """
#     生成二维矩形掩码（float32）
#     Args:
#         H, W: 原图高宽
#         top_left: (r1, c1) 行/列起点（含）
#         bottom_right: (r2, c2) 行/列终点（含）
#     Return:
#         torch.FloatTensor [H, W]，矩形内为1，外为0
#     """
#     r1, c1 = top_left
#     r2, c2 = bottom_right
#     mask = torch.zeros((H, W), dtype=torch.float32)
#     r2 = min(r2, H - 1)
#     c2 = min(c2, W - 1)
#     mask[r1:r2 + 1, c1:c2 + 1] = 1.0
#     return mask
#
#
# # -----------------------------
# # 区域特征提取（轻量 CNN + 掩码池化）
# # 与你原代码 RegionConvSPP 的接口一致：
# # out 形状 [B, 1, C]，堆叠后可得到 [B, N, C]
# # -----------------------------
# class RegionConvSPP(nn.Module):
#     def __init__(self, in_channels=1, out_channels=64):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2   = nn.BatchNorm2d(out_channels)
#         self.act   = nn.ReLU(inplace=True)
#         # 你也可以在这里加轻量 SPP/多尺度分支；这里保持简洁稳定
#
#     @torch.no_grad()
#     def _down_mask(self, mask, size):
#         # mask: [H, W] -> [1,1,h',w']
#         m = mask.unsqueeze(0).unsqueeze(0).float()
#         m = F.interpolate(m, size=size, mode="nearest")
#         return m
#
#     def forward(self, x, mask_2d):
#         """
#         x: [B, 1, H, W]
#         mask_2d: [H, W]
#         return: [B, 1, C]
#         """
#         feat = self.act(self.bn1(self.conv1(x)))         # [B,32,H,W]
#         feat = self.act(self.bn2(self.conv2(feat)))      # [B,C,H,W]
#
#         m = self._down_mask(mask_2d, feat.shape[2:])     # [1,1,H',W']
#         w = m / (m.sum(dim=(2,3), keepdim=True) + 1e-8)  # 归一化权重
#         pooled = (feat * w).sum(dim=(2,3))               # [B,C]
#         return pooled.unsqueeze(1)                       # [B,1,C]
#
#
# # -----------------------------
# # 功能关系邻接（同一样本内，基于节点嵌入相似度）
# # -----------------------------
# def build_functional_adj_from_feats(
#     node_feats,          # [B, N, D]
#     hard=True,           # True: 二值，False: 连续
#     topk=2,              # 每个节点至少保留的功能边数
#     tau=0.2,             # 相似度阈值
#     symmetrize=True,     # 是否对称化
#     remove_self=True,    # 去掉对角
#     detach=True,         # 是否阻断梯度（建议 True）
#     eps=1e-8
# ):
#     X = node_feats.detach() if detach else node_feats  # [B,N,D]
#     Xn = X / (X.norm(dim=-1, keepdim=True) + eps)
#     sim = torch.einsum("bid,bjd->bij", Xn, Xn)        # [B,N,N] 余弦相似
#     if remove_self:
#         eye = torch.eye(sim.size(1), device=sim.device).unsqueeze(0)
#         sim = sim * (1.0 - eye)
#
#     if hard:
#         A = (sim >= tau).float()
#         if topk is not None and topk > 0:
#             vals, idx = torch.topk(sim, k=min(topk, sim.size(1)-1), dim=-1)
#             topk_mask = torch.zeros_like(sim)
#             topk_mask.scatter_(-1, idx, 1.0)
#             A = torch.max(A, topk_mask)  # 阈值 ∪ top-k
#         if symmetrize:
#             A = torch.max(A, A.transpose(1, 2))
#         return A                          # [B,N,N] 0/1
#     else:
#         A = F.relu(sim)
#         if symmetrize:
#             A = 0.5 * (A + A.transpose(1, 2))
#         A = A / (A.sum(dim=-1, keepdim=True) + eps)
#         return A                          # [B,N,N] 连续权重
#
#
# # -----------------------------
# # 多关系图注意力（单层）
# # -----------------------------
# def _masked_softmax(logits, mask, dim=-1):
#     logits = logits.masked_fill(~mask.bool(), float('-inf'))
#     return torch.softmax(logits, dim=dim)
#
# class MultiRelGATLayer(nn.Module):
#     """
#     对每个关系 r 和头 m，学习 (W_{r,m}, a_{r,m})，邻接掩码内 softmax 归一并聚合。
#     输入:
#       H: [B, N, Din]
#       A_dict: {rel: [B,N,N] 0/1}
#       edge_dict: 可选 {rel: [B,N,N,De]}
#     输出:
#       H_out: [B, N, Dout]
#     """
#     def __init__(self, in_dim, out_dim, relations,
#                  num_heads=4, edge_dim=0, dropout=0.1, alpha=0.2, concat=True):
#         super().__init__()
#         assert out_dim % num_heads == 0 if concat else True, \
#             "当 concat=True 时，out_dim 必须能被 num_heads 整除"
#
#         self.in_dim   = in_dim
#         self.out_dim  = out_dim
#         self.R        = list(relations)
#         self.H        = num_heads
#         self.edim     = edge_dim
#         self.concat   = concat
#         self.head_dim = out_dim // num_heads if concat else out_dim
#
#         # 每关系每头线性
#         self.W = nn.ParameterDict({
#             r: nn.Parameter(torch.Tensor(self.H, in_dim, self.head_dim)) for r in self.R
#         })
#         # 注意力向量
#         self.a_src = nn.ParameterDict({ r: nn.Parameter(torch.Tensor(self.H, self.head_dim)) for r in self.R })
#         self.a_dst = nn.ParameterDict({ r: nn.Parameter(torch.Tensor(self.H, self.head_dim)) for r in self.R })
#         self.a_edge = None
#         if edge_dim > 0:
#             self.a_edge = nn.ParameterDict({ r: nn.Parameter(torch.Tensor(self.H, edge_dim)) for r in self.R })
#
#         self.leakyrelu = nn.LeakyReLU(alpha)
#         self.dropout   = nn.Dropout(dropout)
#         self.res_proj  = nn.Linear(in_dim, out_dim, bias=False) if (in_dim != out_dim) else nn.Identity()
#         self.ln        = nn.LayerNorm(out_dim)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for r in self.R:
#             nn.init.xavier_uniform_(self.W[r])
#             nn.init.xavier_uniform_(self.a_src[r].unsqueeze(-1))
#             nn.init.xavier_uniform_(self.a_dst[r].unsqueeze(-1))
#             if self.a_edge is not None:
#                 nn.init.xavier_uniform_(self.a_edge[r].unsqueeze(-1))
#         if isinstance(self.res_proj, nn.Linear):
#             nn.init.xavier_uniform_(self.res_proj.weight)
#
#     def forward(self, H, A_dict, edge_dict=None):
#         B, N, _ = H.shape
#         per_rel = []
#
#         for r in self.R:
#             A = A_dict[r].to(H.device).float()  # [B,N,N] 0/1
#             mask = A > 0
#
#             # Wh: [B,H,N,Dh]
#             Wh = torch.einsum('bnd,hdf->bh nf', H, self.W[r]).contiguous()
#             Wh = Wh.view(B, self.H, N, self.head_dim)
#
#             e_src = torch.einsum('bhnd,hd->bhn', Wh, self.a_src[r])  # [B,H,N]
#             e_dst = torch.einsum('bhnd,hd->bhn', Wh, self.a_dst[r])  # [B,H,N]
#             e = e_src.unsqueeze(-1) + e_dst.unsqueeze(-2)            # [B,H,N,N]
#
#             if (edge_dict is not None) and (r in edge_dict) and (edge_dict[r] is not None):
#                 # 边特征可加权
#                 eta = edge_dict[r].to(H.device).float()              # [B,N,N,De]
#                 e_edge = torch.einsum('bijn,hn->bhij', eta, self.a_edge[r])
#                 e = e + e_edge
#
#             e = self.leakyrelu(e)
#             alpha = _masked_softmax(e, mask.unsqueeze(1).expand(B, self.H, N, N), dim=-1)
#             alpha = self.dropout(alpha)
#
#             # 聚合
#             out_r = torch.einsum('bhij,bhjd->bhid', alpha, Wh)  # [B,H,N,Dh]
#             per_rel.append(out_r)
#
#         # 跨关系平均
#         out = torch.stack(per_rel, dim=0).sum(dim=0) / float(len(per_rel))  # [B,H,N,Dh]
#
#         # 合并头
#         out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.H * self.head_dim)  # [B,N,Dout]
#
#         # 残差 + LN
#         out = self.ln(self.res_proj(H) + self.dropout(out))
#         return out
#
#
# # -----------------------------
# # MR-GAT 模型（单层）
# # 与你原 GCN 的用法兼容：forward(H, return_node_feats=...)
# # 内部自动构建 A_sp / A_sym / A_fun
# # -----------------------------
# class MR_GAT(nn.Module):
#     def __init__(self, nfeat, nhid, mat_path,
#                  use_sym=True, use_fun=True, heads=4, dropout=0.2, edge_dim=0):
#         super().__init__()
#         # 读取空间邻接（细层）
#         adj_np = np.load(mat_path).astype(np.float32)  # [N,N]
#         self.register_buffer('A_sp', torch.from_numpy(adj_np))
#         self.use_sym = use_sym
#         self.use_fun = use_fun
#
#         # 对称邻接（默认 cheeks 在索引 1/2）
#         if use_sym:
#             A_sym = np.zeros_like(adj_np, dtype=np.float32)
#             if A_sym.shape[0] >= 3:
#                 A_sym[1, 2] = 1.0
#                 A_sym[2, 1] = 1.0
#             self.register_buffer('A_sym', torch.from_numpy(A_sym))
#         else:
#             self.A_sym = None
#
#         rels = ['spatial']
#         if use_sym: rels.append('symmetry')
#         if use_fun: rels.append('functional')
#
#         self.gat = MultiRelGATLayer(
#             in_dim=nfeat, out_dim=nhid, relations=rels,
#             num_heads=heads, edge_dim=edge_dim, dropout=dropout, concat=True
#         )
#         self.readout = nn.Sequential(nn.LayerNorm(nhid), nn.Linear(nhid, 2))
#
#     def _build_relations(self, H):
#         """
#         H: [B, N, D]
#         Return: {rel: [B,N,N]}
#         """
#         B, N, _ = H.shape
#         A_dict = {}
#         A_dict['spatial'] = self.A_sp.unsqueeze(0).expand(B, -1, -1).float()
#         if self.use_sym and (self.A_sym is not None):
#             A_dict['symmetry'] = self.A_sym.unsqueeze(0).expand(B, -1, -1).float()
#         if self.use_fun:
#             A_dict['functional'] = build_functional_adj_from_feats(
#                 H, hard=True, topk=2, tau=0.2, symmetrize=True, detach=True
#             ).float()
#         return A_dict
#
#     def forward(self, H, return_node_feats=False):
#         """
#         H: [B, N, nfeat] —— 你的 weighted_region_feats
#         """
#         A_dict = self._build_relations(H)
#         H1 = self.gat(H, A_dict, edge_dict=None)  # [B,N,nhid]
#         g  = H1.mean(dim=1)                       # mean pooling
#         logits = self.readout(g)                  # [B,2]
#         if return_node_feats:
#             return logits, H1
#         return logits


# model.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
    # Pooling 函数
    # =============================
def mean_pooling(H, mask=None):
        # H: [B,N,D]
    return H.mean(dim=1)  # [B,D]

def max_pooling(H, mask=None):
    return H.max(dim=1).values  # [B,D]

def attention_pooling(H, attn_mtx):
        """
        H: [B,N,D]
        attn_mtx: [B,N,N] 邻接内注意力 (通常来自 GAT 的 alpha)
        返回: [B,D] —— graph-level attention pooling
        """
        # 对节点级特征做 attention 聚合
        out = torch.einsum('bij,bjd->bid', attn_mtx, H)  # [B,N,D]
        return out.mean(dim=1)  # [B,D]，也可以选 sum

class ConcatMLPPooling(nn.Module):
        def __init__(self, N, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(N * in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )

        def forward(self, H, mask=None):
            B, N, D = H.shape
            flat = H.reshape(B, N * D)
            return self.mlp(flat)  # [B,out_dim]

# =============================
# 工具：矩形区域掩码
# =============================
def generate_mask(H, W, top_left, bottom_right):
    """
    生成二维矩形掩码（float32）
    Args:
        H, W: 原图高宽
        top_left: (r1, c1) 行/列起点（含）
        bottom_right: (r2, c2) 行/列终点（含）
    Return:
        torch.FloatTensor [H, W]，矩形内为1，外为0
    """
    r1, c1 = top_left
    r2, c2 = bottom_right
    mask = torch.zeros((H, W), dtype=torch.float32)
    r2 = min(r2, H - 1)
    c2 = min(c2, W - 1)
    mask[r1:r2 + 1, c1:c2 + 1] = 1.0
    return mask


# =============================
# 区域特征提取（轻量 CNN + 掩码池化）
# 与你原代码 RegionConvSPP 的接口一致：
# 输出 [B, 1, C]，堆叠后得到 [B, N, C]
# =============================
class RegionConvSPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.act   = nn.ReLU(inplace=True)

    @torch.no_grad()
    def _down_mask(self, mask, size):
        # mask: [H, W] -> [1,1,h',w']
        m = mask.unsqueeze(0).unsqueeze(0).float()
        m = F.interpolate(m, size=size, mode="nearest")
        return m

    def forward(self, x, mask_2d):
        """
        x: [B, 1, H, W]
        mask_2d: [H, W]
        return: [B, 1, C]
        """
        feat = self.act(self.bn1(self.conv1(x)))         # [B,32,H,W]
        feat = self.act(self.bn2(self.conv2(feat)))      # [B,C,H,W]

        m = self._down_mask(mask_2d, feat.shape[2:])     # [1,1,H',W']
        w = m / (m.sum(dim=(2,3), keepdim=True) + 1e-8)  # 归一化权重
        pooled = (feat * w).sum(dim=(2,3))               # [B,C]
        return pooled.unsqueeze(1)                       # [B,1,C]


# =============================
# 功能关系邻接（同一样本内，基于节点嵌入相似度）
# =============================
def build_functional_adj_from_feats(
    node_feats,          # [B, N, D]
    hard=True,           # True: 二值，False: 连续
    topk=1,              # 每个节点至少保留的功能边数（>=1 避免空邻域）
    tau=0.2,             # 相似度阈值
    symmetrize=True,     # 是否对称化
    remove_self=True,    # 去掉对角
    detach=True,         # 是否阻断梯度（建议 True）
    eps=1e-8
):
    X = node_feats.detach() if detach else node_feats
    Xn = X / (X.norm(dim=-1, keepdim=True) + eps)
    sim = torch.einsum("bid,bjd->bij", Xn, Xn)          # [B,N,N] 余弦相似
    if remove_self:
        eye = torch.eye(sim.size(1), device=sim.device).unsqueeze(0)
        sim = sim * (1.0 - eye)

    if hard:
        A = (sim >= tau).float()
        if topk is not None and topk > 0:
            k = min(topk, sim.size(1) - 1)
            vals, idx = torch.topk(sim, k=k, dim=-1)
            topk_mask = torch.zeros_like(sim)
            topk_mask.scatter_(-1, idx, 1.0)
            A = torch.max(A, topk_mask)                 # 阈值 ∪ top-k
        if symmetrize:
            A = torch.max(A, A.transpose(1, 2))
        return A                                        # [B,N,N] 0/1
    else:
        A = F.relu(sim)
        if symmetrize:
            A = 0.5 * (A + A.transpose(1, 2))
        A = A / (A.sum(dim=-1, keepdim=True) + eps)
        return A                                        # [B,N,N] 连续权重


# =============================
# 安全 masked softmax：防空邻域 / 防 NaN
# =============================
def _masked_softmax(logits, mask, dim=-1):
    """
    在邻接掩码内做 softmax；若某行邻域全空，则该行权重置 0（避免 NaN）。
    """
    mask = mask.bool()
    masked = logits.masked_fill(~mask, float('-inf'))

    # 找出“整行都被屏蔽”的情况
    all_false = (~mask).all(dim=dim, keepdim=True)

    # 避免 softmax(-inf...-inf) -> NaN
    masked = masked.masked_fill(all_false, 0.0)
    out = torch.softmax(masked, dim=dim)

    # 真的空邻域行，直接全 0
    out = torch.where(all_false, torch.zeros_like(out), out)
    return out


# =============================
# 多关系图注意力（单层）
# =============================
class MultiRelGATLayer(nn.Module):
    """
    对每个关系 r 和头 m，学习 (W_{r,m}, a_{r,m})，邻接掩码内 softmax 归一并聚合。
    输入:
      H: [B, N, Din]
      A_dict: {rel: [B,N,N] 0/1}
      edge_dict: 可选 {rel: [B,N,N,De]}
    输出:
      H_out: [B, N, Dout]
    """
    def __init__(self, in_dim, out_dim, relations,
                 num_heads=4, edge_dim=0, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        assert (not concat) or (out_dim % num_heads == 0), \
            "当 concat=True 时，out_dim 必须能被 num_heads 整除"

        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.R        = list(relations)
        self.H        = num_heads
        self.edim     = edge_dim
        self.concat   = concat
        self.head_dim = out_dim // num_heads if concat else out_dim

        # 每关系每头线性
        self.W = nn.ParameterDict({
            r: nn.Parameter(torch.Tensor(self.H, in_dim, self.head_dim)) for r in self.R
        })
        # 注意力向量
        self.a_src = nn.ParameterDict({ r: nn.Parameter(torch.Tensor(self.H, self.head_dim)) for r in self.R })
        self.a_dst = nn.ParameterDict({ r: nn.Parameter(torch.Tensor(self.H, self.head_dim)) for r in self.R })
        self.a_edge = None
        if edge_dim > 0:
            self.a_edge = nn.ParameterDict({ r: nn.Parameter(torch.Tensor(self.H, edge_dim)) for r in self.R })

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout   = nn.Dropout(dropout)
        self.res_proj  = nn.Linear(in_dim, out_dim, bias=False) if (in_dim != out_dim) else nn.Identity()
        self.ln        = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for r in self.R:
            nn.init.xavier_uniform_(self.W[r])
            nn.init.xavier_uniform_(self.a_src[r].unsqueeze(-1))
            nn.init.xavier_uniform_(self.a_dst[r].unsqueeze(-1))
            if self.a_edge is not None:
                nn.init.xavier_uniform_(self.a_edge[r].unsqueeze(-1))
        if isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight)

    def forward(self, H, A_dict, edge_dict=None):
        """
        H: [B, N, Din]
        A_dict: {rel: [B,N,N]}
        edge_dict: 可选 {rel: [B,N,N,De]}
        """
        B, N, _ = H.shape
        per_rel = []

        for r in self.R:
            A = A_dict[r].to(H.device).float()              # [B,N,N] 0/1
            mask = A > 0                                    # [B,N,N] bool

            # Wh: [B,H,N,Dh]
            # 修正 einsum（避免 'bh nf' 的空格问题）
            # 线性投影：Wh [B, H, N, Dh]
            Wh = torch.einsum('bnd,hdf->bhnf', H, self.W[r])  # 注意这里是 bhnf

            # 注意力打分（源/目的），a_src/a_dst 形状为 [H, Dh]
            e_src = torch.einsum('bhnf,hf->bhn', Wh, self.a_src[r])  # [B,H,N]
            e_dst = torch.einsum('bhnf,hf->bhn', Wh, self.a_dst[r])  # [B,H,N]
            e = e_src.unsqueeze(-1) + e_dst.unsqueeze(-2)  # [B,H,N,N]

            # （可选）边特征：eta [B,N,N,De]，a_edge[r] [H,De]
            if (edge_dict is not None) and (r in edge_dict) and (edge_dict[r] is not None):
                eta = edge_dict[r].to(H.device).float()  # [B,N,N,De]
                e_edge = torch.einsum('bijd,hd->bhij', eta, self.a_edge[r])
                e = e + e_edge

            e = self.leakyrelu(e)

            # 邻域内 softmax（用你的 _masked_softmax 防空邻域）
            alpha = _masked_softmax(e, mask.unsqueeze(1).expand(B, self.H, N, N), dim=-1)
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

            # 聚合：out_r [B,H,N,Dh]
            out_r = torch.einsum('bhij,bhjn->bhin', alpha, Wh)
            per_rel.append(out_r)

        # 跨关系求和 / 平均
        out = torch.stack(per_rel, dim=0).sum(dim=0) / float(len(per_rel))  # [B,H,N,Dh]

        # 合并多头
        out = out.permute(0, 2, 1, 3).contiguous()  # [B,N,H,Dh]
        out = out.view(B, N, self.H * self.head_dim)  # [B,N,Dout]

        # 残差 + LN
        out = self.ln(self.res_proj(H) + self.dropout(out))
        return out


# =============================
# 自环工具
# =============================
def _add_self_loops(A):  # A: [B, N, N]
    B, N, _ = A.shape
    I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, -1, -1)
    return torch.clamp(A + I, max=1.0)


# =============================
# MR-GAT 模型（单层）
# 与原 GCN 用法兼容：forward(H, return_node_feats=...)
# 内部自动构建 A_sp / A_sym / A_fun（可选）
# =============================
class MR_GAT(nn.Module):
    def __init__(self, nfeat, nhid, mat_path, use_fun=True, heads=4, dropout=0.2, edge_dim=0, pooling='mean'):
        super().__init__()

        # 读取空间邻接（稀疏/二值）
        adj_np = np.load(mat_path).astype(np.float32)  # [N,N]
        self.register_buffer('A_sp', torch.from_numpy(adj_np))
        self.use_fun = use_fun

        rels = ['spatial']
        if use_fun: rels.append('functional')

        self.gat = MultiRelGATLayer(
            in_dim=nfeat, out_dim=nhid, relations=rels,
            num_heads=heads, edge_dim=edge_dim, dropout=dropout, concat=True
        )

        # 🔥 选择 pooling
        self.pooling = pooling.lower()
        if self.pooling == "concat+mlp":
            # 注意：N 要提前知道（这里从 mat_path 读到的邻接矩阵大小）
            N = self.A_sp.shape[0]
            self.concat_pooler = ConcatMLPPooling(N, nhid, hidden_dim=128, out_dim=nhid)

        self.readout = nn.Sequential(nn.LayerNorm(nhid), nn.Linear(nhid, 2))

    def _build_relations(self, H):
        """
        H: [B, N, D]
        Return: {rel: [B,N,N]}，统一加自环，避免空邻域
        """
        B, N, _ = H.shape
        A_dict = {}
        A_dict['spatial'] = self.A_sp.unsqueeze(0).expand(B, -1, -1).float()
        if self.use_fun:
            A_dict['functional'] = build_functional_adj_from_feats(
                H, hard=True, topk=1, tau=0.2, symmetrize=True, detach=True
            ).float()

        # 统一加自环，防止空邻域导致 softmax NaN
        for k in list(A_dict.keys()):
            A_dict[k] = _add_self_loops(A_dict[k])

        return A_dict

    def forward(self, H, return_node_feats=False):
        """
        H: [B, N, nfeat] —— 你的 weighted_region_feats
        """
        A_dict = self._build_relations(H)
        H1 = self.gat(H, A_dict, edge_dict=None)  # [B,N,nhid]

        if self.pooling == "mean":
            g = mean_pooling(H1)
        elif self.pooling == "max":
            g = max_pooling(H1)
        elif self.pooling == "attention":
            # 用功能邻接 A_dict['functional'] 或融合后的邻接来做 attn pooling
            g = attention_pooling(H1, A_dict['functional'])
        elif self.pooling == "concat+mlp":
            g = self.concat_pooler(H1)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        logits = self.readout(g)                  # [B,2]
        if return_node_feats:
            return logits, H1
        return logits
