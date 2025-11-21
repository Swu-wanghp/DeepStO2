import os
import sys
import random
import numpy as np
from typing import Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from sklearn.metrics import f1_score

# 将上一层路径添加到sys系统中
sys.path.append("..")

from utils import parser
from model import generate_mask, RegionConvSPP, MR_GAT
from datasets import StO2Dataset, transformer
from torch.utils.data import DataLoader

# -------------------- 可选：更高 matmul 精度以利 A100/4090 --------------------
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

#
def entropy_loss_from_weights(weights, backprop: bool, eps: float = 1e-8):
    """
    weights: [B,N] 或 [B,N,1]
    backprop: True 则参与反传；False 则仅监控（detach）
    """
    if weights.dim() == 3:
        weights = weights.squeeze(-1)              # [B,N]
    if not backprop:
        weights = weights.detach()
    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
    weights = weights.clamp_min(eps)               # 数值稳定
    ent = -(weights * weights.log()).sum(dim=1)    # [B]
    return ent.mean()

# 双颊对称损失函数
def symmetry_loss(node_feats, left_idx, right_idx, p=1, safe=True, eps=1e-6):
    """
    node_feats: [B, N, D]（建议用 LN 后的节点输出，例如 h_out）
    left_idx/right_idx: 对称成对的索引列表，长度必须相等
    """
    li = torch.as_tensor(left_idx, device=node_feats.device, dtype=torch.long)
    ri = torch.as_tensor(right_idx, device=node_feats.device, dtype=torch.long)
    assert li.numel() == ri.numel(), "left_idx 和 right_idx 必须成对"

    left  = node_feats[:, li, :]  # [B, P, D]
    right = node_feats[:, ri, :]  # [B, P, D]

    if safe:
        # 可选：单位化，避免尺度飘
        left  = left  / (left.norm(dim=-1, keepdim=True)  + eps)
        right = right / (right.norm(dim=-1, keepdim=True) + eps)
        # 清理潜在的 NaN/Inf
        left  = torch.nan_to_num(left,  nan=0.0, posinf=1.0, neginf=-1.0)
        right = torch.nan_to_num(right, nan=0.0, posinf=1.0, neginf=-1.0)

    if p == 1:
        diff = (left - right).abs()
    elif p == 2:
        diff = (left - right).pow(2)
    else:
        # 也可用余弦距离，界于[0,2]，更稳
        cos = F.cosine_similarity(left, right, dim=-1).clamp(-1+eps, 1-eps)
        diff = 1.0 - cos  # [B, P]
        return diff.mean()

    return diff.mean()

# # 仅对 3 对区域做平滑：F-LC, F-RC, LC-RC
SMOOTH_PAIRS = [(0, 1), (0, 2), (1, 2)]  # 索引顺序与你堆叠一致: [F, LC, RC, N, J]
# 额头-脸颊区域一致性损失函数
def simple_pairwise_smoothness(node_feats, pairs=SMOOTH_PAIRS, norm_by_dim=True):
    """
    node_feats: [B, N, D] —— 节点特征（如 GCN/MR-GAT 输出的节点表征）
    pairs: 需要平滑的成对索引列表
    norm_by_dim: 按特征维度 D 归一，便于与其他项同量纲
    """
    B, N, D = node_feats.shape
    loss = node_feats.new_tensor(0.0)
    for (i, j) in pairs:
        diff = node_feats[:, i, :] - node_feats[:, j, :]      # [B, D]
        term = (diff.pow(2).sum(dim=-1) / D).mean() if norm_by_dim else diff.pow(2).mean()
        loss = loss + term
    return loss / max(len(pairs), 1)

# ===== 固定随机种子 =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 训练追求速度 => 不强制 deterministic；允许 benchmark
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# -------------------- 单次前向的区域特征提取（保持你现有 RegionConvSPP API） --------------------
@torch.no_grad()
def _prep_masks_on_device(masks_np: Dict[str, torch.Tensor], device):
    """
    将 numpy/int/bool 掩码 -> torch.float32(0/1)，并缓存到 device 上。
    """
    masks_dev = {}
    for k, m in masks_np.items():
        if isinstance(m, np.ndarray):
            t = torch.from_numpy(m)
        else:
            t = m
        t = t.to(device=device, dtype=torch.float32)  # [H, W]
        masks_dev[k] = t
    return masks_dev

def _extract_region_feats(region_extractor, images, masks_dev):
    """
    区域特征提取
    返回: region_feats [B, 5, C]
    """
    feat_forehead    = region_extractor(images, masks_dev["forehead"])
    feat_left_cheek  = region_extractor(images, masks_dev["left_cheek"])
    feat_right_cheek = region_extractor(images, masks_dev["right_cheek"])
    feat_nose        = region_extractor(images, masks_dev["nose"])
    feat_jaw         = region_extractor(images, masks_dev["jaw"])

    region_feats = torch.stack(
        [feat_forehead, feat_left_cheek, feat_right_cheek, feat_nose, feat_jaw],
        dim=1
    )  # 期望 [B, 5, C] 或 [B, 5, C, 1]
    if region_feats.dim() == 4 and region_feats.size(-1) == 1:
        region_feats = region_feats.squeeze(-1)
    return region_feats  # [B, 5, C]

# -------------------- 训练与验证 --------------------
def validate(model, region_extractor, val_loader, device, masks_dev,
             left_idx=[1], right_idx=[2], adj_dict=None):
    model.eval()
    correct, total = 0, 0
    all_labels = []
    all_preds  = []

    with torch.inference_mode():
        for images, labels, region_weight in val_loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            region_feats = _extract_region_feats(region_extractor, images, masks_dev)  # [B, 5, C]

            weights = torch.stack([
                region_weight['forehead'],
                region_weight['left_cheek'],
                region_weight['right_cheek'],
                region_weight['nose'],
                region_weight['jaw']
            ], dim=1).float().unsqueeze(-1).to(device)  # [B,5,1]

            # 特征未加权 / 对比实验1
            # weighted_region_feats = region_feats
            weighted_region_feats = region_feats * weights  # [B,5,C]

            output = model(weighted_region_feats)           # [B,num_classes]
            predict = output.argmax(dim=1)

            all_labels.append(labels)
            all_preds.append(predict)

            correct += (predict == labels).sum().item()
            total   += labels.size(0)

    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    all_preds  = torch.cat(all_preds, dim=0).cpu().numpy()
    acc = correct / max(1, total)
    f1  = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1, all_labels, all_preds

def train(seed, args, masks_np):
    set_seed(seed)

    # 输出文件
    outputFilename = f"SGD1_48_GCNtrainBatchsize_{args.train_batchsize}_lr{args.lr}.txt"
    fileSavePath = os.path.join(args.output, outputFilename)
    os.makedirs(args.output, exist_ok=True)
    if not os.path.exists(fileSavePath):
        with open(fileSavePath, "w"): pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 预先把 mask、邻接矩阵、对称索引搬到 device
    masks_dev = _prep_masks_on_device(masks_np, device)
    adj_np = np.load(args.adj_path).astype(np.float32)   # [N,N]
    adj_mat = torch.from_numpy(adj_np).to(device)
    adj_dict = {"spatial": adj_mat}
    left_idx = [1]
    right_idx = [2]

    # 记录总体表现
    accaverageAll = 0.0
    f1averageAll = 0.0

    # AMP & Scheduler & Scaler
    scaler = GradScaler(enabled=True)

    with open(fileSavePath, 'w') as f:
        for p in range(args.sub_num):
            p += 0
            # dataloader
            train_dataset = StO2Dataset(
                args.root_path, train=True,  test_subnum=p+1,
                weigthFilePath=args.weightFilePath, transform=transformer.transform()
            )
            test_dataset  = StO2Dataset(
                args.root_path, train=False, test_subnum=p+1,
                weigthFilePath=args.weightFilePath, transform=transformer.transform()
            )
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=args.train_batchsize,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                worker_init_fn=seed_worker
            )
            val_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.test_batchsize,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                worker_init_fn=seed_worker
            )

            # region feature extract model
            region_extractor = RegionConvSPP(in_channels=1, out_channels=64, depth=3).to(device)

            # MR_GAT model
            model = MR_GAT(
                nfeat=64, nhid=12, mat_path=args.adj_path, use_fun=True, heads=4, dropout=0.2, pooling='max'
            ).to(device)

            # Optimizer
            optimizer = optim.Adam(
                list(model.parameters()) + list(region_extractor.parameters()),
                lr=args.lr, weight_decay=1e-2
            )
            # Step = epoch * iters/epoch
            total_steps = max(1, args.train_epoch * max(1, len(train_loader)))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=args.lr * 0.01
            )

            temp_max = 0.0
            F1_max   = 0.0
            best_ckpt_path = os.path.join(args.output, f"subject{p+1}_best.pth")

            for epoch in range(1, args.train_epoch + 1):
                model.train()
                region_extractor.train()

                running_loss = 0.0
                step_in_epoch = 0

                pbar = tqdm(train_loader, leave=False, desc=f"S{p+1} E{epoch}")
                for images, labels, region_weight in pbar:

                    # Input cleaning (to prevent NaN/Inf from being transmitted to the network)
                    images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)

                    step_in_epoch += 1
                    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                    labels = labels.to(device=device, non_blocking=True)

                    # Forward (AMP)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=True):
                        # region feature
                        region_feats = _extract_region_feats(region_extractor, images, masks_dev)  # [B,5,C]

                        # weight
                        weights = torch.stack([
                            region_weight['forehead'],
                            region_weight['left_cheek'],
                            region_weight['right_cheek'],
                            region_weight['nose'],
                            region_weight['jaw']
                        ], dim=1).float().unsqueeze(-1).to(device)  # [B,5,1]

                        weighted_region_feats = region_feats * weights  # [B,5,C]

                        preds, node_feats = model(weighted_region_feats, return_node_feats=True)
                        L_cls = F.cross_entropy(preds, labels)

                        # loss function
                        L_sparse = entropy_loss_from_weights(weights
                                                             , backprop=True)
                        L_sym    = symmetry_loss(node_feats, left_idx=left_idx, right_idx=right_idx, p=1)
                        L_graph = simple_pairwise_smoothness(node_feats)
                        loss = (L_cls
                                + args.lambda_sparse * L_sparse
                                + args.lambda_sym * L_sym
                                + args.lambda_graph * L_graph)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(region_extractor.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()

                    running_loss += loss.item()
                    avg_loss = running_loss / step_in_epoch
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                # Epoch
                epoch_avg = running_loss / max(1, len(train_loader))
                print(f"[Subject {p+1:02d}] Epoch {epoch:03d} | TrainLoss {epoch_avg:.4f}")
                f.write(f"[Subject {p+1:02d}] Epoch {epoch:03d} | TrainLoss {epoch_avg:.4f}\n")

                # evaluate
                acc, f1, all_labels, all_preds = validate(
                    model, region_extractor, val_loader, device, masks_dev,
                    left_idx=left_idx, right_idx=right_idx, adj_dict=adj_dict
                )
                print(f"[Subject {p+1:02d}] Epoch {epoch:03d} | Val Acc {acc:.4f} | F1 {f1:.4f}")
                f.write(f"[Subject {p+1:02d}] Epoch {epoch:03d} | Val Acc {acc:.4f} | F1 {f1:.4f}\n")

                # save label and predict
                f.write(f"labels={all_labels.tolist()}\n")
                f.write(f"preds={all_preds.tolist()}\n")

                # save best result
                if acc > temp_max:
                    temp_max = acc
                    F1_max   = f1
                    # save best weight
                    torch.save({
                        "subject": p+1,
                        "epoch": epoch,
                        "acc": acc,
                        "f1": f1,
                        "model": model.state_dict(),
                        "region_extractor": region_extractor.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict()
                    }, best_ckpt_path)

            print(f"第{p + 1}个被试最大Acc={temp_max:.4f} | 最大F1={F1_max:.4f}")
            f.write(f"[Subject {p+1:02d}] Best Acc={temp_max:.4f} | Best F1={F1_max:.4f}\n")

            accaverageAll   += temp_max
            f1averageAll += F1_max

        # print LOSO mean value
        final_average     = accaverageAll / args.sub_num
        f1_final_average  = f1averageAll / args.sub_num
        print(f"所有被试Acc均值={final_average:.4f} | F1均值={f1_final_average:.4f}")
        f.write(f"[LOSO] Mean Acc={final_average:.4f} | Mean F1={f1_final_average:.4f}\n")


if __name__ == "__main__":
    args = parser.parse_args()

    # 区域定义以及掩码定义（保持与你一致）
    H, W = 481, 411
    mask_forehead    = generate_mask(H, W, (20, 125),  (135, 308))
    mask_left_cheek  = generate_mask(H, W, (208, 51),  (356, 146))
    mask_right_cheek = generate_mask(H, W, (208, 266), (356, 361))
    mask_nose        = generate_mask(H, W, (207, 161), (304, 251))
    mask_jaw         = generate_mask(H, W, (406, 146), (446, 266))

    masks = {
        "forehead":    mask_forehead,
        "left_cheek":  mask_left_cheek,
        "right_cheek": mask_right_cheek,
        "nose":        mask_nose,
        "jaw":         mask_jaw
    }

    print(args)
    train(seed=42, args=args, masks_np=masks)
