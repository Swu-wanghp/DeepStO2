import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math, os
import numpy as np


# 固定边
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Param:
        in_features, out_features, bias
    Input:
        features: N x C (n = # nodes), C = in_features
        adj: adjacency matrix (N x N)
    """

    def __init__(self, in_features, out_features, mat_path, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        adj_mat = np.load(mat_path).astype(np.float32)
        self.register_buffer('adj', torch.from_numpy(adj_mat))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(self.adj.unsqueeze(0).repeat(b, 1, 1), support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# 可学习边
class LearnableGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(LearnableGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes

        # 可学习邻接矩阵（注意保持对称性）
        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes))

        # GCN 权重
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        :param x: [B, N, F] -> batch_size x num_nodes x in_channels
        :return: [B, N, F']
        """
        B, N, F = x.shape
        assert N == self.num_nodes

        # 归一化 A (softmax 保证非负+可解释)
        A_norm = F.softmax(self.A, dim=-1)  # [N, N]

        # 图卷积运算：AXW
        x = torch.matmul(A_norm, x)  # [B, N, F]
        x = self.fc(x)  # [B, N, out_channels]
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid, mat_path,  adj_mode='learnable', normalize=True, reg_lambda=0.001)
        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        self.bn1 = nn.BatchNorm1d(nhid)
        # self.gc2 = GraphConvolution(nhid, nout, mat_path)
        # self.bn2 = nn.BatchNorm1d(nout)
        # self.dropout = dropout
        self.fc = nn.Linear(nhid, 2)

    def forward(self, x, return_node_feats=False):
        x = self.gc1(x)
        x = x.transpose(1, 2).contiguous()
        x = self.bn1(x).transpose(1, 2).contiguous()
        x = F.relu(x)

        # x = F.dropout(x, self.dropout, training=self.training)

        # x = self.gc2(x)

        # x = x.transpose(1, 2).contiguous()
        # x = self.bn2(x).transpose(1, 2).contiguous()
        # x = F.relu(x)

        # x = F.relu(self.gc2(x))
        # x = F.dropout(x, self.dropout, training=self.training)

        # 2025.09.10 开始版本
        # out = x.mean(dim=1)
        # out = self.fc(out)
        #
        # return F.log_softmax(out, dim=1)

        node_feats = x  # ★ 池化前的节点特征

        out = node_feats.mean(dim=1)  # [B, nhid]  节点均值池化
        logits = self.fc(out)  # [B, 2]

        if return_node_feats:
            return logits, node_feats  # ★ 同时返回节点特征
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, dim)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return x


class StO2GCNNet(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        mat_path = os.path.join(
            'assets',
            '{}.npy'.format(opt['dataset'])
        )
        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path))

        self._init_weight()

    def forward(self, x):

        forehead, leftcheek, rigthcheek = x[0], x[1], x[2]

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    import yaml

    path = "D:\论文\StO2数据\最后mat数据\DeepStO2\model"
    savepath = os.path.join(path, "adj.npy")
    x = torch.randn((16, 3, 2))  # (b, n, c)
    model = GCN(nfeat=2, nhid=12, nout=1, mat_path=savepath)
    out = model(x)
    print(out.shape)
