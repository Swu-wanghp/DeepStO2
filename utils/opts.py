import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of StO2 lie classification")

parser.add_argument("--root_path", type=str, default=r"/root/lanyun-fs/hd5_for_train",
                    help="StO2数据根目录")
parser.add_argument("--adj_path", type=str, default=r"/root/lanyun-fs/PoolingMethod/DeepStO2-GAT-mean/model/adj.npy", help="固定边邻接矩阵的地址")
parser.add_argument("--weightFilePath", type=str, default=r"/root/lanyun-fs/PoolingMethod/DeepStO2-GAT-mean/region_results.json", help="区域注意力权重字典的地址")
parser.add_argument("--num_classes", type=int, default=2, help="the numbers of the StO2 image classification task!")
parser.add_argument("--sub_num", type=int, default=48, help="被试的数目")
parser.add_argument("--train_epoch", type=int, default=200, help="训练的周期")
parser.add_argument("--lr", type=int, default=0.001, help="学习率")
parser.add_argument("--train_batchsize", type=int, default=32, help="训练的batchsize大小")
parser.add_argument("--test_batchsize", type=int, default=1, help="测试的batchsize大小")
parser.add_argument("--output", type=str, default=r"/root/lanyun-fs/PoolingMethod/DeepStO2-GAT-mean/tools/results", help="用于存放结果的路径")
parser.add_argument("--lambda_sparse", type=float, default=0.1,help="损失1")
parser.add_argument("--lambda_sym", type=float, default=0.1,help="损失2")
parser.add_argument("--lambda_graph", type=float, default=0.1, help="损失3")


