import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of StO2 lie classification")

parser.add_argument("--root_path", type=str, default=r"D:\Users\12150\PycharmProjects\data-sub1-sub14-all\hd5",
                    help="StO2数据根目录")
parser.add_argument("--num_classes", type=int, default=2, help="the numbers of the StO2 image classification task!")
parser.add_argument("--sub_num", type=int, default=14, help="被试的数目")
parser.add_argument("--train_epoch", type=int, default=1, help="训练的周期")
parser.add_argument("--lr", type=int, default=0.001, help="学习率")
parser.add_argument("--train_batchsize", type=int, default=16, help="训练的batchsize大小")
parser.add_argument("--test_batchsize", type=int, default=11, help="测试的batchsize大小")
parser.add_argument("--output", type=str, default=r"D:\Users\12150\PycharmProjects\DeepStO2\tools\results", help="用于存放结果的路径")
