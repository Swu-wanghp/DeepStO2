import os
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report

sys.path.append("..")

from utils import parser
from model import RegionConvSPP, MR_GAT, generate_mask
from datasets import StO2Dataset, transformer


# ============================================================
# 区域掩码
# ============================================================
@torch.no_grad()
def _prep_masks_on_device(masks_np, device):
    masks_dev = {}
    for k, m in masks_np.items():
        if isinstance(m, np.ndarray):
            t = torch.from_numpy(m)
        else:
            t = m
        masks_dev[k] = t.to(device=device, dtype=torch.float32)
    return masks_dev


@torch.no_grad()
def _extract_region_feats(region_extractor, images, masks_dev):
    feat_forehead    = region_extractor(images, masks_dev["forehead"])
    feat_left_cheek  = region_extractor(images, masks_dev["left_cheek"])
    feat_right_cheek = region_extractor(images, masks_dev["right_cheek"])
    feat_nose        = region_extractor(images, masks_dev["nose"])
    feat_jaw         = region_extractor(images, masks_dev["jaw"])

    region_feats = torch.stack(
        [feat_forehead, feat_left_cheek, feat_right_cheek, feat_nose, feat_jaw],
        dim=1
    )

    if region_feats.dim() == 4 and region_feats.size(-1) == 1:
        region_feats = region_feats.squeeze(-1)

    return region_feats


# ============================================================
# 单个被试测试
# ============================================================
@torch.no_grad()
def eval_one_subject(args, masks_np, subject_id, device):
    """
    对单个被试做测试评估
    """
    masks_dev = _prep_masks_on_device(masks_np, device)

    # ---- 数据加载 ----
    test_dataset = StO2Dataset(
        args.root_path, train=False, test_subnum=subject_id,
        weigthFilePath=args.weightFilePath, transform=transformer.transform()
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ---- 模型结构必须与训练一致 ----
    region_extractor = RegionConvSPP(in_channels=1, out_channels=64, depth=3).to(device)
    model = MR_GAT(
        nfeat=64, nhid=12, mat_path=args.adj_path,
        use_fun=True, heads=4, dropout=0.2, pooling='max'
    ).to(device)

    # ---- 加载最佳 checkpoint ----
    ckpt_path = os.path.join(args.output, f"subject{subject_id}_best.pth")
    assert os.path.exists(ckpt_path), f"[ERROR] 找不到模型: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    region_extractor.load_state_dict(ckpt["region_extractor"])

    print(f"✔ Loaded subject {subject_id} best model | Acc={ckpt['acc']:.4f}, F1={ckpt['f1']:.4f}")

    model.eval()
    region_extractor.eval()

    all_labels = []
    all_preds  = []

    for images, labels, region_weight in test_loader:
        images = torch.nan_to_num(images).to(device, dtype=torch.float32)
        labels = labels.to(device)

        region_feats = _extract_region_feats(region_extractor, images, masks_dev)

        weights = torch.stack([
            region_weight['forehead'],
            region_weight['left_cheek'],
            region_weight['right_cheek'],
            region_weight['nose'],
            region_weight['jaw']
        ], dim=1).float().unsqueeze(-1).to(device)

        weighted_region_feats = region_feats * weights
        logits = model(weighted_region_feats)
        preds = logits.argmax(dim=1)

        all_labels.append(labels)
        all_preds.append(preds)

    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds  = torch.cat(all_preds).cpu().numpy()

    acc = (all_labels == all_preds).mean()
    f1  = f1_score(all_labels, all_preds, average='weighted')

    # 保存结果
    save_file = os.path.join(args.output, f"subject{subject_id}_test_results.txt")
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds)

    with open(save_file, "w") as f:
        f.write(f"[Subject {subject_id:02d}] Test ACC={acc:.4f} | F1={f1:.4f}\n\n")
        f.write("LABELS=" + str(all_labels.tolist()) + "\n")
        f.write("PREDS=" + str(all_preds.tolist()) + "\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
        f.write("Classification Report:\n" + cr + "\n")

    print(f"[Subject {subject_id:02d}] Test ACC={acc:.4f} | F1={f1:.4f}")
    print(f"保存结果到 {save_file}")

    return acc, f1


# ============================================================
# 主函数：LOSO 全体测试
# ============================================================
def main():
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- 区域 mask（与训练一致）----
    H, W = 481, 411
    masks = {
        "forehead":    generate_mask(H, W, (20, 125),  (135, 308)),
        "left_cheek":  generate_mask(H, W, (208, 51),  (356, 146)),
        "right_cheek": generate_mask(H, W, (208, 266), (356, 361)),
        "nose":        generate_mask(H, W, (207, 161), (304, 251)),
        "jaw":         generate_mask(H, W, (406, 146), (446, 266))
    }

    sub_num = args.sub_num

    all_acc = []
    all_f1  = []

    summary_path = os.path.join(args.output, "LOSO_test_summary.txt")
    with open(summary_path, "w") as fw:
        fw.write("============ LOSO TEST SUMMARY ============\n")
        fw.write(str(args) + "\n\n")

        for sid in range(1, sub_num + 1):
            print(f"\n========== Testing Subject {sid}/{sub_num} ==========")
            acc, f1 = eval_one_subject(args, masks, sid, device)

            all_acc.append(acc)
            all_f1.append(f1)

            fw.write(f"[Subject {sid:02d}] ACC={acc:.4f} | F1={f1:.4f}\n")

        fw.write("\n============ LOSO Mean ============\n")
        fw.write(f"Mean ACC={np.mean(all_acc):.4f} | Mean F1={np.mean(all_f1):.4f}\n")

    print("\n============ LOSO DONE ============")
    print(f"Mean ACC={np.mean(all_acc):.4f} | Mean F1={np.mean(all_f1):.4f}")
    print(f"汇总写入 {summary_path}")


if __name__ == "__main__":
    main()
