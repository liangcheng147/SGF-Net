#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root)

os.chdir(project_root)

from torchvision import transforms
import torch

from src.utils import (
    evaluate_model,
    get_loaders,
    SupConLoss,
)
from src.data import get_dual_branch_transforms_2
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score
import torch.nn as nn
from src.models.models import Model
from src.models.dual_branch_2 import RINEWithDWTBranch
from src.models.freq_domain import LightweightMLP
from src.models.freq_domain.freq_domain_branch_2 import FreqDomainBranch2

device = "cuda:0"

experiment = {
    "training_set": "diffusion_datasets",
    "backbone": ["ViT-L/14", 1024],
    "factor": 0.2,
    "nproj": 4,
    "proj_dim": 1024,
    "model_path": "/home/yons/artificial/rine-main/ckpt/dwt/model_forensics_dwt_trainable_4class.pth",
    "batch_size": 80,
    "with_dwt_branch": True,
    "visualization": False,
}


def get_diffusion_dwt_model(device):
    rine_model = Model(
        backbone=experiment["backbone"],
        nproj=experiment["nproj"],
        proj_dim=experiment["proj_dim"],
        device=device,
    )

    dwt_branch = FreqDomainBranch2()

    classifier = LightweightMLP(input_dim=1024)

    model = RINEWithDWTBranch(rine_model, dwt_branch, classifier)

    ckpt_path = experiment["model_path"]
    print(f"加载模型: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)

    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'], strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    return model


def main():
    dataset_name = experiment["training_set"]

    transforms_train, transforms_test = get_dual_branch_transforms_2()
    transforms_val = transforms_test

    _, _, test = get_loaders(
        experiment,
        transforms_train,
        transforms_val,
        transforms_test,
        workers=12,
    )

    print(f"\n模型在{dataset_name}数据集上的评估结果：")
    model = get_diffusion_dwt_model(device)
    model.to(device)

    print(f"数据集大小：{len(test[0][1].dataset)}")

    bce_loss = nn.BCEWithLogitsLoss()
    supcon_loss = SupConLoss()
    factor = experiment["factor"]

    evaluation_root = f"/home/yons/artificial/rine-main/results/dwt/{dataset_name}"
    os.makedirs(evaluation_root, exist_ok=True)

    evaluation_results = {}

    for g, test_loader in test:
        print(f"\n评估生成器: {g}")
        val_loss, val_acc, val_ap, max_memory_used = evaluate_model(
            model, test_loader, bce_loss, supcon_loss, factor, device
        )

        print(f"验证损失: {val_loss:.4f}")
        print(f"验证准确率: {val_acc:.4f}")
        print(f"验证平均精度: {val_ap:.4f}")
        print(f"最大显存使用: {max_memory_used:.2f} GB")

        evaluation_results[g] = {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_ap": val_ap,
            "max_memory_used": max_memory_used
        }

    total_acc = 0
    total_ap = 0
    total_loss = 0
    total_memory = 0

    generator_results = {k: v for k, v in evaluation_results.items() if k != "average"}
    num_generators = len(generator_results)

    for g, results in generator_results.items():
        total_acc += results["val_acc"]
        total_ap += results["val_ap"]
        total_loss += results["val_loss"]
        total_memory += results["max_memory_used"]

    average_acc = total_acc / num_generators if num_generators > 0 else 0
    average_ap = total_ap / num_generators if num_generators > 0 else 0
    average_loss = total_loss / num_generators if num_generators > 0 else 0
    average_memory = total_memory / num_generators if num_generators > 0 else 0

    total_samples = 0
    if test:
        for g, test_loader in test:
            total_samples += len(test_loader.dataset)

    evaluation_results["average"] = {
        "val_acc": average_acc,
        "val_ap": average_ap,
        "val_loss": average_loss,
        "max_memory_used": average_memory,
        "num_generators": num_generators,
        "dataset_name": dataset_name,
        "total_samples": total_samples
    }

    json_output_path = os.path.join(evaluation_root, f"evaluation_results_{dataset_name}.json")
    with open(json_output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\n数据集平均准确率: {average_acc:.4f}")
    print(f"数据集平均AP: {average_ap:.4f}")
    print(f"数据集平均损失: {average_loss:.4f}")
    print(f"数据集平均显存使用: {average_memory:.2f} GB")
    print(f"评估结果已保存到 {json_output_path}")

    print(f"\n模型在{dataset_name}数据集上的评估完成！")
    print(f"\n评估结果已保存到：{evaluation_root}")


if __name__ == "__main__":
    main()
