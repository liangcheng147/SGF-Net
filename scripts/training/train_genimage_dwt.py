#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import random
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_dual_branch_transforms_2
from src.utils import train_one_experiment
from src.models.freq_domain.freq_domain_branch_2 import FreqDomainBranch2
from src.models.dual_branch_2 import RINEWithDWTBranch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0"

workers = 12

epochss = [2]

epochs_reduce_lr = []


experiment = {
    "training_set": "forensynths",
    "backbone": ["ViT-L/14", 1024],
    "factor": 0.2,
    "nproj": 4,
    "proj_dim": 1024,
    "batch_size": 96,
    "lr": 1e-4,
    "categories": ['cat', 'chair', 'horse', 'car'],
    "savpath": "results/dwt/forensics/ViT-L-14_forensics_dwt_0.2_4_1024_64_0.0001_2class",
    "model_path": "ckpt/dwt/model_forensics_dwt_trainable_2class.pth",
    "with_dwt_branch": True,
}

os.makedirs(os.path.dirname(experiment["savpath"]), exist_ok=True)
os.makedirs(os.path.dirname(experiment["model_path"]), exist_ok=True)

transforms_train, transforms_test = get_dual_branch_transforms_2()

from src.utils.utils import (
    train_one_epoch, evaluate_model, test_model,
    seed_everything, SupConLoss, copy, json, EarlyStopping,
    get_generators, get_loaders
)
import torch.nn as nn
from src.models.models import Model
from src.models.freq_domain.lightweight_mlp import LightweightMLP

def train_one_experiment_dwt(experiment, epochss, epochs_reduce_lr, transforms_train, transforms_val, transforms_test, workers, device, store=False, ds_frac=None, patience=10, min_delta=1e-6, early_stopping_metric="acc"):
    seed_everything(0)

    train, val, test = get_loaders(
        experiment=experiment,
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        transforms_test=transforms_test,
        workers=workers,
        ds_frac=ds_frac,
    )

    if experiment.get("with_dwt_branch", False):
        rine_model = Model(
            backbone=experiment["backbone"],
            nproj=experiment["nproj"],
            proj_dim=experiment["proj_dim"],
            device=device,
        )
        dwt_branch = FreqDomainBranch2()
        classifier = LightweightMLP(input_dim=1024, hidden_dim=1024)
        model = RINEWithDWTBranch(rine_model, dwt_branch, classifier)
    else:
        model = Model(
            backbone=experiment["backbone"],
            nproj=experiment["nproj"],
            proj_dim=experiment["proj_dim"],
            device=device,
        )
    model.to(device)

    bce = nn.BCEWithLogitsLoss(reduction="sum")
    supcon = SupConLoss()

    print(json.dumps(experiment, indent=2))
    results = {"train_loss": [], "val_loss": [], "val_acc": [], "val_ap": [], "test": {}}
    rlr = 0
    training_time = 0

    max_memory_used = 0

    for name, param in model.rine_model.named_parameters():
        if "clip" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    for param in model.dwt_branch.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_params = model.parameters()

    optimizer = torch.optim.AdamW(trainable_params, lr=experiment["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochss), eta_min=1e-6)

    early_stopping = None
    if patience > 0:
        ckpt_name = experiment.get("model_path", "ckpt/model_genimage_trainable.pth")
        os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, metric=early_stopping_metric,
                                      verbose=True, model_save_path=ckpt_name, verbose_save=True)

    for epoch in range(max(epochss)):
        training_epoch_start = time.time()

        train_loss, epoch_memory = train_one_epoch(
            model=model,
            train_loader=train,
            optimizer=optimizer,
            bce_loss=bce,
            supcon_loss=supcon,
            factor=experiment["factor"],
            device=device,
            epoch=epoch,
            max_epochs=max(epochss),
            training_time=training_time
        )

        training_time += time.time() - training_epoch_start

        max_memory_used = max(max_memory_used, epoch_memory)

        val_loss, val_acc, val_ap, val_memory = evaluate_model(
            model=model,
            val_loader=val,
            bce_loss=bce,
            supcon_loss=supcon,
            factor=experiment["factor"],
            device=device
        )

        max_memory_used = max(max_memory_used, val_memory)

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_ap"].append(val_ap)

        current_memory = torch.cuda.memory_allocated(device) / 1024**3
        max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated(device) / 1024**3)

        print(f"Epoch {epoch+1}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_ap: {val_ap:.4f} | Mem: {current_memory:.2f}GB/{max_memory_used:.2f}GB")

        if early_stopping is not None:
            if early_stopping_metric == "loss":
                metric_value = val_loss
            elif early_stopping_metric == "acc":
                metric_value = val_acc
            elif early_stopping_metric == "ap":
                metric_value = val_ap
            else:
                raise ValueError(f"未知的早停指标: {early_stopping_metric}")

            early_stopping.step(metric_value, model, epoch=epoch)

            if early_stopping.should_stop():
                print(f"早停触发，恢复最佳模型权重...")
                model.load_state_dict(early_stopping.get_best_weights())
                break

        scheduler.step()

    print("\n=== 训练完成，开始测试最佳模型 ===")
    if early_stopping is not None and early_stopping.get_best_weights() is not None:
        print("加载最佳模型权重...")
        model.load_state_dict(early_stopping.get_best_weights())

    test_results, mean_acc, mean_ap = test_model(
        model=model,
        test_loaders=test,
        device=device
    )

    results["test"]["final"] = test_results

    filename = f'{experiment["savpath"]}_final.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    log = {
        "epochs": max(epochss),
        "config": experiment,
        "results": copy.deepcopy(results),
        "training_time": training_time,
        "max_memory_used": max_memory_used,
        "best_epoch": early_stopping.best_epoch if early_stopping is not None else max(epochss)-1,
        "best_metric": early_stopping.best_score if early_stopping is not None else max(results["val_acc"])
    }
    with open(filename, "w") as h:
        json.dump(log, h, indent=2)

    if store:
        print(f"模型已在验证过程中保存到: {experiment.get('model_path', 'ckpt/model_genimage_trainable.pth')}")

    print(f"\n=== 训练完成 ===")
    print(f"总训练时间: {training_time:.2f}秒")
    print(f"最大显存使用: {max_memory_used:.2f}GB")
    print(f"最佳模型测试结果 - 平均准确率: {100 * mean_acc:.1f}%, 平均AP: {100 * mean_ap:.1f}%")


train_one_experiment_dwt(
    experiment=experiment,
    epochss=epochss,
    epochs_reduce_lr=epochs_reduce_lr,
    transforms_train=transforms_train,
    transforms_val=transforms_test,
    transforms_test=transforms_test,
    workers=workers,
    device=device,
    store=True,
    patience=5,
    min_delta=1e-6,
    early_stopping_metric="acc"
)

print("\nGenImage数据集DWT双分支训练完成！")
print(f"- 模型权重已保存至: {experiment['model_path']}")
print(f"- 实验结果已保存至: {experiment['savpath']}_final.json")
