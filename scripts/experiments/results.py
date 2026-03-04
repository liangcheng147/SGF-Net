import pickle
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

from src.utils import get_our_trained_model, get_generators, seed_everything, get_loaders
from src.data import TrainingDataset
from src.models.models import Hook
from src.extras import resnet50, CLIPModel


def get_hyperparam_results(ncls, flag):
    hp = {"backbone": {}, "factor": {}, "nproj": {}, "proj_dim": {}}
    experiments = os.listdir("results/grid")
    for experiment in experiments:
        with open(f"results/grid/{experiment}", "rb") as h:
            log = pickle.load(h)
            if log["epochs"] > 1 or len(log["config"]["classes"]) != ncls:
                continue
            if flag and log["config"]["backbone"][0] == "ViT-B/32":
                continue

            backbone = log["config"]["backbone"][0]
            factor = log["config"]["factor"]
            nproj = log["config"]["nproj"]
            proj_dim = log["config"]["proj_dim"]
            acc = np.mean(
                [log["results"]["test"][x]["acc"] for x in log["results"]["test"]]
            )
            ap = np.mean(
                [log["results"]["test"][x]["ap"] for x in log["results"]["test"]]
            )
            if factor in hp["factor"]:
                hp["factor"][factor].append({"acc": acc, "ap": ap})
            else:
                hp["factor"][factor] = [{"acc": acc, "ap": ap}]

            if nproj in hp["nproj"]:
                hp["nproj"][nproj].append({"acc": acc, "ap": ap})
            else:
                hp["nproj"][nproj] = [{"acc": acc, "ap": ap}]

            if proj_dim in hp["proj_dim"]:
                hp["proj_dim"][proj_dim].append({"acc": acc, "ap": ap})
            else:
                hp["proj_dim"][proj_dim] = [{"acc": acc, "ap": ap}]

            if backbone in hp["backbone"]:
                hp["backbone"][backbone].append({"acc": acc, "ap": ap})
            else:
                hp["backbone"][backbone] = [{"acc": acc, "ap": ap}]
    return hp


def get_epochs_results(ncls):
    epochs = [1, 3, 5, 10, 15]
    hp = {"epochs": {}}
    experiments = best_configs(ncls=ncls, nbest=3, nepochs=1, showtxt=False)
    for experiment in experiments:
        for epoch in epochs:
            filename = f'{experiment["config"]["savpath"].replace("grid", "epochs" if epoch > 1 else "grid")}_{epoch}.pickle'
            with open(filename, "rb") as h:
                log = pickle.load(h)
            acc = np.mean(
                [log["results"]["test"][x]["acc"] for x in log["results"]["test"]]
            )
            ap = np.mean(
                [log["results"]["test"][x]["ap"] for x in log["results"]["test"]]
            )
            if epoch in hp["epochs"]:
                hp["epochs"][epoch].append({"acc": acc, "ap": ap})
            else:
                hp["epochs"][epoch] = [{"acc": acc, "ap": ap}]
    return hp


def plot_hyperparams(hp, ncls, k, ylims, text, dpi):
    fig, ax = plt.subplots()
    r = np.arange(len(hp[k]))
    K = list(hp[k].keys())
    K.sort()
    width = 0.15
    acc = []
    ap = []
    for k_ in K:
        acc.append([hp[k][k_][i]["acc"] for i in range(len(hp[k][k_]))])
        ap.append([hp[k][k_][i]["ap"] for i in range(len(hp[k][k_]))])
    bp1 = ax.boxplot(
        ap,
        notch=False,
        positions=r,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="gold"),
    )
    bp2 = ax.boxplot(
        acc,
        notch=False,
        positions=r + width,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="olive"),
    )
    fs = 14
    plt.title(f"{ncls}-class", fontsize=fs)
    plt.ylim(ylims)
    if k == "backbone":
        plt.xlim(-0.5, r[-1] + 2 * width)
    else:
        plt.xlim(-1.0, r[-1] + 2 * width)
    plt.yticks(np.arange(ylims[0], ylims[1], 0.05), fontsize=fs)
    ax.grid(axis="y")
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]], ["AP", "ACC"], loc="lower left", fontsize=fs
    )

    plt.xticks(r + 0.5 * width, K, fontsize=fs)

    plt.xlabel(text, fontsize=fs + 2)
    plt.savefig(f"results/figs/{k}_ncls_{ncls}.jpg", dpi=dpi)


def plot_epochs(hp, ncls, k, ylims, text, dpi):
    fig, ax = plt.subplots()
    r = np.arange(len(hp[k]))
    K = list(hp[k].keys())
    K.sort()
    acc_max = []
    ap_max = []
    acc_avg = []
    ap_avg = []
    for k_ in K:
        acc_max.append(np.max([hp[k][k_][i]["acc"] for i in range(len(hp[k][k_]))]))
        ap_max.append(np.max([hp[k][k_][i]["ap"] for i in range(len(hp[k][k_]))]))
        acc_avg.append(np.mean([hp[k][k_][i]["acc"] for i in range(len(hp[k][k_]))]))
        ap_avg.append(np.mean([hp[k][k_][i]["ap"] for i in range(len(hp[k][k_]))]))

    fs = 14
    plt.title(f"{ncls}-class", fontsize=fs)
    plt.plot(r, acc_max, "o-", linewidth=2, label="ACC (max)", color="olive")
    plt.plot(r, ap_max, "d-", linewidth=2, label="AP (max)", color="gold")
    plt.plot(r, acc_avg, "--", linewidth=2, label="ACC (avg)", color="olive")
    plt.plot(r, ap_avg, "--", linewidth=2, label="AP (avg)", color="gold")
    plt.ylim(ylims)
    plt.yticks(np.arange(ylims[0], ylims[1], 0.05), fontsize=fs)
    ax.grid()
    ax.legend(fontsize=fs)
    plt.xticks(r, K, fontsize=fs)
    plt.xlabel(text, fontsize=fs + 2)
    plt.savefig(f"results/figs/{k}_ncls_{ncls}.jpg", dpi=dpi)


def plot_importance(importance, dpi):
    fig, ax = plt.subplots(figsize=(14, 3))
    fontsize = 15

    width = 0.25
    bottom = 0.2
    c = 0
    colors = ["gold", "olive", "peru"]
    for ncls in importance:
        df = pd.Series(np.argmax(importance[ncls], axis=0)).value_counts().sort_index()
        df.index += 1
        y = [df[x] / df.sum() for x in df.index]
        plt.bar(
            df.index + c * width, y, width=width, color=colors[c], label=f"{ncls}-class"
        )
        c += 1
    plt.yticks(fontsize=fontsize)
    plt.ylim([0, 0.1])
    plt.xticks(np.arange(1, 25) + width, np.arange(1, 25), fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xlabel("Transformer block", fontsize=fontsize)
    fig.subplots_adjust(bottom=bottom)
    plt.savefig(f"results/figs/importance.jpg", dpi=dpi)


def best_configs(ncls, nbest, nepochs=None, showtxt=False):
    base_dir = "results/diffusion" if ncls == "ldm" else "results/grid"
    experiments = os.listdir(base_dir)
    epochs = []
    config = []
    results = []
    for experiment in experiments:
        with open(f"{base_dir}/{experiment}", "rb") as h:
            log = pickle.load(h)
            if nepochs is not None and log["epochs"] != nepochs:
                continue
            if ncls == "ldm" or len(log["config"]["classes"]) == ncls:
                epochs.append(log["epochs"])
                config.append(log["config"])
                results.append(log["results"]["test"])

    index = np.argsort(
        [
            np.mean([x[y]["acc"] for y in x]) + np.mean([x[y]["ap"] for y in x])
            for x in results
        ]
    )[::-1][:nbest]
    best = [
        {"config": config[i], "results": results[i], "epochs": epochs[i]} for i in index
    ]

    if showtxt:
        print(json.dumps(best[0], indent=2))
        print(
            f"AVG acc: {100*np.mean([best[0]['results'][y]['acc'] for y in best[0]['results']]):1.1f}"
        )
        print(
            f"AVG ap: {100*np.mean([best[0]['results'][y]['ap'] for y in best[0]['results']]):1.1f}"
        )
    return best


def ablations(ncls, generators):
    """
    执行消融实验分析
    
    该函数分析模型各组件(对比学习、alpha参数、中间层特征)对整体性能的贡献。
    
    参数:
        ncls (int): 分类任务类别数 (1, 2, 或 4)
        generators (list): 生成器列表
        
    返回:
        tuple: (accs, aps) - 准确率和平均精度列表
    """
    accs, aps = [], []
    for without in ["contrastive", "alpha", "intermediate"]:
        with open(f"results/ablations/ncls_{ncls}_{without}.pickle", "rb") as h:
            log = pickle.load(h)
            avg_acc = np.mean(
                [
                    log["results"]["test"][y]["acc"]
                    for y in log["results"]["test"]
                    if y in generators
                ]
            )
            avg_ap = np.mean(
                [
                    log["results"]["test"][y]["ap"]
                    for y in log["results"]["test"]
                    if y in generators
                ]
            )
            print(
                f"[w/o {without}] avg acc: {100*avg_acc:1.1f} - avg ap: {100*avg_ap:1.1f}"
            )
            accs.append(avg_acc)
            aps.append(avg_ap)

    log = best_configs(ncls=ncls, nbest=1, nepochs=1, showtxt=False)[0]["results"]
    avg_acc = np.mean([log[y]["acc"] for y in log if y in generators])
    avg_ap = np.mean([log[y]["ap"] for y in log if y in generators])
    print(f"[full] avg acc: {100*avg_acc:1.1f} - avg ap: {100*avg_ap:1.1f}")
    accs.append(avg_acc)
    aps.append(avg_ap)
    return accs, aps


def print_ablations(ncls_list, generators):
    """
    打印消融实验结果
    
    该函数对多个分类任务执行消融实验，并打印格式化的结果。
    
    参数:
        ncls_list (list): 分类任务类别数列表 [1, 2, 4]
        generators (list): 生成器列表
        
    返回:
        None: 打印消融实验结果
    """
    accs, aps = [], []
    for ncls in ncls_list:
        print(f"{ncls}-class")
        acc, ap = ablations(ncls, generators)
        accs.append(acc)
        aps.append(ap)
        print()
    print("AVG")
    print("ACC\t AP")
    r = np.concatenate(
        (np.mean(accs, axis=0, keepdims=True), np.mean(aps, axis=0, keepdims=True))
    )
    print(np.round(100 * r.T, 1))


def plot_hyperparams_vs_model_size():
    """
    绘制模型大小与性能关系图
    
    该函数分析不同投影头配置(数量和维度)导致的模型大小变化对性能的影响。
    
    参数:
        None
        
    返回:
        None: 结果保存为图像文件
    """
    params = {
        (1, 128): 183937,
        (1, 256): 466177,
        (1, 512): 1325569,
        (1, 1024): 4224001,
        (2, 128): 216961,
        (2, 256): 597761,
        (2, 512): 1850881,
        (2, 1024): 6323201,
        (4, 128): 283009,
        (4, 256): 860929,
        (4, 512): 2901505,
        (4, 1024): 10521601,
    }

    experiments = os.listdir("results/grid")
    accs = {}
    for experiment in experiments:
        with open(f"results/grid/{experiment}", "rb") as h:
            log = pickle.load(h)
            if log["config"]["backbone"][0] == "ViT-B/32":
                continue
            nproj = log["config"]["nproj"]
            proj_dim = log["config"]["proj_dim"]
            acc = np.mean(
                [log["results"]["test"][x]["acc"] for x in log["results"]["test"]]
            )
            if (nproj, proj_dim) in accs:
                accs[(nproj, proj_dim)].append(acc)
            else:
                accs[(nproj, proj_dim)] = [acc]

    fig, ax = plt.subplots(figsize=(14, 5))
    bp1 = ax.boxplot(
        [accs[x] for x in sorted(list(accs.keys())) if x[0] == 1],
        positions=np.arange(1, 5),
        patch_artist=True,
        boxprops=dict(facecolor="gold"),
    )
    bp2 = ax.boxplot(
        [accs[x] for x in sorted(list(accs.keys())) if x[0] == 2],
        positions=np.arange(5, 9),
        patch_artist=True,
        boxprops=dict(facecolor="olive"),
    )
    bp3 = ax.boxplot(
        [accs[x] for x in sorted(list(accs.keys())) if x[0] == 4],
        positions=np.arange(9, 13),
        patch_artist=True,
        boxprops=dict(facecolor="peru"),
    )
    plt.xticks(
        np.arange(1, 13),
        [f"{params[x]/10**6:1.2f}M\n{x}" for x in sorted(list(accs.keys()))],
        fontsize=12,
    )
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
        ["q=1", "q=2", "q=4"],
        loc="lower left",
        fontsize=12,
    )
    plt.ylabel("ACC", fontsize=12)
    plt.yticks(fontsize=12)
    ax.grid(axis="y")
    plt.savefig("results/figs/accuracy_vs_size.png")


class WangModelFeatures(nn.Module):
    """
    Wang模型特征提取器
    
    该类封装了Wang等人提出的模型，用于提取模型特征以进行特征空间分析。
    模型从预训练权重中加载参数，并使用Hook机制提取中间层特征。
    """
    
    def __init__(self):
        """
        初始化Wang模型特征提取器
        
        加载预训练的ResNet50模型，并设置Hook以提取avgpool层的特征。
        """
        super().__init__()
        self.model = resnet50(num_classes=1)
        model_path = "competitive/wang_blur_jpg_prob0.1.pth"
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict["model"])
        self.hook = [
            Hook(name, module)
            for name, module in self.model.named_modules()
            if "avgpool" in name
        ][0]

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入图像张量
            
        返回:
            torch.Tensor: avgpool层的特征输出
        """
        self.model(x)
        return self.hook.output


def get_trained_model(name, device):
    """
    获取训练好的模型
    
    根据模型名称加载对应的预训练模型，并将其移动到指定设备。
    
    参数:
        name (str): 模型名称 ("ours", "wang", 或 "ufd")
        device (str): 设备类型 ("cuda" 或 "cpu")
        
    返回:
        torch.nn.Module: 加载的模型
    """
    if name == "ours":
        model = get_our_trained_model(ncls=4, device=device)
    elif name == "wang":
        model = WangModelFeatures()
    elif name == "ufd":
        model = CLIPModel()
        model_path = "competitive/fc_weights.pth"
        state_dict = torch.load(model_path, map_location=device)
        model.fc.load_state_dict(state_dict)
    model.to(device)
    return model


def get_feaure_space(model, batch_size, max_samples_per_gen, device, name):
    """
    提取模型特征空间
    
    该函数从不同生成器生成的图像中提取模型特征，用于后续的特征空间可视化。
    
    参数:
        model (torch.nn.Module): 特征提取模型
        batch_size (int): 批处理大小
        max_samples_per_gen (int): 每个生成器最大样本数
        device (str): 设备类型
        name (str): 模型名称
        
    返回:
        tuple: (DATA, LABELS, GENS) - 特征数据、标签和生成器名称
    """
    seed_everything(0)
    
    # 实验配置
    experiment = {
        "training_set": "genimage",  # 使用GenImage数据集
        "batch_size": batch_size,  # 批次大小
    }
    
    # 获取数据变换
    transforms_train, transforms_val, transforms_test = get_loaders.__globals__['get_transforms']()
    
    # 根据模型名称选择适当的变换
    if name == "wang":
        # 对于wang模型，使用ImageNet归一化
        in_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transforms_test = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                in_norm,
            ]
        )
    # 否则使用CLIP归一化（已经在get_transforms中设置）
    
    # 使用标准方法获取数据加载器
    _, _, test = get_loaders(
        experiment,
        transforms_train,
        transforms_val,
        transforms_test,
        workers=12,  # 数据加载工作进程数
    )
    
    # 过滤只包含"gan"或"diffusion"的生成器
    test = [(g, loader) for g, loader in test if "gan" in g or "diffusion" in g]
    DATA = []
    LABELS = []
    GENS = []
    for g, loader in test:
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                images, labels = data
                if name == "ufd":
                    outputs = model(images.to(device), return_feature=True)
                elif name == "ours":
                    outputs = model(images.to(device))[1]
                elif name == "wang":
                    outputs = model(images.to(device)).squeeze()
                DATA.append(outputs.cpu().numpy())
                LABELS.extend(labels.numpy().tolist())
                GENS.extend([g] * labels.shape[0])
                if batch_size * (i + 1) >= max_samples_per_gen:
                    break
    return DATA, LABELS, GENS


def plot_2d_feature_space(data_2d, labels, gens, filename, dpi):
    """
    绘制二维特征空间图
    
    该函数使用t-SNE降维后的特征数据绘制二维散点图，展示不同类型图像的特征分布。
    
    参数:
        data_2d (numpy.ndarray): 二维特征数据
        labels (list): 图像标签 (0: 真实图像, 1: 生成图像)
        gens (list): 生成器名称列表
        filename (str): 保存文件名
        dpi (int): 图像分辨率
        
    返回:
        None: 结果保存为图像文件
    """
    plt.figure(figsize=(6, 3))
    labels = np.array(labels)
    model_type = np.array(["gan" if "gan" in x else "diffusion" for x in gens])
    fake_gan = np.where(np.logical_and(labels == 1, model_type == "gan"))
    fake_diffusion = np.where(np.logical_and(labels == 1, model_type == "diffusion"))
    real = np.where(labels == 0)
    s = 15
    plt.scatter(
        data_2d[fake_gan, 0],
        data_2d[fake_gan, 1],
        s=s,
        facecolors="none",
        edgecolors="r",
        marker="o",
        label="GAN",
    )
    plt.scatter(
        data_2d[fake_diffusion, 0],
        data_2d[fake_diffusion, 1],
        s=s,
        facecolors="none",
        edgecolors="brown",
        marker="^",
        label="Diffusion",
    )
    plt.scatter(
        data_2d[real, 0],
        data_2d[real, 1],
        s=s,
        facecolors="none",
        edgecolors="b",
        marker="d",
        label="real",
    )
    plt.legend(loc="upper left", fontsize=12)
    plt.xlim(-115, 110)
    plt.ylim([-125, 160])
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.close()


if __name__ == "__main__":
    # 基本配置
    ncls_list = [1, 2, 4]  # 分类任务类别数列表
    dpi = 400  # 图像分辨率

    # t-SNE特征空间可视化
    device = "cuda:0"
    for name in ["wang", "ufd", "ours"]:
        print(f"t-SNE feature space for {name}")
        if os.path.exists(f"results/figs/data_2d_{name}.pickle"):
            with open(f"results/figs/data_2d_{name}.pickle", "rb") as h:
                data_2d, labels_, gens_ = pickle.load(h)
        else:
            model = get_trained_model(name, device)
            data_, labels_, gens_ = get_feaure_space(
                model=model,
                batch_size=100,
                max_samples_per_gen=500,
                device=device,
                name=name,
            )
            data_2d = TSNE(
                n_components=2,
                learning_rate=10,
                init="pca",
                perplexity=30,
                n_iter=5000,
                random_state=0,
            ).fit_transform(np.concatenate(data_, axis=0))
            with open(f"results/figs/data_2d_{name}.pickle", "wb") as h:
                pickle.dump([data_2d, labels_, gens_], h)

        plot_2d_feature_space(
            data_2d,
            labels_,
            gens_,
            filename=f"results/figs/feature_space_{name}.png",
            dpi=dpi,
        )

    # 最佳模型配置和结果
    for ncls in ncls_list + ["ldm"]:
        setting = ncls.upper() if ncls == "ldm" else f"{ncls}-class"
        print(f"\n|=== Best config & results for {setting} training ===|")
        _ = best_configs(ncls=ncls, nbest=1, nepochs=None, showtxt=True)

    # 消融实验分析
    ## 所有生成器
    print("\n|=== Ablations (all generators) ===|")
    print_ablations(ncls_list, get_generators())
    ## 非GAN生成器
    print("\n|=== Ablations (non-GAN generators) ===|")
    nongan = [x for x in get_generators() if "gan" not in x]
    print_ablations(ncls_list, nongan)

    # 超参数分析
    plot_hyperparams_vs_model_size()

    for ncls in ncls_list:
        hp = get_hyperparam_results(ncls=ncls, flag=False)
        ## 骨干网络分析
        plot_hyperparams(
            hp=hp, ncls=ncls, k="backbone", ylims=[0.65, 1.0], text="encoder", dpi=dpi
        )

        hp = get_hyperparam_results(ncls=ncls, flag=True)
        ## 因子分析
        plot_hyperparams(
            hp=hp, ncls=ncls, k="factor", ylims=[0.75, 1.0], text=r"$\xi$", dpi=dpi
        )

    # 训练轮次与准确率关系
    for ncls in ncls_list:
        hp = get_epochs_results(ncls=ncls)
        plot_epochs(
            hp=hp, ncls=ncls, k="epochs", ylims=[0.73, 1.0], text="epochs", dpi=dpi
        )

    # alpha参数在处理阶段的分布 (TIE)
    print("\nDistribution of alpha across processing stages...")
    print("[Figure]\n")
    importance = {}
    for ncls in ncls_list:
        importance[ncls] = (
            torch.softmax(
                torch.load(f"ckpt/model_{ncls}class_trainable.pth")["alpha"], dim=1
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    plot_importance(importance, dpi=dpi)

    # 1、2、4类模型的可训练参数数量
    print("Model sizes")
    for ncls in ncls_list:
        state_dict = torch.load(f"ckpt/model_{ncls}class_trainable.pth")
        num_params = sum([np.prod(state_dict[x].shape) for x in state_dict.keys()])
        print(f"{ncls}-class, trainable params: {num_params:,}")

    # 对扰动的鲁棒性
    print("\nRobustness to perturbations")
    for ncls in ncls_list:
        print(f"{ncls}-class")
        latex_acc_all = []
        latex_ap_all = []
        for perturb in ["blur", "crop", "compress", "noise", "combined"]:
            with open(f"results/perturbations/{perturb}_{ncls}class.pickle", "rb") as h:
                log = pickle.load(h)
                avg_acc = np.mean([log[y]["acc"] for y in log])
                avg_ap = np.mean([log[y]["ap"] for y in log])

                latext_acc = [f"{log[y]['acc']*100:1.1f}" for y in log]
                latext_acc = (
                    [perturb]
                    + latext_acc[:10]
                    + [""]
                    + latext_acc[10:16]
                    + [""]
                    + latext_acc[16:]
                    + [f"{avg_acc*100:1.1f}"]
                )
                latext_ap = [f"{log[y]['ap']*100:1.1f}" for y in log]
                latext_ap = (
                    [perturb]
                    + latext_ap[:10]
                    + [""]
                    + latext_ap[10:16]
                    + [""]
                    + latext_ap[16:]
                    + [f"{avg_ap*100:1.1f}"]
                )
                latex_acc_all.append("&".join(latext_acc) + "\\\\")
                latex_ap_all.append("&".join(latext_ap) + "\\\\")
                # print(json.dumps(log, indent=2))
        print("ACC")
        for r in latex_acc_all:
            print(r)
        print("AP")
        for r in latex_ap_all:
            print(r)

    # 训练数据规模影响
    print("\nTraining data size effect")
    mapping = {
        "progan": "ProGAN",
        "stylegan": "StyleGAN",
        "stylegan2": "StyleGAN2",
        "biggan": "BigGAN",
        "cyclegan": "CycleGAN",
        "stargan": "StarGAN",
        "gaugan": "GauGAN",
        "deepfake": "Deepfake",
        "seeingdark": "SITD",
        "san": "SAN",
        "crn": "CRN",
        "imle": "IMLE",
        "diffusion_datasets/guided": "Guided",
        "diffusion_datasets/ldm_200": "LDM200",
        "diffusion_datasets/ldm_200_cfg": "LDM200CFG",
        "diffusion_datasets/ldm_100": "LDM100",
        "diffusion_datasets/glide_100_27": "Glide-100-27",
        "diffusion_datasets/glide_50_27": "Glide-50-27",
        "diffusion_datasets/glide_100_10": "Glide-100-10",
        "diffusion_datasets/dalle": "DALL-E",
    }

    for ncls in ncls_list:
        fig_acc, ax_acc = plt.subplots(figsize=(14, 3))
        fig_ap, ax_ap = plt.subplots(figsize=(14, 3))
        fontsize = 14
        rotation = 37
        bottom = 0.35
        width = 0.2

        for i, ds_frac in enumerate([0.2, 0.5, 0.8, 1.0]):
            files = [
                x
                for x in os.listdir("results/dataset_size")
                if f"ViT-L-14_{ncls}" in x and f"{ds_frac}.pickle" in x
            ]
            assert len(files) == 1
            filename = files[0]
            with open(f"results/dataset_size/{filename}", "rb") as h:
                log = pickle.load(h)
                acc = [log["results"]["test"][x]["acc"] for x in log["results"]["test"]]
                ap = [log["results"]["test"][x]["ap"] for x in log["results"]["test"]]

                ax_acc.bar(
                    np.arange(len(acc)) + i * width,
                    acc,
                    width=width,
                    color="green",
                    alpha=ds_frac,
                    label=f"{int(ds_frac*100)}%",
                )
                ax_ap.bar(
                    np.arange(len(ap)) + i * width,
                    ap,
                    width=width,
                    color="green",
                    alpha=ds_frac,
                    label=f"{int(ds_frac*100)}%",
                )

                avg_acc = np.mean(acc)
                avg_ap = np.mean(ap)
                ds_size = TrainingDataset(
                    split="train",
                    classes=log["config"]["classes"],
                    transforms=None,
                    ds_frac=ds_frac,
                ).__len__()
                print(
                    f"[{ncls}-class | frac={ds_frac} | size={ds_size:,}] AVG ACC {avg_acc*100:1.1f} / AVG AP {avg_ap*100:1.1f}"
                )
        ax_acc.set_title(f"{ncls}-class", fontsize=fontsize)
        ax_acc.set_ylabel("ACC", fontsize=fontsize)
        ax_acc.set_xticks(
            np.arange(len(mapping)) + 1.5 * width,
            [mapping[x] for x in log["results"]["test"]],
            rotation=rotation,
        )
        ax_acc.set_ylim([0.4, 1.02])
        ax_acc.set_xlim([-0.5, 22])
        ax_acc.tick_params(axis="both", labelsize=fontsize)
        ax_acc.legend()
        fig_acc.subplots_adjust(bottom=bottom)
        fig_acc.savefig(f"results/figs/acc_{ncls}_class.png")

        ax_ap.set_title(f"{ncls}-class", fontsize=fontsize)
        ax_ap.set_ylabel("AP", fontsize=fontsize)
        ax_ap.set_xticks(
            np.arange(len(mapping)) + 1.5 * width,
            [mapping[x] for x in log["results"]["test"]],
            rotation=rotation,
        )
        ax_ap.set_ylim([0.4, 1.02])
        ax_ap.set_xlim([-0.5, 22])
        ax_ap.tick_params(axis="both", labelsize=fontsize)
        ax_ap.legend()
        fig_ap.subplots_adjust(bottom=bottom)
        fig_ap.savefig(f"results/figs/ap_{ncls}_class.png")