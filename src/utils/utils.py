

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import os
from io import BytesIO
import pickle
import copy
import json
import random
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import accuracy_score, average_precision_score

from src.data import EvaluationDataset, EvaluationDatasetChameleon, EvaluationDatasetForenSynths, get_dataset_config
from src.models.models import Model



def get_transforms():
    transforms_train = transforms.Compose(
        [
            transforms.Lambda(lambda img: data_augment(img)),
            transforms.Lambda(lambda img: resize_if_needed(img, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    
    transforms_test_1 = transforms.Compose(
        [
            transforms.Lambda(lambda img: resize_if_needed(img, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    
    transforms_test_2 = transforms.Compose(
        [
            transforms.Lambda(lambda img: resize_if_needed(img, 256)),
            transforms.TenCrop(224),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.PILToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(lambda x: x / 255),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    
    return transforms_train, transforms_test_1, transforms_test_2


def get_loaders(experiment, transforms_train, transforms_val, transforms_test, workers, ds_frac=None):
    training_set = experiment.get("training_set", "synthbuster")
    
    dataset_config = get_dataset_config(training_set)
    if dataset_config is None:
        raise ValueError(f"未知的数据集类型: {training_set}")
    
    generators = get_generators(training_set)
    
    categories = experiment.get("categories", None)
    
    train = None
    if dataset_config["train_class"] is not None:
        train = DataLoader(
            dataset_config["train_class"](split="train", transforms=transforms_train, categories=categories),
            batch_size=experiment["batch_size"],
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
    
    val = None
    if "val" in dataset_config["splits"] and dataset_config["train_class"] is not None:
        val = DataLoader(
            dataset_config["train_class"](split="val", transforms=transforms_val, categories=categories),
            batch_size=experiment["batch_size"],
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
    elif "test" in dataset_config["splits"] and dataset_config["train_class"] is not None:
        val = DataLoader(
            dataset_config["train_class"](split="test", transforms=transforms_val, categories=categories),
            batch_size=experiment["batch_size"],
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

    test = []
    for g in generators:
        if training_set == "chameleon":
            dataset = EvaluationDatasetChameleon(split=g, transforms=transforms_test)
        elif training_set == "forensynths":
            dataset = EvaluationDatasetForenSynths(g, transforms=transforms_test, is_test=True)
        elif training_set == "gangen":
            from src.data import EvaluationDatasetGANGen
            dataset = EvaluationDatasetGANGen(g, transforms=transforms_test)
        elif training_set == "diffusion_datasets":
            from src.data import EvaluationDatasetDiffusion
            dataset = EvaluationDatasetDiffusion(g, transforms=transforms_test)
        elif training_set == "diffusion_forensics":
            from src.data import EvaluationDatasetDiffusionForensics
            dataset = EvaluationDatasetDiffusionForensics(g, transforms=transforms_test, is_test=True)
        else:
            dataset = EvaluationDataset(g, transforms=transforms_test)
        
        test_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
        test.append((g, test_loader))
    
    return train, val, test


def train_one_epoch(model, train_loader, optimizer, bce_loss, supcon_loss, factor, device, epoch, max_epochs, training_time):
    model.train()
    epoch_loss = 0.0
    total_samples = 0
    start_time = time.time()
    max_memory_used = 0
    
    for i, data in enumerate(train_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        if isinstance(outputs, tuple):
            logits, features = outputs
        else:
            logits, features = outputs[0], outputs[1]
        
        optimizer.zero_grad()
        
        bce = bce_loss(logits, labels.float().view(-1, 1))
        
        supcon = supcon_loss(
            F.normalize(features).unsqueeze(1), labels
        )

        loss = bce + factor * supcon
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        total_samples += labels.size(0)
        
        current_memory = torch.cuda.memory_allocated(device) / 1024**3
        max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated(device) / 1024**3)
        
        elapsed_time = time.time() - start_time
        print(
            f"\r[Epoch {epoch + 1:02d}/{max_epochs:02d} | Batch {i + 1:04d}/{len(train_loader):04d} | Time {training_time + elapsed_time:1.1f}s] "
            f"loss: {loss.item():1.4f} | Mem: {current_memory:1.2f}GB/{max_memory_used:1.2f}GB",
            end="",
        )
    
    avg_loss = epoch_loss / total_samples
    return avg_loss, max_memory_used


def evaluate_model(model, val_loader, bce_loss, supcon_loss, factor, device):
    model.eval()
    val_loss = 0.0
    total_samples = 0
    y_true = []
    y_score = []
    max_memory_used = 0
    
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits, features = outputs
            else:
                logits, features = outputs[0], outputs[1]
            
            bce = bce_loss(logits, labels.float().view(-1, 1))
            supcon = supcon_loss(
                F.normalize(features).unsqueeze(1), labels
            )
            loss = bce + factor * supcon
            
            val_loss += loss.item()
            total_samples += labels.size(0)
            
            y_true.extend(labels.cpu().numpy().tolist())
            y_score.extend(
                torch.sigmoid(logits).squeeze().cpu().numpy().tolist()
            )
            
            current_memory = torch.cuda.memory_allocated(device) / 1024**3
            max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated(device) / 1024**3)
    
    avg_loss = val_loss / total_samples
    val_acc = accuracy_score(np.array(y_true), np.array(y_score) > 0.5)
    val_ap = average_precision_score(y_true, y_score)
    
    return avg_loss, val_acc, val_ap, max_memory_used


def test_model(model, test_loaders, device):
    model.eval()
    results = {}
    accs = []
    aps = []
    
    print("测试: ACC / AP")
    with torch.no_grad():
        for g, loader in test_loaders:
            y_true = []
            y_score = []
            
            for data in loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs[0]
                
                y_true.extend(labels.cpu().numpy().tolist())
                y_score.extend(
                    torch.sigmoid(logits).squeeze().cpu().numpy().tolist()
                )
            
            test_acc = accuracy_score(np.array(y_true), np.array(y_score) > 0.5)
            test_ap = average_precision_score(y_true, y_score)
            
            results[g] = {
                "acc": test_acc,
                "ap": test_ap
            }
            accs.append(test_acc)
            aps.append(test_ap)
            
            print(f"{g}: {100 * test_acc:1.1f} / {100 * test_ap:1.1f}")
    
    avg_acc = sum(accs) / len(accs) if accs else 0
    avg_ap = sum(aps) / len(aps) if aps else 0
    print(f"Mean: {100 * avg_acc:1.1f} / {100 * avg_ap:1.1f}")
    
    return results, avg_acc, avg_ap


def train_one_experiment(experiment, epochss, epochs_reduce_lr, transforms_train, transforms_val, transforms_test, workers, device, store=False, ds_frac=None, patience=10, min_delta=1e-6, early_stopping_metric="acc"):
    seed_everything(0)

    train, val, test = get_loaders(
        experiment=experiment,
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        transforms_test=transforms_test,
        workers=workers,
        ds_frac=ds_frac,
    )
    
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
    
    early_stopping = None
    if patience > 0:
        ckpt_name = experiment.get("model_path", "ckpt/model_synthbuster_trainable.pth")
        os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, metric=early_stopping_metric, 
                                      verbose=True, model_save_path=ckpt_name, verbose_save=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochss), eta_min=1e-6)
    
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
                
                print("\n早停触发，开始测试最佳模型...")
                test_results, mean_acc, mean_ap = test_model(
                    model=model,
                    test_loaders=test,
                    device=device
                )
                
                results["test"]["early_stop"] = test_results
                
                filename = f'{experiment["savpath"]}_early_stop.json'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                log = {
                    "epochs": epoch + 1,
                    "config": experiment,
                    "results": copy.deepcopy(results),
                    "training_time": training_time,
                    "max_memory_used": max_memory_used,
                    "best_epoch": early_stopping.best_epoch,
                    "best_metric": early_stopping.best_score
                }
                with open(filename, "w") as h:
                    json.dump(log, h, indent=2)
                break

        scheduler.step()

    print("\n=== 训练完成 ===")
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
        print(f"模型已在验证过程中保存到: {experiment.get('model_path', 'ckpt/model_synthbuster_trainable.pth')}")
    
    print(f"\n=== 训练完成 ===")
    print(f"总训练时间: {training_time:.2f}秒")
    print(f"最大显存使用: {max_memory_used:.2f}GB")
    print(f"最佳模型测试结果 - 平均准确率: {100 * mean_acc:.1f}%, 平均AP: {100 * mean_ap:.1f}%")


def get_our_trained_model(ncls, device):
    if ncls == 1:
        nproj = 4
        proj_dim = 1024
    elif ncls == 2:
        nproj = 4
        proj_dim = 128
    elif ncls == 4:
        nproj = 2
        proj_dim = 1024
    elif ncls in ["ldm"]:
        nproj = 4
        proj_dim = 1024

    model = Model(
        backbone=("ViT-L/14", 1024),
        nproj=nproj,
        proj_dim=proj_dim,
        device=device,
    )
    ckpt_path = "ckpt/model_synthbuster_trainable.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    for name in state_dict:
        exec(
            f'model.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])'
        )
    return model


def get_generators(data="synthbuster"):
    dataset_config = get_dataset_config(data)
    if dataset_config is None:
        return []
    
    if data == "synthbuster":
        synthbuster_dir = "data/synthbuster/"
        if os.path.exists(synthbuster_dir):
            generators = [d for d in os.listdir(synthbuster_dir) if os.path.isdir(os.path.join(synthbuster_dir, d))]
            return generators
        else:
            return []
    elif data == "chameleon":
        chameleon_dir = "data/Chameleon/"
        if os.path.exists(chameleon_dir):
            generators = [d for d in os.listdir(chameleon_dir) if os.path.isdir(os.path.join(chameleon_dir, d))]
            return generators
        else:
            return []
    elif data == "forensynths":
        forensynths_test_dir = "data/ForenSynths/test/"
        if os.path.exists(forensynths_test_dir):
            generators = [d for d in os.listdir(forensynths_test_dir) if os.path.isdir(os.path.join(forensynths_test_dir, d))]
            return generators
        else:
            return []
    elif data == "gangen":
        gangen_dir = "data/GANGen/"
        if os.path.exists(gangen_dir):
            generators = [d for d in os.listdir(gangen_dir) if os.path.isdir(os.path.join(gangen_dir, d))]
            return generators
        else:
            return []
    elif data == "diffusion_forensics":
        test_dir = "data/DiffusionForensics/test/"
        if os.path.exists(test_dir):
            generators = []
            for base_dataset in os.listdir(test_dir):
                base_dataset_dir = os.path.join(test_dir, base_dataset)
                if os.path.isdir(base_dataset_dir):
                    for gen in os.listdir(base_dataset_dir):
                        gen_dir = os.path.join(base_dataset_dir, gen)
                        if os.path.isdir(gen_dir) and gen != "real":
                            generators.append(f"{base_dataset}/{gen}")
            return generators
        else:
            return []
    else:
        data_dir = dataset_config["data_dir"]
        if os.path.exists(data_dir):
            generators = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d not in ["imagenet", "laion"]]
            return generators
        else:
            return []


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluation(model, test, device, training="synthbuster", ours=False, filename=None):
    accs = []
    aps = []
    log = {}
    print("评估: ACC / AP")
    for g, loader in test:
        model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for data in loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                if ours:
                    outputs = model(images)[0]
                else:
                    outputs = model(images)
                y_true.extend(labels.cpu().numpy().tolist())
                y_score.extend(torch.sigmoid(outputs).cpu().numpy().tolist())

        test_acc = accuracy_score(np.array(y_true), np.array(y_score) > 0.5)
        test_ap = average_precision_score(y_true, y_score)
        accs.append(test_acc)
        aps.append(test_ap)
        log[g] = {
            "acc": test_acc,
            "ap": test_ap,
        }
        print(f"{g}: acc={100 * test_acc:1.1f} / ap={100 * test_ap:1.1f}")
    print(
        f"Mean: acc={100 * sum(accs) / len(accs):1.1f} / ap={100 * sum(aps) / len(aps):1.1f}"
    )
    if filename is not None:
        with open(filename, "wb") as h:
            pickle.dump(log, h, protocol=pickle.HIGHEST_PROTOCOL)


def resize_if_needed(img, min_size=256):
    width, height = img.size
    
    if width >= min_size and height >= min_size:
        return img
    
    scale = min_size / min(width, height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return img.resize((new_width, new_height), Image.LANCZOS)


def data_augment(img):
    img = np.array(img)

    if random.random() < 0.5:
        sig = sample_continuous([0.0, 3.0])
        gaussian_blur(img, sig)

    if random.random() < 0.5:
        method = sample_discrete(["cv2", "pil"])
        qual = sample_discrete([30, 100])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features`需要是[批次大小, 视图数量, ...]格式,"
                "至少需要3维数据"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("不能同时定义`labels`和`mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("标签数量与特征数量不匹配")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("未知模式: {}".format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6, metric="acc", verbose=False, model_save_path=None, verbose_save=False):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.verbose = verbose
        self.model_save_path = model_save_path
        self.verbose_save = verbose_save
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.best_model_weights = None
        self.best_epoch = 0
    
    def step(self, metric_value, model, epoch=0):
        if self.metric == "loss":
            score = -metric_value
        else:
            score = metric_value
        
        if self.best_score is None:
            self.best_score = score
            self.save_best_model_weights(model)
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"早停计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"触发早停!")
        else:
            self.best_score = score
            self.save_best_model_weights(model)
            self.best_epoch = epoch
            self.counter = 0
    
    def save_best_model_weights(self, model):
        self.best_model_weights = copy.deepcopy(model.state_dict())
        
        if self.model_save_path is not None:
            torch.save(
                {
                    k: model.state_dict()[k]
                    for k in model.state_dict()
                    if "clip" not in k
                },
                self.model_save_path
            )
            if self.verbose_save:
                print(f"最佳模型已保存到: {self.model_save_path}")
    
    def get_best_weights(self):
        return self.best_model_weights
    
    def should_stop(self):
        return self.early_stop
