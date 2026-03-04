import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from torchvision import transforms
from src.perturbation import perturbation



DATASET_REGISTRY = {}


def register_dataset(name, train_class, eval_class, data_dir, splits, description=""):
    
    DATASET_REGISTRY[name] = {
        "train_class": train_class,
        "eval_class": eval_class,
        "data_dir": data_dir,
        "splits": splits,
        "description": description
    }


def get_dataset_config(name):
    
    return DATASET_REGISTRY.get(name, None)


def list_datasets():
    """列出所有已注册的数据集
    
    返回:
        list: 数据集名称列表
    """
    return list(DATASET_REGISTRY.keys())








# 定义评估数据集类，用于模型性能测试和评估 synthbuster
class EvaluationDataset(Dataset):
    """模型评估专用数据集
    
    继承自PyTorch的Dataset基类，支持多种生成器类型的测试数据加载和评估。
    该类设计用于处理多种不同结构的数据集，包括Synthbuster和其他自定义数据集。
    
    支持的数据集类型:
    1. 基于类别组织的生成器数据(如ProGAN、StyleGAN等)
    2. Guided diffusion类型的生成器数据
    3. LDM、GLIDE、DALL-E等扩散模型的生成数据
    4. 其他生成器类型(BigGAN、StarGAN等)
    5. Synthbuster数据集(仅包含伪造图像)
    
    属性:
        real: 真实图像路径和标签的列表，标签为0
        fake: 伪造图像路径和标签的列表，标签为1
        images: 合并后的所有图像路径和标签列表
        transforms: 数据预处理变换函数
        perturb: 图像扰动类型，用于模型鲁棒性测试
        data_dir: 数据根目录路径
    """
    
    # 初始化函数，根据生成器类型加载不同格式的测试数据
    def __init__(self, generator, transforms=None, perturb=None):
        """初始化评估数据集
        
        根据不同的生成器类型，采用不同的数据加载策略，以适应不同的数据集结构。
        
        参数:
            generator (str): 生成器类型，用于确定数据加载方式
            transforms (callable, optional): 数据预处理变换函数
            perturb (str, optional): 图像扰动类型，用于模型鲁棒性测试
            
        返回:
            None
        """
        # 设置数据目录
        self.data_dir = "data"
        
        # 处理基于类别组织的生成器数据(如ProGAN、StyleGAN等)
        if generator in ["cyclegan", "progan", "stylegan", "stylegan2"]:
            # 加载每个类别下的真实图像
            self.real = [
                (f"data/test/{generator}/{y}/0_real/{x}", 0)  # 构建真实图像路径，标签为0
                for y in os.listdir(f"data/test/{generator}")  # 遍历所有类别
                for x in os.listdir(f"data/test/{generator}/{y}/0_real")  # 遍历每个类别下的真实图像
            ]
            # 加载每个类别下的伪造图像
            self.fake = [
                (f"data/test/{generator}/{y}/1_fake/{x}", 1)  # 构建伪造图像路径，标签为1
                for y in os.listdir(f"data/test/{generator}")  # 遍历所有类别
                for x in os.listdir(f"data/test/{generator}/{y}/1_fake")  # 遍历每个类别下的伪造图像
            ]
        # 处理guided diffusion类型的生成器数据
        elif "diffusion_datasets/guided" in generator:
            # 加载ImageNet真实图像作为参照
            self.real = [
                (f"data/test/diffusion_datasets/imagenet/0_real/{x}", 0)
                for x in os.listdir(f"data/test/diffusion_datasets/imagenet/0_real")
            ]
            # 加载对应生成器的伪造图像
            self.fake = [
                (f"data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"data/test/{generator}/1_fake")
            ]
        # 处理LDM、GLIDE、DALL-E等扩散模型的生成数据
        elif (
            "diffusion_datasets/ldm" in generator
            or "diffusion_datasets/glide" in generator
            or "diffusion_datasets/dalle" in generator
        ):
            # 加载LAION数据集的真实图像作为参照
            self.real = [
                (f"data/test/diffusion_datasets/laion/0_real/{x}", 0)
                for x in os.listdir(f"data/test/diffusion_datasets/laion/0_real")
            ]
            # 加载对应生成器的伪造图像
            self.fake = [
                (f"data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"data/test/{generator}/1_fake")
            ]
        # 处理其他生成器类型(BigGAN、StarGAN等)
        elif any(
            [
                x in generator
                for x in [
                    "biggan",
                    "stargan",
                    "gaugan",
                    "deepfake",
                    "seeingdark",
                    "san",
                    "crn",
                    "imle",
                ]
            ]
        ):
            # 加载该生成器对应的真实图像
            self.real = [
                (f"data/test/{generator}/0_real/{x}", 0)
                for x in os.listdir(f"data/test/{generator}/0_real")
            ]
            # 加载该生成器生成的伪造图像
            self.fake = [
                (f"data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"data/test/{generator}/1_fake")
            ]
        # Handle Synthbuster dataset
        # Synthbuster has a different structure: data/synthbuster/{generator}/
        # with fake images directly in the generator directory
        synthbuster_generators = ["dalle2", "dalle3", "glide", "stable-diffusion", "midjourney", "firefly", "vqdm"]
        if any(gen in generator for gen in synthbuster_generators):
            # For Synthbuster, fake images are directly in the generator directory
            fake_dir = os.path.join(self.data_dir, "synthbuster", generator)
            fake_files = sorted([f for f in os.listdir(fake_dir) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.fake = [(os.path.join(fake_dir, f), 1) for f in fake_files]
            
            # For Synthbuster dataset, we only need fake images, no real images needed
            self.real = []
        
        # 合并真实和伪造图像列表
        # For Synthbuster, we only have fake images
        if hasattr(self, 'fake') and hasattr(self, 'real'):
            self.images = self.real + self.fake
        else:
            self.images = []

        # 存储数据预处理变换函数和图像扰动选项
        self.transforms = transforms
        self.perturb = perturb

    # 返回评估数据集的总样本数
    def __len__(self):
        """返回评估数据集的总样本数量
        
        返回:
            int: 数据集的样本总数
        """
        return len(self.images)

    # 根据索引获取单个评估样本
    def __getitem__(self, idx):
        """根据索引获取单个评估样本(图像和标签)
        
        参数:
            idx (int or torch.Tensor): 样本索引
            
        返回:
            list: 包含处理后的图像张量和对应标签的列表 [image, target]
        """
        # 如果索引是张量类型，则转换为Python列表
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径和对应的标签
        image_path, target = self.images[idx]
        # 打开图像文件并转换为RGB格式
        image = Image.open(image_path).convert("RGB")
        # 应用数据预处理变换
        if self.transforms is not None and self.perturb is None:
            # 仅应用常规数据预处理
            image = self.transforms(image)
        elif self.transforms is not None and self.perturb is not None:
            # 50%概率应用图像扰动，50%概率应用常规数据预处理
            if random.random() < 0.5:
                image = perturbation(self.perturb)(image)  # 应用图像扰动
            else:
                image = self.transforms(image)  # 应用常规预处理
        # 返回处理后的图像和标签
        return [image, target]












# 初始化数据集注册表
def initialize_dataset_registry():
    """初始化数据集注册表，注册所有预定义的数据集"""
    

    

    
    # 注册ForenSynths数据集
    register_dataset(
        name="forensynths",
        train_class=TrainingDatasetForenSynths,  # 使用新创建的ForenSynths训练类
        eval_class=EvaluationDatasetForenSynths,  # 使用新创建的ForenSynths评估类
        data_dir="data/ForenSynths",
        splits=["train", "val", "test"],
        description="ForenSynths数据集，包含多种生成模型生成的图像"
    )
    
    # 注册GANGen数据集
    register_dataset(
        name="gangen",
        train_class=None,  # GANGen仅用于测试，不需要训练类
        eval_class=EvaluationDatasetGANGen,  # 使用新创建的GANGen评估类
        data_dir="data/GANGen",
        splits=["test"],  # GANGen仅用于测试
        description="GANGen数据集，包含多种GAN生成的图像"
    )
    

    
    # 注册diffusion_datasets数据集
    register_dataset(
        name="diffusion_datasets",
        train_class=None,  # diffusion_datasets仅用于测试，不需要训练类
        eval_class=EvaluationDatasetDiffusion,  # 使用新创建的diffusion_datasets评估类
        data_dir="data/diffusion_datasets",
        splits=["test"],  # diffusion_datasets仅用于测试
        description="diffusion_datasets数据集，包含多种扩散模型生成的图像"
    )
    


def get_dual_branch_transforms_2():
    """
    获取DWT双分支模型的数据预处理和增强的变换函数
    
    返回：
        transforms_train (callable): 训练数据的变换函数
        transforms_test (callable): 测试数据的变换函数
    """
    # 避免循环导入，在函数内部导入
    import torch.nn as nn
    from src.utils.freq_domain.preprocessing_2 import DWTPreprocessor  # 导入DWT预处理类
    from src.utils.utils import data_augment, resize_if_needed  # 导入数据增强和 resize 函数
    
    # 初始化DWT预处理类
    dwt_preprocessor = DWTPreprocessor()
    
    # 训练数据的变换：包含数据增强、尺寸调整、裁剪、翻转、转张量，但不归一化
    def dual_branch_transform_2(image):
        """
        DWT双分支模型的图像变换函数
        
        参数：
            image (PIL.Image): 输入图像
            
        返回：
            combined_image (torch.Tensor): 堆叠后的4通道图像张量
                - 第1通道：RINE分支图像
                - 第2-4通道：DWT三个高频子带
        """
        # ------------------------------
        # 1. RINE分支处理
        # ------------------------------
        rine_image = transforms.Compose([
            transforms.Lambda(lambda img: data_augment(img)),  # 应用数据增强函数
            transforms.Lambda(lambda img: resize_if_needed(img, 256)),  # 如果需要，调整图像大小
            transforms.RandomCrop(224),  # 随机裁剪到224x224
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )  # CLIP归一化
        ])(image)
        
        # ------------------------------
        # 2. DWT分支处理
        # ------------------------------
        # 2.1 图像预处理：放大到448x448，应用相同的数据增强
        dwt_image = transforms.Compose([
            transforms.Lambda(lambda img: data_augment(img)),  # 应用数据增强函数
            transforms.Resize((448, 448)),  # 放大到448x448
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
            transforms.ToTensor(),  # 转换为张量
        ])(image)
        
        # 2.2 获取三个DWT高频子带
        dwt_bands = dwt_preprocessor.preprocess(dwt_image)
        
        # 2.3 对每个高频子带进行层归一化
        normalized_bands = []
        for band in dwt_bands:
            # 确保尺寸是224x224
            if band.shape[1] != 224 or band.shape[2] != 224:
                band = transforms.Resize((224, 224))(band)
            # 使用函数式API进行层归一化，避免创建计算图
            import torch.nn.functional as F
            # 层归一化，计算图在模型内部创建，不在数据加载时
            band = F.layer_norm(band, band.shape[1:])
            normalized_bands.append(band)
        
        # ------------------------------
        # 3. 堆叠四个张量
        # ------------------------------
        # 堆叠图像：1个RINE张量 + 3个DWT高频子带张量，形状为[4, 3, 224, 224]
        combined_image = torch.cat([
            rine_image.unsqueeze(0),  # [1, 3, 224, 224] - RINE分支
            normalized_bands[0].unsqueeze(0),  # [1, 3, 224, 224] - DWT LH子带
            normalized_bands[1].unsqueeze(0),  # [1, 3, 224, 224] - DWT HL子带
            normalized_bands[2].unsqueeze(0)   # [1, 3, 224, 224] - DWT HH子带
        ], dim=0)
        
        return combined_image
    
    # 测试数据的变换：不包含数据增强，只进行必要的预处理和归一化
    def dual_branch_transform_test_2(image):
        """
        DWT双分支模型的测试图像变换函数
        
        参数：
            image (PIL.Image): 输入图像
            
        返回：
            combined_image (torch.Tensor): 堆叠后的4通道图像张量
        """
        # ------------------------------
        # 1. RINE分支处理（无数据增强）
        # ------------------------------
        rine_image = transforms.Compose([
            transforms.Lambda(lambda img: resize_if_needed(img, 256)),  # 如果需要，调整图像大小
            transforms.CenterCrop(224),  # 中心裁剪到224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )  # CLIP归一化
        ])(image)
        
        # ------------------------------
        # 2. DWT分支处理（无数据增强）
        # ------------------------------
        # 2.1 图像预处理：放大到448x448，无数据增强
        dwt_image = transforms.Compose([
            transforms.Resize((448, 448)),  # 放大到448x448
            transforms.ToTensor(),  # 转换为张量
        ])(image)
        
        # 2.2 获取三个DWT高频子带
        dwt_bands = dwt_preprocessor.preprocess(dwt_image)
        
        # 2.3 对每个高频子带进行层归一化
        normalized_bands = []
        for band in dwt_bands:
            # 确保尺寸是224x224
            if band.shape[1] != 224 or band.shape[2] != 224:
                band = transforms.Resize((224, 224))(band)
            # 层归一化
            band = nn.LayerNorm(band.shape[1:])(band)
            normalized_bands.append(band)
        
        # ------------------------------
        # 3. 堆叠四个张量
        # ------------------------------
        # 堆叠图像：1个RINE张量 + 3个DWT高频子带张量，形状为[4, 3, 224, 224]
        combined_image = torch.cat([
            rine_image.unsqueeze(0),  # [1, 3, 224, 224] - RINE分支
            normalized_bands[0].unsqueeze(0),  # [1, 3, 224, 224] - DWT LH子带
            normalized_bands[1].unsqueeze(0),  # [1, 3, 224, 224] - DWT HL子带
            normalized_bands[2].unsqueeze(0)   # [1, 3, 224, 224] - DWT HH子带
        ], dim=0)
        
        return combined_image
    
    return dual_branch_transform_2, dual_branch_transform_test_2





# 定义扩散模型数据集专用的评估数据集类
class EvaluationDatasetDiffusion(Dataset):
    """diffusion_datasets数据集专用的评估数据集
    
    继承自PyTorch的Dataset基类，用于加载和评估diffusion_datasets数据集中的测试数据。
    该类支持从diffusion_datasets数据集目录中加载测试数据。
    
    数据集结构:
    data/diffusion_datasets/{generator}/1_fake/ - 伪造图像目录
    data/diffusion_datasets/imagenet/0_real/ - ImageNet真实图像目录
    data/diffusion_datasets/laion/0_real/ - LAION真实图像目录
    
    属性:
        real: 真实图像路径和标签的列表，标签为0
        fake: 伪造图像路径和标签的列表，标签为1
        images: 合并后的所有图像路径和标签列表
        transforms: 数据预处理变换函数
        perturb: 图像扰动类型，用于模型鲁棒性测试
    """
    
    # 初始化函数，设置数据集参数
    def __init__(self, generator, transforms=None, perturb=None):
        """初始化diffusion_datasets评估数据集
        
        参数:
            generator (str): 生成器类型，用于确定数据加载方式
            transforms (callable, optional): 数据预处理变换函数
            perturb (str, optional): 图像扰动类型，用于模型鲁棒性测试
            
        返回:
            None
        """
        # diffusion_datasets数据集的基础目录
        diffusion_dir = "data/diffusion_datasets/"
        
        # 加载真实图像
        # 根据生成器类型选择不同的真实图像目录
        self.real = []
        
        # 对于guided、ldm等生成器，使用imagenet作为真实图像
        if generator in ["guided"]:
            real_dir = os.path.join(diffusion_dir, "imagenet", "0_real")
            if os.path.exists(real_dir):
                self.real = [
                    (os.path.join(real_dir, x), 0)
                    for x in os.listdir(real_dir)
                ]
        # 对于dalle等生成器，使用laion作为真实图像
        elif generator in ["dalle", "ldm_100", "ldm_200", "ldm_200_cfg"]:
            real_dir = os.path.join(diffusion_dir, "laion", "0_real")
            if os.path.exists(real_dir):
                self.real = [
                    (os.path.join(real_dir, x), 0)
                    for x in os.listdir(real_dir)
                ]
        # 对于glide系列生成器，可以根据需要选择真实图像目录
        elif "glide" in generator:
            real_dir = os.path.join(diffusion_dir, "imagenet", "0_real")
            if os.path.exists(real_dir):
                self.real = [
                    (os.path.join(real_dir, x), 0)
                    for x in os.listdir(real_dir)
                ]
        # 如果生成器是imagenet或laion，则跳过，因为它们是真实图像目录
        elif generator in ["imagenet", "laion"]:
            self.real = []
        # 对于其他生成器，默认使用imagenet作为真实图像
        else:
            real_dir = os.path.join(diffusion_dir, "imagenet", "0_real")
            if os.path.exists(real_dir):
                self.real = [
                    (os.path.join(real_dir, x), 0)
                    for x in os.listdir(real_dir)
                ]
        
        # 加载伪造图像
        fake_dir = os.path.join(diffusion_dir, generator, "1_fake")
        if os.path.exists(fake_dir):
            self.fake = [
                (os.path.join(fake_dir, x), 1)
                for x in os.listdir(fake_dir)
            ]
        else:
            self.fake = []
        
        # 合并真实和伪造图像列表
        self.images = self.real + self.fake
        
        # 存储数据预处理变换函数和图像扰动选项
        self.transforms = transforms
        self.perturb = perturb
    
    # 返回评估数据集的总样本数
    def __len__(self):
        """返回diffusion_datasets评估数据集的总样本数量
        
        返回:
            int: 数据集的样本总数
        """
        return len(self.images)
    
    # 根据索引获取单个评估样本
    def __getitem__(self, idx):
        """根据索引获取单个diffusion_datasets评估样本(图像和标签)
        
        参数:
            idx (int or torch.Tensor): 样本索引
            
        返回:
            list: 包含处理后的图像张量和对应标签的列表 [image, target]
        """
        # 如果索引是张量类型，则转换为Python列表
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径和对应的标签
        image_path, target = self.images[idx]
        # 打开图像文件并转换为RGB格式
        image = Image.open(image_path).convert("RGB")
        # 应用数据预处理变换
        if self.transforms is not None and self.perturb is None:
            # 仅应用常规数据预处理
            image = self.transforms(image)
        elif self.transforms is not None and self.perturb is not None:
            # 50%概率应用图像扰动，50%概率应用常规数据预处理
            if random.random() < 0.5:
                image = perturbation(self.perturb)(image)  # 应用图像扰动
            else:
                image = self.transforms(image)  # 应用常规预处理
        # 返回处理后的图像和标签
        return [image, target]


# 定义ForenSynths数据集专用的训练数据集类
class TrainingDatasetForenSynths(Dataset):
    """ForenSynths数据集专用的训练数据集
    
    继承自PyTorch的Dataset基类，用于加载和处理ForenSynths数据集中的训练数据。
    该类支持从ForenSynths数据集目录中加载真实图像和伪造图像，并进行数据预处理。
    
    数据集结构:
    data/ForenSynths/{split}/{category}/{0_real|1_fake}/
    
    属性:
        real: 真实图像路径和标签的列表，标签为0
        fake: 伪造图像路径和标签的列表，标签为1
        images: 合并后的所有图像路径和标签列表
        transforms: 数据预处理和增强的变换函数
    """
    
    # 初始化函数，设置数据集参数
    def __init__(self, split="train", transforms=None, categories=None):
        """初始化ForenSynths训练数据集
        
        参数:
            split (str): 数据集分割类型，可选值为'train'、'val'或'test'
            transforms (callable, optional): 数据预处理和增强的变换函数
            categories (list, optional): 要加载的类别列表，默认加载所有类别
            
        返回:
            None
        """
        # ForenSynths数据集的基础目录
        forensynths_dir = "data/ForenSynths/"
        
        # 获取所有类别
        all_categories = [d for d in os.listdir(os.path.join(forensynths_dir, split)) 
                         if os.path.isdir(os.path.join(forensynths_dir, split, d))]
        
        # 如果指定了类别，则使用指定的类别，否则使用所有类别
        self.categories = categories if categories is not None else all_categories
        
        # 收集所有真实图像的路径和标签(0表示真实图像)
        self.real = []
        # 收集所有伪造图像的路径和标签(1表示伪造图像)
        self.fake = []
        
        # 遍历所有指定的类别
        for category in self.categories:
            # 真实图像路径
            real_dir = os.path.join(forensynths_dir, split, category, "0_real")
            # 伪造图像路径
            fake_dir = os.path.join(forensynths_dir, split, category, "1_fake")
            
            # 检查目录是否存在
            if os.path.exists(real_dir):
                # 收集真实图像
                for img_file in os.listdir(real_dir):
                    img_path = os.path.join(real_dir, img_file)
                    self.real.append((img_path, 0))
            
            # 检查目录是否存在
            if os.path.exists(fake_dir):
                # 收集伪造图像
                for img_file in os.listdir(fake_dir):
                    img_path = os.path.join(fake_dir, img_file)
                    self.fake.append((img_path, 1))

        # 合并真实和伪造图像列表
        self.images = self.real + self.fake
        # 随机打乱数据集顺序，增加训练随机性
        random.shuffle(self.images)
        # 存储数据预处理和增强的变换函数
        self.transforms = transforms

    # 返回数据集的总样本数
    def __len__(self):
        """返回ForenSynths数据集的总样本数量
        
        返回:
            int: 数据集的样本总数
        """
        return len(self.images)

    # 根据索引获取单个样本
    def __getitem__(self, idx):
        """根据索引获取单个ForenSynths训练样本(图像和标签)
        
        参数:
            idx (int or torch.Tensor): 样本索引
            
        返回:
            list: 包含处理后的图像张量和对应标签的列表 [image, target]
        """
        # 如果索引是张量类型，则转换为Python列表
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径和对应的标签
        image_path, target = self.images[idx]
        # 打开图像文件并转换为RGB格式
        image = Image.open(image_path).convert("RGB")
        # 应用数据预处理和增强变换
        if self.transforms is not None:
            image = self.transforms(image)
        # 返回处理后的图像和标签
        return [image, target]

# 定义ForenSynths数据集专用的评估数据集类
class EvaluationDatasetForenSynths(Dataset):
    """ForenSynths数据集专用的评估数据集
    
    继承自PyTorch的Dataset基类，用于加载和评估ForenSynths数据集中的测试数据。
    该类支持从ForenSynths数据集目录中加载验证集和测试集数据。
    
    数据集结构:
    验证集: data/ForenSynths/val/{category}/{0_real|1_fake}/
    测试集: data/ForenSynths/test/{generator}/{0_real|1_fake}/
    
    属性:
        real: 真实图像路径和标签的列表，标签为0
        fake: 伪造图像路径和标签的列表，标签为1
        images: 合并后的所有图像路径和标签列表
        transforms: 数据预处理变换函数
        perturb: 图像扰动类型，用于模型鲁棒性测试
    """
    
    # 初始化函数，设置数据集参数
    def __init__(self, generator_or_split, transforms=None, is_test=False, perturb=None):
        """初始化ForenSynths评估数据集
        
        参数:
            generator_or_split (str): 生成器名称(测试集)或数据集分割(验证集)
            transforms (callable, optional): 数据预处理和增强的变换函数
            is_test (bool, optional): 是否为测试集，默认为False(验证集)
            perturb (str, optional): 图像扰动类型，用于模型鲁棒性测试
            
        返回:
            None
        """
        # ForenSynths数据集的基础目录
        forensynths_dir = "data/ForenSynths/"
        
        # 加载真实图像和伪造图像
        self.real = []
        self.fake = []
        
        if is_test:
            # 测试集处理：支持两种目录结构
            # 结构1: data/ForenSynths/test/{generator}/{0_real|1_fake}/
            # 结构2: data/ForenSynths/test/{generator}/{subcategory}/{0_real|1_fake}/
            
            generator_dir = os.path.join(forensynths_dir, "test", generator_or_split)
            
            # 检查生成器目录下是否直接包含0_real目录
            is_direct_structure = os.path.exists(os.path.join(generator_dir, "0_real"))
            
            if is_direct_structure:
                # 直接结构：test/{generator}/0_real/ 和 test/{generator}/1_fake/
                real_dir = os.path.join(generator_dir, "0_real")
                fake_dir = os.path.join(generator_dir, "1_fake")
                
                # 收集真实图像
                if os.path.exists(real_dir):
                    self.real = [(os.path.join(real_dir, x), 0) for x in os.listdir(real_dir)]
                else:
                    self.real = []
                
                # 收集伪造图像
                if os.path.exists(fake_dir):
                    self.fake = [(os.path.join(fake_dir, x), 1) for x in os.listdir(fake_dir)]
                else:
                    self.fake = []
            else:
                # 子类别结构：test/{generator}/{subcategory}/0_real/ 和 test/{generator}/{subcategory}/1_fake/
                # 获取所有子类别
                subcategories = [d for d in os.listdir(generator_dir) if os.path.isdir(os.path.join(generator_dir, d))]
                
                # 遍历所有子类别
                self.real = []
                self.fake = []
                for subcategory in subcategories:
                    # 真实图像路径
                    real_dir = os.path.join(generator_dir, subcategory, "0_real")
                    # 伪造图像路径
                    fake_dir = os.path.join(generator_dir, subcategory, "1_fake")
                    
                    # 检查目录是否存在
                    if os.path.exists(real_dir):
                        # 收集真实图像
                        for img_file in os.listdir(real_dir):
                            img_path = os.path.join(real_dir, img_file)
                            self.real.append((img_path, 0))
                    
                    # 检查目录是否存在
                    if os.path.exists(fake_dir):
                        # 收集伪造图像
                        for img_file in os.listdir(fake_dir):
                            img_path = os.path.join(fake_dir, img_file)
                            self.fake.append((img_path, 1))
        else:
            # 验证集：data/ForenSynths/val/{category}/{0_real|1_fake}/
            # 获取所有类别
            categories = [d for d in os.listdir(os.path.join(forensynths_dir, "val")) 
                         if os.path.isdir(os.path.join(forensynths_dir, "val", d))]
            
            # 遍历所有类别
            for category in categories:
                # 真实图像路径
                real_dir = os.path.join(forensynths_dir, "val", category, "0_real")
                # 伪造图像路径
                fake_dir = os.path.join(forensynths_dir, "val", category, "1_fake")
                
                # 检查目录是否存在
                if os.path.exists(real_dir):
                    # 收集真实图像
                    for img_file in os.listdir(real_dir):
                        img_path = os.path.join(real_dir, img_file)
                        self.real.append((img_path, 0))
                
                # 检查目录是否存在
                if os.path.exists(fake_dir):
                    # 收集伪造图像
                    for img_file in os.listdir(fake_dir):
                        img_path = os.path.join(fake_dir, img_file)
                        self.fake.append((img_path, 1))
            
            # 如果是验证集，直接返回，不需要后续处理
            self.images = self.real + self.fake
            self.transforms = transforms
            self.perturb = perturb
            return
        
        # 合并真实和伪造图像列表
        self.images = self.real + self.fake
        
        # 存储数据预处理变换函数和图像扰动选项
        self.transforms = transforms
        self.perturb = perturb

    # 返回评估数据集的总样本数
    def __len__(self):
        """返回ForenSynths评估数据集的总样本数量
        
        返回:
            int: 数据集的样本总数
        """
        return len(self.images)

    # 根据索引获取单个评估样本
    def __getitem__(self, idx):
        """根据索引获取单个ForenSynths评估样本(图像和标签)
        
        参数:
            idx (int or torch.Tensor): 样本索引
            
        返回:
            list: 包含处理后的图像张量和对应标签的列表 [image, target]
        """
        # 如果索引是张量类型，则转换为Python列表
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径和对应的标签
        image_path, target = self.images[idx]
        # 打开图像文件并转换为RGB格式
        image = Image.open(image_path).convert("RGB")
        # 应用数据预处理变换
        if self.transforms is not None and self.perturb is None:
            # 仅应用常规数据预处理
            image = self.transforms(image)
        elif self.transforms is not None and self.perturb is not None:
            # 50%概率应用图像扰动，50%概率应用常规数据预处理
            if random.random() < 0.5:
                image = perturbation(self.perturb)(image)  # 应用图像扰动
            else:
                image = self.transforms(image)  # 应用常规预处理
        # 返回处理后的图像和标签
        return [image, target]



# 定义GANGen数据集专用的评估数据集类
class EvaluationDatasetGANGen(Dataset):
    """GANGen数据集专用的评估数据集
    
    继承自PyTorch的Dataset基类，用于加载和评估GANGen数据集中的测试数据。
    该类支持从GANGen数据集目录中加载测试数据，并支持图像扰动用于鲁棒性测试。
    
    数据集结构:
    data/GANGen/{generator}/{0_real|1_fake}/
    
    属性:
        real: 真实图像路径和标签的列表，标签为0
        fake: 伪造图像路径和标签的列表，标签为1
        images: 合并后的所有图像路径和标签列表
        transforms: 数据预处理变换函数
        perturb: 图像扰动类型，用于模型鲁棒性测试
    """
    
    # 初始化函数，设置数据集参数
    def __init__(self, generator, transforms=None, perturb=None):
        """初始化GANGen评估数据集
        
        参数:
            generator (str): 生成器类型，用于确定数据加载方式
            transforms (callable, optional): 数据预处理变换函数
            perturb (str, optional): 图像扰动类型，用于模型鲁棒性测试
            
        返回:
            None
        """
        # GANGen数据集的基础目录
        gangen_dir = "data/GANGen/"
        
        # 加载真实图像
        real_dir = os.path.join(gangen_dir, generator, "0_real")
        if os.path.exists(real_dir):
            self.real = [
                (os.path.join(real_dir, x), 0)
                for x in os.listdir(real_dir)
            ]
        else:
            self.real = []
        
        # 加载伪造图像
        fake_dir = os.path.join(gangen_dir, generator, "1_fake")
        if os.path.exists(fake_dir):
            self.fake = [
                (os.path.join(fake_dir, x), 1)
                for x in os.listdir(fake_dir)
            ]
        else:
            self.fake = []
        
        # 合并真实和伪造图像列表
        self.images = self.real + self.fake
        
        # 存储数据预处理变换函数和图像扰动选项
        self.transforms = transforms
        self.perturb = perturb
    
    # 返回评估数据集的总样本数
    def __len__(self):
        """返回GANGen评估数据集的总样本数量
        
        返回:
            int: 数据集的样本总数
        """
        return len(self.images)
    
    # 根据索引获取单个评估样本
    def __getitem__(self, idx):
        """根据索引获取单个GANGen评估样本(图像和标签)
        
        参数:
            idx (int or torch.Tensor): 样本索引
            
        返回:
            list: 包含处理后的图像张量和对应标签的列表 [image, target]
        """
        # 如果索引是张量类型，则转换为Python列表
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径和对应的标签
        image_path, target = self.images[idx]
        # 打开图像文件并转换为RGB格式
        image = Image.open(image_path).convert("RGB")
        # 应用数据预处理变换
        if self.transforms is not None and self.perturb is None:
            # 仅应用常规数据预处理
            image = self.transforms(image)
        elif self.transforms is not None and self.perturb is not None:
            # 50%概率应用图像扰动，50%概率应用常规数据预处理
            if random.random() < 0.5:
                image = perturbation(self.perturb)(image)  # 应用图像扰动
            else:
                image = self.transforms(image)  # 应用常规预处理
        # 返回处理后的图像和标签
        return [image, target]







# 在模块导入时自动初始化数据集注册表
initialize_dataset_registry()