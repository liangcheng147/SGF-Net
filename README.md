# Usage Guide

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/liangcheng147/SGF-Net.git
cd SGF-Net
```

### 2. Create and Activate Virtual Environment

```bash
conda create -n sgf_net python=3.9
conda activate sgf_net
```

### 3. Install Dependencies

Install PyTorch and torchvision (with CUDA support)
It is recommended to use PyTorch 2.0.0 or higher

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Store the datasets in the `data/` directory:

1. **Download ForenSynths Dataset**:
   Download as described at https://github.com/PeterWang512/CNNDetection.

2. **Download GANGen Dataset**:
   Download as described at https://github.com/chuangchuangtan/GANGen-Detection.

3. **Download diffusion_datasets Dataset**:
   Download as described at https://github.com/WisconsinAIVision/UniversalFakeDetect for the diffusion_datasets dataset.

### Dataset Directory Structure

The `data/` directory should have the following structure:

```
data
├── diffusion_datasets
│   ├── dalle
│   │   └── 1_fake
│   ├── glide_100_10
│   │   └── 1_fake
│   ├── glide_100_27
│   │   └── 1_fake
│   ├── glide_50_27
│   │   └── 1_fake
│   ├── guided
│   │   └── 1_fake
│   ├── imagenet
│   │   └── 0_real
│   ├── laion
│   │   └── 0_real
│   ├── ldm_100
│   │   └── 1_fake
│   ├── ldm_200
│   │   └── 1_fake
│   └── ldm_200_cfg
│       └── 1_fake
├── ForenSynths
│   ├── test
│   ├── train
│   └── val
│
└── GANGen
    ├── AttGAN
    │   ├── 0_real
    │   └── 1_fake
    ├── BEGAN
    │   ├── 0_real
    │   └── 1_fake
    ├── CramerGAN
    │   ├── 0_real
    │   └── 1_fake
    ├── InfoMaxGAN
    │   ├── 0_real
    │   └── 1_fake
    ├── MMDGAN
    │   ├── 0_real
    │   └── 1_fake
    ├── RelGAN
    │   ├── 0_real
    │   └── 1_fake
    ├── S3GAN
    │   ├── 0_real
    │   └── 1_fake
    ├── SNGAN
    │   ├── 0_real
    │   └── 1_fake
    └── STGAN
        ├── 0_real
        └── 1_fake
```

## Model Evaluation

The pre-trained model is [here](https://github.com/liangcheng147/SGF-Net/releases/tag/Pretrainedmodel)

### Evaluate GAN Models

To evaluate the model performance on the GANGen dataset, run:

```bash
python scripts/evaluation/validation_train-progan_gangen_dwt.py
```

### Evaluate Diffusion Models

To evaluate the diffusion model performance on the diffusion_datasets dataset, run:

```bash
python scripts/evaluation/validation_train-progan_diffdataset_dwt.py
```

Results will be displayed in the terminal.

## Model Training

To train the model, run the following command:

```bash
python scripts/training/train_genimage_dwt.py
```

### Training Configuration Parameters

In the training script, you can configure the training parameters by modifying the `experiment` dictionary:

```python
experiment = {
    "training_set": "forensynths",  # Dataset to use
    "categories": ['cat', 'chair', 'horse', 'car'],  # Categories to use, None means all categories
    "savpath": "results/dwt/forensics/...",  # Results save path
    "model_path": "ckpt/dwt/model_forensics_dwt_trainable_2class.pth",  # Model save path
    ...
}
```

Main configuration parameter descriptions:

- **training_set**: Dataset to use (e.g., `forensynths`)
- **categories**: Categories to use (e.g., `['cat', 'chair', 'horse', 'car']`, set to `None` for all categories)
- **model_path**: Model save path
- **savpath**: Results save path

## Code Structure

The main code structure of the project is as follows:

- **src/**: Model architecture and data processing code
- **scripts/**: Training and evaluation code
- **data/**: Dataset directory
- **ckpt/**: Trained model directory

## License

This project is licensed under the MIT license.
