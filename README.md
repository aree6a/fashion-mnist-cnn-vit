# Enhancing Fashion-MNIST Classification Using a Hybrid CNN–Vision Transformer

A computer vision project comparing a baseline Vision Transformer (ViT) with a modified hybrid CNN–ViT architecture for Fashion-MNIST image classification.

The modified model combines convolutional feature extraction with a Vision Transformer and incorporates data augmentation, MixUp, label smoothing, and cosine learning-rate scheduling with warmup.

## Results

| Model          | Test Accuracy |  Training Time | Parameters |
| -------------- | ------------: | -------------: | ---------: |
| Baseline ViT   |        83.80% |     325.86 min |      3.19M |
| Hybrid CNN–ViT |    **90.22%** | **142.99 min** |      3.45M |
| Improvement    |  **+6.42 pp** |     **−56.1%** |     +0.26M |

The hybrid model improved test accuracy by **6.42 percentage points** over the baseline.

The recorded training time was reduced by **56.1%**, despite the modified model having slightly more parameters. Training time is hardware-dependent and should not be interpreted as a general speed benchmark.

## Project Overview

Vision Transformers have become an important approach in computer vision, but their performance can depend on the amount and quality of training data and the training strategy used.

This project investigates whether combining convolutional feature extraction with a Vision Transformer can improve Fashion-MNIST classification performance.

The project uses two main models:

* **Baseline ViT** - a Vision Transformer trained directly on Fashion-MNIST images.
* **Hybrid CNN–ViT** - a modified architecture that first extracts spatial features using convolutional layers before passing the resulting feature maps to the Vision Transformer.

The modified model also uses several training improvements:

* Data augmentation
* MixUp
* Label smoothing
* Cosine learning-rate scheduling
* Three-epoch learning-rate warmup

Because these changes were evaluated together rather than through separate ablation experiments, the project measures their **combined effect** rather than attributing the improvement to any single modification.

## Dataset

The project uses the **Fashion-MNIST** dataset.

Fashion-MNIST contains:

* 60,000 training images
* 10,000 test images
* 10 clothing categories
* 28×28 grayscale images

The images are resized to **32×32** before being passed to the models.

The ten classes are:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset is downloaded automatically through `torchvision` and is not included in this repository.

## Baseline Model

The baseline is a Vision Transformer implemented using the Hugging Face Transformers library.

### Configuration

* Image size: 32×32
* Patch size: 4×4
* Input channels: 1
* Hidden size: 256
* Transformer layers: 6
* Attention heads: 8
* Intermediate size: 512
* Number of classes: 10
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 128
* Epochs: 10

The baseline model contains approximately **3.19 million parameters**.

The final test accuracy was **83.80%**.

## Hybrid CNN–ViT Model

The modified model adds a convolutional feature extractor before the Vision Transformer.

### CNN feature extractor

The CNN consists of:

```text
Input
  ↓
Conv2D: 1 → 32
  ↓
BatchNorm + ReLU
  ↓
Conv2D: 32 → 64
  ↓
BatchNorm + ReLU
  ↓
MaxPool
  ↓
64-channel feature map
```

The 32×32 input is reduced to a **16×16 spatial feature map** after pooling.

The resulting 64-channel feature map is passed to the Vision Transformer, which performs its patch embedding on these feature maps.

The hybrid model contains approximately **3.45 million parameters**.

## Training Improvements

### 1. Data Augmentation

The training pipeline includes:

* Random rotation up to 10 degrees
* Random translation up to 10%
* Random horizontal flipping
* Normalization

The test set uses resizing, tensor conversion, and normalization without random augmentation.

### 2. MixUp

MixUp is applied during training using a Beta distribution with:

```text
alpha = 0.2
```

Images and their corresponding labels are mixed using randomly sampled interpolation weights.

### 3. Label Smoothing

A label-smoothing value of **0.1** is used with a custom label-smoothing loss.

This prevents the training targets from being represented as completely one-hot distributions.

### 4. Learning-Rate Scheduling

The modified model uses a cosine learning-rate schedule with a **three-epoch warmup**.

The learning rate starts below the maximum value, reaches the configured maximum learning rate of **0.001**, and then gradually decreases during training.

## Final Performance

The hybrid model achieved the following results on the 10,000-image Fashion-MNIST test set:

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **90.22%** |
| Macro Precision |     90.18% |
| Macro Recall    |     90.22% |
| Macro F1        |     90.15% |
| Weighted F1     |     90.15% |

### Per-Class Performance

| Class       | Precision | Recall |     F1 |
| ----------- | --------: | -----: | -----: |
| T-shirt/top |    0.8486 | 0.8690 | 0.8587 |
| Trouser     |    0.9829 | 0.9790 | 0.9810 |
| Pullover    |    0.8542 | 0.8670 | 0.8605 |
| Dress       |    0.9014 | 0.8870 | 0.8942 |
| Coat        |    0.8171 | 0.8800 | 0.8474 |
| Sandal      |    0.9757 | 0.9650 | 0.9703 |
| Shirt       |    0.7691 | 0.6830 | 0.7235 |
| Sneaker     |    0.9212 | 0.9700 | 0.9450 |
| Bag         |    0.9761 | 0.9790 | 0.9775 |
| Ankle boot  |    0.9712 | 0.9430 | 0.9569 |

The strongest F1 scores were obtained for **Bag, Trouser, Sandal, Ankle boot, and Sneaker**.

**Shirt** was the most difficult class, with an F1 score of **0.7235**, while visually similar clothing categories such as T-shirt/top, Pullover, and Coat also produced lower scores than the footwear and bag categories.

## Visual Results

The repository includes training and evaluation visualisations:

* Baseline training curves
* Modified model training results
* Confusion matrix
* Per-class precision, recall, and F1 scores

These are available in the `results/` directory.

## Repository Structure

```text
fashion-mnist-cnn-vit/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── baseline_vit.ipynb
│   ├── modified_vit.ipynb
│   └── evaluation.ipynb
│
├── results/
│   ├── baseline_training_curves.png
│   ├── modified_model_results.png
│   ├── confusion_matrix_modified.png
│   └── per_class_metrics_modified.png
│
├── docs/
│   └── report.pdf
│
└── sources/
    └── sources.md
```

## Reproducibility

A random seed of **42** is used in the modified-model notebook for PyTorch, NumPy, and Python's random module to improve reproducibility.

To run the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/fashion-mnist-cnn-vit.git
cd fashion-mnist-cnn-vit
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks

Open the notebooks in Jupyter or another compatible environment and run them in the following order:

1. `dataset_exploration.ipynb`
2. `baseline_vit.ipynb`
3. `modified_vit.ipynb`
4. `evaluation.ipynb`

The Fashion-MNIST dataset will be downloaded automatically through `torchvision`.

## Hardware

The experiments were conducted on a **2017 13-inch MacBook Pro** with:

* Intel Core i5 2.3 GHz
* Intel Iris Plus Graphics 640
* 8 GB RAM

The reported training times are specific to this hardware and software environment.

## Limitations

This project has several limitations:

* The experiments were conducted on Fashion-MNIST, which is relatively simple compared with modern computer-vision benchmarks.
* The modified model has slightly more parameters than the baseline.
* No extensive hyperparameter search was performed.
* More advanced architectures and pretrained models were not investigated.
* Most importantly, the CNN, augmentation, MixUp, label smoothing, and learning-rate schedule were introduced together. Therefore, the individual contribution of each modification cannot be isolated from these experiments.

## Future Work

Potential extensions include:

* Performing ablation studies to evaluate each modification independently.
* Testing different CNN feature extractors.
* Exploring different ViT configurations.
* Performing more extensive hyperparameter optimisation.
* Comparing against additional CNN and Transformer architectures.
* Evaluating the approach on more challenging image-classification datasets.
* Investigating whether pretrained Vision Transformers provide further improvements.

## Project Report

The full academic report is available in:

```text
docs/report.pdf
```

## References

Key sources used in the project are documented in:

```text
sources/sources.md
```

## Project Context

This project was developed as an academic machine-learning project to investigate the practical effect of combining convolutional feature extraction with a Vision Transformer for image classification.

The main objective was not simply to build the largest model possible, but to explore whether targeted architectural and training modifications could improve classification performance while remaining practical to train on limited hardware.
