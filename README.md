# Handwritten Digit Recognition using Machine Learning and Deep Learning

## Project Overview

本项目实现了基于机器学习和深度学习手写数字识别。它包含了使用K-近邻 (KNN)、支持向量机 (SVM)、随机森林分类器 (RFC) 等传统机器学习算法以及卷积神经网络 (CNN) 进行数字识别的代码。项目已更新，支持本地数据集，并优化了环境配置和依赖管理。

## 环境要求

本项目推荐使用 [Anaconda](https://www.anaconda.com/products/individual) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 来管理Python环境和依赖。

### 推荐的 Conda 环境设置

为了避免依赖冲突，建议为本项目创建一个专用的 Conda 环境：

1.  **创建 Conda 环境**:
    ```bash
    conda create -n TF python=3.9
    ```

2.  **激活环境**:
    ```bash
    conda activate TF
    ```

3.  **安装依赖**:
    项目的所有依赖都列在 `requirements.txt` 文件中。请确保在激活 `TF` 环境后运行以下命令安装：
    ```bash
    pip install -r requirements.txt
    ```
    当前 `requirements.txt` 中的关键依赖包括：
    *   `tensorflow==2.13.0`
    *   `keras==2.13.1`
    *   `scikit-learn==1.5.0`
    *   `numpy`
    *   `matplotlib`
    *   `opencv-python`
    *   `pandas`
    *   `seaborn`

    **注意**: 在环境配置过程中，我们遇到了 `tensorflow` 与 `torch` 在 `typing-extensions` 上的版本冲突。当前的 `requirements.txt` 已调整以兼容 `tensorflow==2.13.0` 和 `keras==2.13.1`，并已成功解决所有依赖冲突。如果您遇到新的冲突，可能需要根据具体错误信息调整 `requirements.txt` 中的版本。

## 数据集

本项目不再依赖于在线下载 MNIST 数据集或其 `ubyte` 格式文件。现在，所有算法都使用本地 `Dataset/` 文件夹中的数据。

### 数据集结构

请确保您的项目根目录下存在一个名为 `Dataset/` 的文件夹，其结构如下：

```
Dataset/
├── train.csv         # 训练数据标签，CSV格式
├── test.csv          # 测试数据标签，CSV格式
├── train/            # 训练图像，每个图像为一个TXT文件
│   ├── 0.txt
│   ├── 1.txt
│   └── ...
└── test_no_label/    # 测试图像，每个图像为一个TXT文件 (无标签)
    ├── 0.txt
    ├── 1.txt
    └── ...
```

*   `train.csv` 和 `test.csv` 包含图像对应的标签。
*   `train/` 和 `test_no_label/` 文件夹包含图像数据，每个图像是一个 `32x32` 的二进制文本文件。

### 数据加载

所有机器学习和深度学习脚本现在都通过项目根目录下的 `MNIST_Dataset_Loader/mnist_loader.py` 模块加载数据。该模块负责从上述本地文件路径读取数据，并进行必要的预处理（例如，将 `32x32` 图像缩放为 `28x28`，并展平数据以适应某些模型的要求）。

## 使用说明

请确保您已按照"环境要求"部分设置好 Conda 环境并安装了所有依赖，并且"数据集"部分的数据集已正确放置在项目根目录下的 `Dataset/` 文件夹中。

1.  **激活 Conda 环境**:
    在运行任何脚本之前，请激活您的 Conda 环境：
    ```bash
    conda activate TF
    ```

2.  **运行机器学习算法 (KNN, SVM, Random Forest)**:
    这些算法的代码位于各自的文件夹中 (`1.KNN/`、`2.SVM/`、`3.RFC/`)。
    *   **K-近邻 (KNN)**:
        ```bash
        python .\main.py --model KNN
        ```
    *   **支持向量机 (SVM)**:
        ```bash
        python .\main.py --model SVM
        ```
    *   **随机森林分类器 (RFC)**:
        ```bash
        python .\main.py --model RFC
        ```
    运行这些脚本时，图像数据会从 `Dataset/` 文件夹加载，并自动进行展平处理以适应模型的输入要求。

3.  **运行卷积神经网络 (CNN)**:
    CNN 代码位于 `4. CNN_Keras/CNN_MNIST.py`。此脚本也会从本地 `Dataset/` 文件夹加载数据。
    ```bash
    cd "4. CNN_Keras/"
    python CNN_MNIST.py
    ```

    *   **保存 CNN 模型权重**:
        ```bash
        python .\main.py --model CNN --save_model 1 --save_weights cnn_weights.hdf5
        ```
        权重文件将保存到 `Results/CNN/` 目录下。
    *   **加载预训练 CNN 模型权重**:
        ```bash
        python .\main.py --model CNN --load_model 1 --save_weights cnn_weights.hdf5
        ```
        权重文件将从 `Results/CNN/` 目录下加载。

4.  **运行投票集成 (Voting Ensemble)**:
    集成模型代码位于 `voting_ensemble.py`。
    ```bash
    python .\voting_ensemble.py
    ```
    此脚本结合了不同模型的预测结果。在我们的调整中，`cnn_boost_factor` 已被设置为 `2.0`，以进一步强调 CNN 模型在集成中的影响力。

5.  **绘制数据降维可视化结果**:
    您可以通过 `main.py` 的 `--plot_dim_reduction` 参数来生成训练数据的 PCA 和 t-SNE 可视化图。这些图将保存在 `Results/Dimensionality_Reduction/` 目录下。
    ```bash
    python .\main.py --model KNN --plot_dim_reduction
    ```
    **注意**: `--model` 参数是 `main.py` 的必需参数，即使您只打算生成降维图，也需要指定一个模型（例如 `KNN`）。

6.  **日志输出**:
    所有脚本的输出都将记录到 `summary.log` 文件中。如果您希望在命令行中直接看到输出，请根据脚本内部的注释修改相关行（通常是注释掉将输出重定向到日志文件的部分）。

## 准确率

请注意，以下准确率是基于模型的**测试集**性能。

### 机器学习算法准确率:

*   **K-近邻**: 97.46%
*   **支持向量机**: 97.78%
*   **随机森林分类器**: 98.31%

### 深度神经网络准确率:

*   **三层卷积神经网络 (TensorFlow/Keras)**: 99.89%

## 输出图像解释

项目在运行过程中会生成多种可视化图像，以帮助理解模型的性能和内部机制。所有输出文件（包括日志、模型权重和图表）都统一存放在项目根目录下的 `Results/` 目录中。`Results/` 目录的结构如下：

```
Results/
├── CNN/                # 卷积神经网络 (CNN) 相关输出
│   ├── summary.log
│   ├── cnn_weights.hdf5
│   ├── CNN_Confusion_Matrix.png
│   ├── CNN_Loss_Accuracy_Curves.png
│   └── CNN_Prediction_Examples.png
├── KNN/                # K-近邻 (KNN) 相关输出
│   ├── summary.log
│   ├── MNIST_KNN.pickle
│   ├── KNN_Confusion_Matrices.png
│   ├── KNN_Prediction_Examples.png
│   └── KNN_Nearest_Neighbors.png
├── RFC/                # 随机森林分类器 (RFC) 相关输出
│   ├── summary.log
│   ├── MNIST_RFC.pickle
│   ├── RFC_Confusion_Matrices.png
│   ├── RFC_Prediction_Examples.png
│   └── RFC_Feature_Importance.png
├── SVM/                # 支持向量机 (SVM) 相关输出
│   ├── summary.log
│   ├── MNIST_SVM.pickle
│   ├── SVM_Confusion_Matrices.png
│   └── SVM_Prediction_Examples.png
├── Ensemble/           # 投票集成 (Voting Ensemble) 相关输出
│   ├── summary.log
│   ├── Ensemble_Confusion_Matrix.png
│   └── Ensemble_Prediction_Examples.png
└── Dimensionality_Reduction/ # 降维可视化结果
    ├── PCA_2D.png
    └── tSNE_2D.png
```

以下是不同类型输出图像的详细解释：

1.  **`PCA_2D.png` 和 `tSNE_2D.png` (降维可视化)**
    *   **含义**: 这些图像展示了训练数据在经过主成分分析 (PCA) 和 t-SNE 降维到二维空间后的分布情况。PCA 是一种线性降维技术，旨在找到数据中方差最大的方向；t-SNE 是一种非线性降维技术，旨在保留高维数据中相似点的局部结构。这些图可以帮助我们直观地理解数据的内在结构，查看不同数字类别在低维空间中的聚类和可分离性。它们保存到 `Results/Dimensionality_Reduction/` 目录下，与具体分类器无关。

2.  **`*_Confusion_Matrices.png` (例如 `KNN_Confusion_Matrices.png`, `SVM_Confusion_Matrices.png`, `RFC_Confusion_Matrices.png`)**
    *   **含义**: 混淆矩阵是一种特殊的表格布局，用于可视化监督学习算法的性能，特别是分类器的性能。矩阵的每一行代表真实标签的实例，而每一列代表预测标签的实例。对角线上的值表示正确分类的数量，非对角线上的值表示错误分类的数量。这些图像展示了模型在验证集和测试集上对每个数字（0-9）的分类表现，可以清晰地看出哪些数字容易被误分类。

3.  **`*_Prediction_Examples.png` (例如 `KNN_Prediction_Examples.png`, `SVM_Prediction_Examples.png`, `RFC_Prediction_Examples.png`)**
    *   **含义**: 这些图像展示了模型对测试集中随机选择的图像的预测结果。每张小图会显示原始图像，并标注其真实标签和模型的预测标签。这提供了一种直观的方式来检查模型的预测质量，特别是对于那些模型可能分类错误的图像。

4.  **`KNN_Nearest_Neighbors.png` (K-近邻特有)**
    *   **含义**: 这是K-近邻算法特有的可视化。它会选择一个随机的测试样本，并显示该测试样本以及训练集中与其最相似的 K 个（通常为 5 个）最近邻图像。每张邻居图像下方会标注其真实标签。此图旨在直观地展示 K-近邻算法是如何根据最近邻的标签来进行分类决策的。

5.  **`RFC_Feature_Importance.png` (随机森林分类器特有)**
    *   **含义**: 这是随机森林分类器特有的可视化。它将模型的特征重要性（在此项目中是像素的重要性）以热力图的形式展示出来。图像中颜色越亮（通常是暖色调），表示对应的像素点在模型进行数字识别时越重要。这有助于理解模型在识别手写数字时主要关注图像的哪些区域，例如数字的笔画边缘或核心部分。
