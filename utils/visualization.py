import matplotlib
matplotlib.use('Agg') # Set the backend to Agg for non-interactive plotting
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_knn_neighbors(X_train, y_train, X_test_sample, k, save_path=None):
    """
    可视化 K-NN 算法中一个测试样本的 K 个最近邻。
    显示测试样本及其 K 个最近邻图像，并标注它们的实际标签。

    Args:
        X_train (np.array): 训练数据。
        y_train (np.array): 训练标签。
        X_test_sample (np.array): 单个测试样本 (1, N) 格式。
        k (int): K-近邻的数量。
        save_path (str, optional): 保存图像的路径。如果为 None，则显示图像。
    """
    # 假设 X_train 和 X_test_sample 已经是展平的图像数据
    # 计算测试样本与所有训练样本的欧氏距离
    distances = np.linalg.norm(X_train - X_test_sample, axis=1)

    # 获取距离最小的 K 个训练样本的索引
    k_nearest_indices = np.argsort(distances)[:k]

    plt.figure(figsize=(10, 2))
    plt.suptitle(f"Test Sample and its {k} Nearest Neighbors", fontsize=16)

    # 显示测试样本
    ax = plt.subplot(1, k + 1, 1)
    ax.imshow(X_test_sample.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title("Test Sample")
    ax.axis('off')

    # 显示 K 个最近邻
    for i, idx in enumerate(k_nearest_indices):
        ax = plt.subplot(1, k + 1, i + 2)
        ax.imshow(X_train[idx].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f"Neighbor {i+1}\nLabel: {y_train[idx]}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(importances, feature_names, title, save_path=None):
    """
    可视化随机森林或其他基于树模型特征重要性。

    Args:
        importances (np.array): 特征重要性数组。
        feature_names (list): 特征名称列表 (例如，像素索引)。
        title (str): 图表的标题。
        save_path (str, optional): 保存图像的路径。如果为 None，则显示图像。
    """
    # 将重要性数据重塑为 28x28 图像形式
    importance_image = importances.reshape(28, 28)

    plt.figure(figsize=(6, 6))
    plt.imshow(importance_image, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Feature Importance')
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_dimensionality_reduction(X, y, method='pca', n_components=2, title="", save_path=None, sample_size=5000):
    """
    使用 PCA 或 t-SNE 可视化降维后的数据。

    Args:
        X (np.array): 输入数据 (例如，图像像素)。
        y (np.array): 标签。
        method (str): 降维方法 ('pca' 或 'tsne')。
        n_components (int): 降维后的维度数量 (2 或 3)。
        title (str): 图表标题。
        save_path (str, optional): 保存图像的路径。如果为 None，则显示图像。
        sample_size (int, optional): 用于降维可视化的样本数量，因为 t-SNE 计算成本较高。
    """
    if sample_size and len(X) > sample_size:
        # Randomly sample data for visualization to speed up computation
        np.random.seed(42)
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sampled = X[indices]
        y_sampled = y[indices]
    else:
        X_sampled = X
        y_sampled = y

    X_reduced = None
    if method == 'pca':
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X_sampled)
        title = title or f'PCA Projection ({n_components} Components)'
    elif method == 'tsne':
        if n_components == 3:
            # t-SNE to 3D can be very slow and may require more computational resources.
            # For most visual purposes, 2D is preferred.
            print("Warning: t-SNE to 3 components can be very slow. Consider n_components=2.")
        tsne = TSNE(n_components=n_components, random_state=42, init='pca', learning_rate='auto', perplexity=30, max_iter=1000)
        X_reduced = tsne.fit_transform(X_sampled)
        title = title or f't-SNE Projection ({n_components} Components)'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")

    plt.figure(figsize=(10, 8))
    if n_components == 2:
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_sampled, cmap='tab10', s=10, alpha=0.7)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D # Import 3D plotting toolkit
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_sampled, cmap='tab10', s=10, alpha=0.7)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

    plt.title(title)
    plt.colorbar(scatter, ticks=range(10), label='Digit Class')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 