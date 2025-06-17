import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging

def plot_confusion_matrices(y_true_train, y_pred_train, y_true_test, y_pred_test, title_prefix="", save_path=None):
    """
    绘制训练集和测试集的混淆矩阵在同一张图中。
    """
    conf_mat_train = confusion_matrix(y_true_train, y_pred_train)
    conf_mat_test = confusion_matrix(y_true_test, y_pred_test)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 绘制训练集混淆矩阵
    sns.heatmap(conf_mat_train, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_title(f'{title_prefix}Confusion Matrix for Validation Data')
    axes[0].set_ylabel('True label')
    axes[0].set_xlabel('Predicted label')

    # 绘制测试集混淆矩阵
    sns.heatmap(conf_mat_test, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1])
    axes[1].set_title(f'{title_prefix}Confusion Matrix for Test Data')
    axes[1].set_ylabel('True label')
    axes[1].set_xlabel('Predicted label')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()
    plt.clf()
    plt.close(fig)

def display_prediction_examples(test_img, test_labels, test_predictions, img_shape=(28, 28), title_prefix="", save_path=None, individual_predictions_dict=None):
    """
    展示正确和错误预测的示例图片。
    """
    # 确保 test_labels 和 test_predictions 是一维数组，以便进行比较
    if test_labels.ndim > 1:
        test_labels_flat = np.argmax(test_labels, axis=1)
    else:
        test_labels_flat = test_labels

    if test_predictions.ndim > 1:
        test_predictions_flat = np.argmax(test_predictions, axis=1)
    else:
        test_predictions_flat = test_predictions

    correct_indices = np.where(test_labels_flat == test_predictions_flat)[0]
    incorrect_indices = np.where(test_labels_flat != test_predictions_flat)[0]

    num_display = 5
    selected_correct_indices = np.random.choice(correct_indices, size=min(num_display, len(correct_indices)), replace=False)
    selected_incorrect_indices = np.random.choice(incorrect_indices, size=min(num_display, len(incorrect_indices)), replace=False)

    fig_samples, axes_samples = plt.subplots(2, num_display, figsize=(15, 6))

    # 绘制正确预测的图片
    for idx, num in enumerate(selected_correct_indices):
        img_data = test_img[num]
        if img_data.ndim == 4: # For CNN data with channel dimension
            image_display = (img_data[:, :, 0] * 255).astype(np.uint8)
        else: # For other models (flattened 1D)
            image_display = (np.reshape(img_data, img_shape) * 255).astype(np.uint8)

        axes_samples[0, idx].imshow(image_display, interpolation='nearest', cmap='gray')
        axes_samples[0, idx].set_title('Original: {0}\nPredicted: {1}'.format(test_labels_flat[num], test_predictions_flat[num]))
        axes_samples[0, idx].axis('off')
    if len(selected_correct_indices) > 0:
        axes_samples[0, 0].text(-0.1, 0.5, "Correct Predictions", rotation=90, va='center', ha='right', transform=axes_samples[0, 0].transAxes, fontsize=12)

    # 绘制错误预测的图片
    for idx, num in enumerate(selected_incorrect_indices):
        img_data = test_img[num]
        if img_data.ndim == 4: # For CNN data with channel dimension
            image_display = (img_data[:, :, 0] * 255).astype(np.uint8)
        else: # For other models (flattened 1D)
            image_display = (np.reshape(img_data, img_shape) * 255).astype(np.uint8)

        title_text = f'Original: {test_labels_flat[num]}\nPredicted: {test_predictions_flat[num]}'
        if individual_predictions_dict:
            for model_name, preds in individual_predictions_dict.items():
                title_text += f'\n{model_name}: {preds[num]}'

        axes_samples[1, idx].imshow(image_display, interpolation='nearest', cmap='gray')
        axes_samples[1, idx].set_title(title_text)
        axes_samples[1, idx].axis('off')
    if len(selected_incorrect_indices) > 0:
        axes_samples[1, 0].text(-0.1, 0.5, "Incorrect Predictions", rotation=90, va='center', ha='right', transform=axes_samples[1, 0].transAxes, fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Prediction examples plot saved to {save_path}")
    else:
        plt.show()
    plt.clf()
    plt.close(fig_samples)

def print_classification_report(y_true, y_pred, title=""):
    """
    打印分类报告。
    """
    logging.info(f'\n{title}Classification Report:\n')
    logging.info(classification_report(y_true, y_pred))

def display_voting_correction_details(test_img, test_labels, ensemble_predictions, individual_predictions_dict, img_shape=(28, 28), save_path=None, img_filenames=None):
    """
    展示通过投票纠正的错误预测示例，并显示所有模型的预测。
    """
    corrected_by_voting_indices = []
    for i in range(len(test_labels)):
        # Check if ensemble prediction is correct
        if ensemble_predictions[i] == test_labels[i]:
            # Check if at least one individual model was incorrect for this sample
            individual_incorrect = False
            for model_name, preds in individual_predictions_dict.items():
                if preds[i] != test_labels[i]:
                    individual_incorrect = True
                    break
            if individual_incorrect:
                corrected_by_voting_indices.append(i)
    
    if not corrected_by_voting_indices:
        logging.info("No examples found where voting ensemble corrected individual model errors.")
        return

    num_display = min(5, len(corrected_by_voting_indices))
    selected_indices = np.random.choice(corrected_by_voting_indices, size=num_display, replace=False)

    fig, axes = plt.subplots(1, num_display, figsize=(3 * num_display, 7))
    if num_display == 1:
        axes = [axes]

    fig.suptitle('Examples Corrected by Voting Ensemble', fontsize=16, y=0.77)

    for idx, original_idx in enumerate(selected_indices):
        img_data = test_img[original_idx]
        if img_data.ndim == 4: # For CNN data with channel dimension
            image_display = (img_data[:, :, 0] * 255).astype(np.uint8)
        else: # For other models (flattened 1D)
            image_display = (np.reshape(img_data, img_shape) * 255).astype(np.uint8)

        ax = axes[idx]
        ax.imshow(image_display, interpolation='nearest', cmap='gray')
        ax.axis('off')

        true_label = test_labels[original_idx]
        ensemble_pred = ensemble_predictions[original_idx]

        pred_text = f'True: {true_label}\nEnsemble: {ensemble_pred} (Correct)'
        for model_name, preds in individual_predictions_dict.items():
            pred_text += f'\n{model_name}: {preds[original_idx]}'

        ax.set_title(f'Example {idx + 1}')
        if img_filenames and len(img_filenames) > original_idx:
            ax.text(0.5, -0.07, f'File: {img_filenames[original_idx]}.txt', transform=ax.transAxes, ha='center', va='bottom', fontsize=8, color='blue')

        ax.text(0.5, -0.1, pred_text, transform=ax.transAxes, ha='center', va='top', fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Voting corrected examples plot saved to {save_path}")
    else:
        plt.show()
    plt.clf()
    plt.close(fig) 