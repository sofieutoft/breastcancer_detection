import os
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
import warnings
from sklearn.metrics import jaccard_score

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')


def load_images_and_masks(dataset_path, img_size=(256, 256)):
    images, masks = [], []
    categories = ['benign', 'malignant', 'normal']

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        files = sorted(os.listdir(category_path))

        image_files = [f for f in files if 'mask' not in f.lower()]
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_name = base_name + '_mask.png'

            img_path = os.path.join(category_path, img_file)
            mask_path = os.path.join(category_path, mask_name)

            if os.path.exists(img_path) and os.path.exists(mask_path):
                image = cv.imread(img_path)
                mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

                if image is not None and mask is not None:
                    resized_img = cv.resize(image, img_size) / 255.0
                    resized_mask = cv.resize(mask, img_size, interpolation=cv.INTER_NEAREST)
                    binary_mask = (resized_mask > 127).astype(np.uint8)

                    images.append(resized_img)
                    masks.append(binary_mask)

    return np.array(images), np.array(masks)


def get_best_cluster_label(mask, cluster_labels, num_clusters=3):
    """
    Returns the cluster label that best matches the polyp region in the mask using IoU.
    """
    best_iou = 0
    best_label = 0
    for label in range(num_clusters):
        binary_cluster = (cluster_labels == label).astype(np.uint8)
        iou = jaccard_score(mask.flatten(), binary_cluster.flatten())
        if iou > best_iou:
            best_iou = iou
            best_label = label
    return best_label, best_iou

import matplotlib.pyplot as plt

def plot_cluster_segmentation(images, true_masks, cluster_labels_list, best_labels, num_images=5):
    """
    Visualize KMeans cluster segmentations alongside ground truth masks.

    Args:
        images (array): List/array of input images (H, W, 3).
        true_masks (array): List/array of ground truth masks (H, W).
        cluster_labels_list (array): List/array of predicted cluster labels (H, W).
        best_labels (list): List of best cluster label per image.
        num_images (int): Number of images to display.
    """
    plt.figure(figsize=(15, 5 * num_images))

    for i in range(num_images):
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.title(f"Image {i + 1}")
        plt.axis('off')

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(true_masks[i], cmap='gray')
        plt.title(f"True Mask {i + 1}")
        plt.axis('off')

        plt.subplot(num_images, 3, i * 3 + 3)
        pred_mask = (cluster_labels_list[i] == best_labels[i]).astype(np.uint8)
        plt.imshow(pred_mask, cmap='gray')
        plt.title(f"KMeans Prediction {i + 1}")
        plt.axis('off')
    plt.savefig(f"kmeans_segmentation.png")

def cross_validation(images, masks, num_folds=5, visualize=False):
    fold_size = len(images) // num_folds
    ious = []

    all_val_images = []
    all_val_true_masks = []
    all_val_cluster_labels = []
    all_val_best_labels = []

    for fold in range(num_folds):
        print(f"Processing Fold {fold + 1}/{num_folds}...")

        start = fold * fold_size
        end = (fold + 1) * fold_size if fold != num_folds - 1 else len(images)

        val_images = images[start:end]
        val_masks = masks[start:end]
        train_images = np.concatenate((images[:start], images[end:]), axis=0)
        train_masks = np.concatenate((masks[:start], masks[end:]), axis=0)

        # Only train KMeans on foreground pixels (mask == 1)
        mask_pixels = (train_masks > 0).reshape(-1)
        train_pixels = train_images.reshape(-1, 3)
        train_pixels = train_pixels[mask_pixels == 1]

        # Fit KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(train_pixels)

        for i, (image, mask) in enumerate(zip(val_images, val_masks)):
            h, w, _ = image.shape
            pixels = image.reshape(-1, 3)
            cluster_labels = kmeans.predict(pixels).reshape(h, w)

            best_label, best_iou = get_best_cluster_label(mask, cluster_labels, num_clusters=2)
            predicted_mask = (cluster_labels == best_label).astype(np.uint8)

            iou = jaccard_score(mask.flatten(), predicted_mask.flatten())
            ious.append(iou)

            if visualize:
                all_val_images.append(image)
                all_val_true_masks.append(mask)
                all_val_cluster_labels.append(cluster_labels)
                all_val_best_labels.append(best_label)

        print(f"Fold {fold + 1} Mean IoU: {np.mean(ious[-(end-start):]):.4f}")

    print(f"\nOverall Mean IoU across all folds: {np.mean(ious):.4f}")

    if visualize:
        plot_cluster_segmentation(
            all_val_images,
            all_val_true_masks,
            all_val_cluster_labels,
            all_val_best_labels,
            num_images=3
        )


def main():
    dataset_path = os.path.expanduser(
        "/Users/sofiautoft/.cache/kagglehub/datasets/aryashah2k/"
        "breast-ultrasound-images-dataset/versions/1/Dataset_BUSI_with_GT"
    )

    print("Loading dataset...")
    images, masks = load_images_and_masks(dataset_path)
    print(f"Loaded {len(images)} images.")
    cross_validation(images, masks, visualize=True)


if __name__ == "__main__":
    main()