import numpy as np
import cv2
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100, tolerance=1e-4):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids

    return labels, centroids

def segment_image_with_kmeans(image_path, k):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    labels, centroids = kmeans(pixel_values, k)
    segmented_image = centroids[labels].reshape(image.shape).astype(np.uint8)
    return image, segmented_image, labels.reshape(image.shape[:2])

def extract_number_mask(labels, target_cluster):
    return (labels == target_cluster).astype(np.uint8) * 255

def refine_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed

if __name__ == "__main__":
    image_path = "Input/74.jpg"  # Change this to your input image
    k = 7

    original, segmented, label_map = segment_image_with_kmeans(image_path, k)

    # Plot all clusters to visually inspect
    print("Showing individual clusters to help pick the one with the number:")
    cluster_masks = []
    for i in range(k):
        raw_mask = extract_number_mask(label_map, target_cluster=i)
        refined_mask = refine_mask(raw_mask)
        cluster_masks.append(refined_mask)

        plt.imshow(refined_mask, cmap='gray')
        plt.title(f"Cluster {i}")
        plt.axis('off')
        plt.show()

    # Pick the best cluster by manually checking above
    TARGET_CLUSTER_INDEX = int(input("Enter the cluster index that contains the number: "))
    number_mask = cluster_masks[TARGET_CLUSTER_INDEX]

    # Show final result
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Extracted Number Mask")
    plt.imshow(number_mask, cmap='gray')
    plt.axis('off')

    plt.show()
