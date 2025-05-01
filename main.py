import numpy as np
import cv2
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100, tolerance=1e-4):
    # Randomly initialize k centroids
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iters):
        # Assign each point to the closest centroid
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Recompute centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids

    return labels, centroids

def segment_image_with_kmeans(image_path, k):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape image to 2D array of pixels
    pixel_values = image.reshape((-1, 3)).astype(np.float32)

    # Run K-means
    labels, centroids = kmeans(pixel_values, k)

    # Replace each pixel with its centroid color
    segmented_image = centroids[labels].reshape(image.shape).astype(np.uint8)

    return image, segmented_image, labels.reshape(image.shape[:2])

def extract_number_mask(labels, target_cluster):
    # Create a binary mask for the chosen cluster
    mask = (labels == target_cluster).astype(np.uint8) * 255
    return mask

if __name__ == "__main__":
    image_path = "Input/6.jpg" 
    k = 3

    original, segmented, label_map = segment_image_with_kmeans(image_path, k)

    # Choose the target cluster index by observing which one contains the number
    number_mask = extract_number_mask(label_map, target_cluster=1)

    # Display results
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
