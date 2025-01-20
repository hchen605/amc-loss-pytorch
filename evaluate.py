import cv2  # For resizing heatmap
from PIL import Image  # For blending images
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import homogeneity_score, completeness_score
import numpy as np
import torch

# def visualize_grad_cam(model, dataloader, device):
#     model.eval()
#     data_iter = iter(dataloader)
#     inputs, _ = next(data_iter)
#     inputs = inputs.to(device)

#     feature_maps = []
#     gradients = []

#     def forward_hook(module, input, output):
#         feature_maps.append(output)

#     def backward_hook(module, grad_in, grad_out):
#         gradients.append(grad_out[0])

#     # Register hooks on the convolutional layer of interest
#     model.conv3.register_forward_hook(forward_hook)
#     model.conv3.register_backward_hook(backward_hook)

#     # Get embeddings and logits
#     embeddings, logits = model(inputs)

#     # Process only the first sample in the batch
#     class_idx = logits[0].argmax().item()  # Select class index for the first sample
#     loss = logits[0, class_idx]
#     loss.backward()

#     fmap = feature_maps[0][0].cpu().detach().numpy()  # Feature map for the first sample
#     grad = gradients[0][0].cpu().detach().numpy()  # Gradient for the first sample

#     weights = np.mean(grad, axis=(1, 2))
#     cam = np.sum(weights[:, None, None] * fmap, axis=0)
#     cam = np.maximum(cam, 0)
#     cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize

#     plt.imshow(cam, cmap='jet')
#     plt.title("Grad-CAM Heatmap")
#     plt.colorbar()
#     plt.show()

class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

def visualize_grad_cam(model, dataloader, device, class_names, index):
    model.eval()
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    inputs = inputs.to(device)

    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks on the convolutional layer of interest
    model.conv3_3.register_forward_hook(forward_hook)
    model.conv3_3.register_backward_hook(backward_hook)

    # Select the specific sample
    input_sample = inputs[index].unsqueeze(0)  # Add batch dimension
    label_sample = labels[index]

    # Get embeddings and logits
    embeddings, logits = model(input_sample)

    # Process only the first sample in the batch
    class_idx = logits[0].argmax().item()  # Select class index for the first sample
    true_label_idx = label_sample.item()
    loss = logits[0, class_idx]
    loss.backward()

    # Decode class names
    predicted_class = class_names[class_idx]
    true_class = class_names[true_label_idx]

    fmap = feature_maps[0][0].cpu().detach().numpy()  # Feature map for the first sample
    grad = gradients[0][0].cpu().detach().numpy()  # Gradient for the first sample

    # Compute weights and Grad-CAM
    weights = np.mean(grad, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * fmap, axis=0)
    cam = np.maximum(cam, 0)  # ReLU
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize

    # Resize the heatmap to match input size
    cam_resized = cv2.resize(cam, (inputs.shape[2], inputs.shape[3]))  # (H, W)
    cam_resized = (cam_resized * 255).astype(np.uint8)  # Scale to 0-255

    # Apply colormap
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Get the original image
    input_image = input_sample[0].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    input_image = (input_image * 0.5 + 0.5) * 255  # Denormalize and scale to 0-255
    input_image = input_image.astype(np.uint8)

    # Overlay heatmap on the original image
    overlay = cv2.addWeighted(input_image, 0.6, heatmap, 0.4, 0)

    print(predicted_class, '/', true_class)
    # Plot original image and overlay side by side
    plt.figure(figsize=(4, 2))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title("Original Image")
    plt.axis("off")

    # Grad-CAM Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, logits = model(inputs)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Visualize latent space clustering
def visualize_latent_space(model, dataloader, device, class_names):
    model.eval()
    features = []
    labels = []
    logits = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            embeddings, logit = model(inputs)
            features.append(embeddings.cpu().numpy())
            labels.append(targets.numpy())
            logits.append(logit) #

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    logits = torch.cat(logits, dim=0)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # For simplicity, use KMeans to assign cluster labels
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
    #predicted_labels = kmeans.fit_predict(reduced_features)
    predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()

    # Calculate Homogeneity and Completeness
    homogeneity = homogeneity_score(labels, predicted_labels)
    completeness = completeness_score(labels, predicted_labels)

    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Completeness: {completeness:.4f}")
    #return homogeneity, completeness

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10', alpha=0.7)

    # Customize color bar with class names
    cbar = plt.colorbar(scatter, ticks=range(len(class_names)))
    cbar.ax.set_yticklabels(class_names)
    cbar.set_label('Classes')
    
    plt.title("Latent Space Clustering with t-SNE")
    plt.show()

def measure_clustering_performance(model, dataloader, device):
    model.eval()
    features = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            embeddings, _ = model(inputs)
            features.append(embeddings.cpu().numpy())
            true_labels.append(labels.numpy())

    # Combine all features and labels
    features = np.concatenate(features, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Perform dimensionality reduction (optional, e.g., using t-SNE)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # For simplicity, use KMeans to assign cluster labels
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=len(np.unique(true_labels)), random_state=42)
    predicted_labels = kmeans.fit_predict(reduced_features)

    # Calculate Homogeneity and Completeness
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)

    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Completeness: {completeness:.4f}")
    return homogeneity, completeness