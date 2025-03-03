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

def visualize_grad_cam(model, dataloader, device, class_names, index, dataset, use_amc_loss=False):

    infidelity_score = compute_infidelity_sample(model, dataloader, device, index, noise_std=0.003, perturbations=50) 

    model.eval()
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    inputs = inputs.to(device)

    print("Input image shape:", inputs.shape)
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
    #print("Input image shape:", input_sample.shape)
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
    if dataset == 'mnist':  # Single channel but has an explicit channel dimension
        input_sample = input_sample.repeat(1, 3, 1, 1)
    input_image = input_sample[0].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    input_image = (input_image * 0.5 + 0.5) * 255  # Denormalize and scale to 0-255
    input_image = input_image.astype(np.uint8)

    # Overlay heatmap on the original image
    overlay = cv2.addWeighted(input_image, 0.4, heatmap, 0.6, 0)

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

    # Add a title for the entire figure
    plt.suptitle(f"{predicted_class} / {true_class}, Infidelity: {infidelity_score:.6f}", fontsize=16)
    plt.tight_layout()
    #plt.show()
    path = './plot/cifar10_re/' + dataset + '_' + true_class + '_' + str(index) + '_' + str(use_amc_loss) + '_cam_re.png'
    plt.savefig(path)

def visualize_grad_cam_sc(model, batch, device, class_names, index, dataset, use_amc_loss=False):
    model.eval()
    #data_iter = iter(dataloader)
    inputs, labels = batch
    inputs = inputs.to(device)

    print("Input image shape:", inputs.shape)
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
    #print("Input image shape:", input_sample.shape)
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
    cam_resized = cv2.resize(cam, (inputs.shape[3], inputs.shape[2]))  # (H, W)
    cam_resized = (cam_resized * 255).astype(np.uint8)  # Scale to 0-255

    # Apply colormap
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Get the original image
    mel_spec_db = 10 * torch.log10(input_sample[0] + 1e-10).cpu().permute(1, 2, 0).numpy()
    mel_spec_norm = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mel_spec_color = cv2.cvtColor(mel_spec_norm, cv2.COLOR_GRAY2BGR)
    # if dataset == 'mnist' or 'speechcommands':  # Single channel but has an explicit channel dimension
    #     input_sample = input_sample.repeat(1, 3, 1, 1)
    # input_image = input_sample[0].cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    # #input_image = (input_image * 0.5 + 0.5) * 255  # Denormalize and scale to 0-255
    # input_image = input_image.astype(np.uint8)

    # Overlay heatmap on the original image
    overlay = cv2.addWeighted(mel_spec_color, 0.3, heatmap, 0.7, 0)

    print(predicted_class, '/', true_class)
    # Plot original image and overlay side by side
    plt.figure(figsize=(4, 2))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(mel_spec_db, origin='lower', aspect='auto')
    plt.title("Original Image")
    plt.axis("off")

    # Grad-CAM Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(overlay, origin='lower', aspect='auto')
    plt.title("Grad-CAM Overlay")
    plt.axis("off")

    # Add a title for the entire figure
    plt.suptitle(f"{predicted_class}, '/', {true_class}", fontsize=16)
    plt.tight_layout()
    #plt.show()
    path = './plot/sc_re/' + dataset + '_' + true_class +  '_' + str(index) + '_' + str(use_amc_loss) + '_cam.png'
    plt.savefig(path)

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
    print(f"Total data evaluated in test set: {total}")
    return accuracy

# Visualize latent space clustering
def visualize_latent_space(model, dataloader, device, class_names, dataset, use_amc_loss=False):
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
    #kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
    #predicted_labels = kmeans.fit_predict(reduced_features)
    predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()

    # Calculate Homogeneity and Completeness
    homogeneity = homogeneity_score(labels, predicted_labels)
    completeness = completeness_score(labels, predicted_labels)

    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Completeness: {completeness:.4f}")
    #return homogeneity, completeness

    # plt.figure(figsize=(8, 8))
    # scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10', alpha=0.7)

    # # Customize color bar with class names
    # cbar = plt.colorbar(scatter, ticks=range(len(class_names)))
    # cbar.ax.set_yticklabels(class_names)
    # cbar.set_label('Classes')
    
    # plt.title("Latent Space Clustering with t-SNE")
    # #plt.show()
    # path = './plot/' + dataset + '_' + str(use_amc_loss) + '_latent_ang_1_resent_mdl_adapt.png'
    # plt.savefig(path)

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

from captum.metrics import infidelity
from captum.attr import Saliency, FeatureAblation
from captum.robust import MinParamPerturbation
#from captum.robust.criteria import Misclassification

def compute_infidelity(model, dataloader, device, noise_std=0.01, perturbations=10):
    """
    Computes infidelity using Captum for a dataset.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader containing the dataset for evaluation.
        device: The device to run the model on.
        noise_std: Standard deviation of the noise to add.
        perturbations: Number of noise perturbations for each input.

    Returns:
        Average infidelity score over the dataset.
    """
    model.eval()
    # Define a wrapper to extract logits
    def forward_fn(inputs):
        """
        Forward function wrapper to return only logits for Captum.
        """
        _, logits = model(inputs)
        return logits

    saliency = Saliency(forward_fn)
    #saliency = Saliency(model)

    def perturb_func(inputs):
        """
        Perturbation function to add random noise to the inputs.
        """
        perturbations = torch.randn_like(inputs) * noise_std
        perturbed_inputs = inputs + perturbations
        return perturbations, perturbed_inputs

    infidelity_scores = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Compute saliency maps using Captum
        attributions = saliency.attribute(inputs, target=labels, abs=False)

        # Compute infidelity using Captum
        score = infidelity(
            forward_fn,
            perturb_func,
            inputs,
            attributions,
            target=labels,
            n_perturb_samples=perturbations,
        )
        #print(score)
        infidelity_scores.append(score.mean().item())
        #print(infidelity_scores)

    # Compute average infidelity across all batches
    average_infidelity = sum(infidelity_scores) / len(infidelity_scores)
    print(f"Average Infidelity: {average_infidelity:.6f}")
    return average_infidelity

def compute_infidelity_sample(model, dataloader, device, sample_index, noise_std=0.01, perturbations=10):
    """
    Computes infidelity using Captum for a single sample in the dataset.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader containing the dataset for evaluation.
        device: The device to run the model on.
        sample_index: Index of the sample in the dataset to evaluate.
        noise_std: Standard deviation of the noise to add.
        perturbations: Number of noise perturbations for each input.

    Returns:
        Infidelity score for the single sample.
    """
    model.eval()

    # Define a wrapper to extract logits
    def forward_fn(inputs):
        """
        Forward function wrapper to return only logits for Captum.
        """
        _, logits = model(inputs)
        return logits

    saliency = Saliency(forward_fn)

    def perturb_func(inputs):
        """
        Perturbation function to add random noise to the inputs.
        """
        perturbations = torch.randn_like(inputs) * noise_std
        perturbed_inputs = inputs + perturbations
        return perturbations, perturbed_inputs

    # Retrieve the specific sample based on the index
    dataset = dataloader.dataset
    inputs, labels = dataset[sample_index]
    inputs, labels = inputs.to(device).unsqueeze(0), torch.tensor(labels).to(device)

    # Compute saliency map using Captum
    attributions = saliency.attribute(inputs, target=labels, abs=False)

    # Compute infidelity for the single sample
    score = infidelity(
        forward_fn,
        perturb_func,
        inputs,
        attributions,
        target=labels,
        n_perturb_samples=perturbations,
    )
    infidelity_score = score.item()
    print(f"Infidelity Score for Sample {sample_index}: {infidelity_score:.6f}")
    return infidelity_score

def compute_infidelity_ecg(model, dataloader, device, noise_std=0.01, perturbations=10):
    """
    Computes the infidelity metric for an ECG dataset using Captum.
    
    Args:
        model: The model to evaluate. It should return (logits, embedding).
        dataloader: DataLoader containing the ECG dataset for evaluation.
        device: The device to run the model on.
        noise_std: Standard deviation of the noise to add.
        n_perturb_samples: Number of noise perturbations per input.
    
    Returns:
        Average infidelity score over the dataset.
    """
    model.eval()
    
    # Define a forward function that returns logits only
    def forward_fn(inputs):
        # Our model returns (logits, embedding)
        logits, _ = model(inputs)
        return logits

    # Initialize the Captum Saliency object with the forward function
    saliency = Saliency(forward_fn)

    # Define the perturbation function that adds random noise to inputs.
    # For time-series data, this simply adds noise with a given standard deviation.
    def perturb_func(inputs):
        noise = torch.randn_like(inputs) * noise_std
        perturbed_inputs = inputs + noise
        return noise, perturbed_inputs

    infidelity_scores = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Compute saliency maps using Captum for each input.
        attributions = saliency.attribute(inputs, target=labels, abs=False)

        # Compute the infidelity metric using Captum's infidelity function.
        score = infidelity(
            forward_fn,
            perturb_func,
            inputs,
            attributions,
            target=labels,
            n_perturb_samples=perturbations,
        )
        # The score is per input; take the mean across the batch.
        infidelity_scores.append(score.mean().item())

    average_infidelity = sum(infidelity_scores) / len(infidelity_scores)
    print(f"Average Infidelity: {average_infidelity:.6f}")
    return average_infidelity

import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_grad_cam_ecg(model, dataloader, device, class_names_ecg5000, index, use_amc_loss=False):
    model.eval()
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    inputs, labels = next(data_iter)
    inputs = inputs.to(device)
    
    print("Input ECG shape:", inputs.shape)  # Expected shape: (batch, channels, sequence_length)
    feature_maps = []
    gradients = []
    
    # Define hooks to capture features and gradients from a target convolutional layer.
    def forward_hook(module, input, output):
        feature_maps.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    # Assume you want to register hooks on the last convolutional layer of your feature extractor.
    # Change `target_layer` to the appropriate layer in your model.
    target_layer = model.feature_extractor[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    
    # Select a single sample from the batch.
    input_sample = inputs[index].unsqueeze(0)  # Shape: (1, channels, sequence_length)
    label_sample = labels[index]
    
    # Forward pass: get logits (and ignore embedding here).
    _, logits = model(input_sample)
    class_idx = logits[0].argmax().item()
    loss = logits[0, class_idx]
    loss.backward()
    
    # Retrieve feature map and gradients from the hooks.
    fmap = feature_maps[0][0].cpu().detach().numpy()  # Shape: (channels, L)
    grad = gradients[0][0].cpu().detach().numpy()       # Shape: (channels, L)
    
    # Compute channel-wise weights (global average pooling over the temporal dimension)
    weights = np.mean(grad, axis=1)  # Shape: (channels,)
    
    # Compute the 1D Grad-CAM: weighted combination of feature maps.
    cam = np.zeros(fmap.shape[1], dtype=np.float32)  # Initialize CAM with length L
    for i, w in enumerate(weights):
        cam += w * fmap[i, :]
    
    # Apply ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Avoid division by zero
    
    # Upsample CAM to match input length if necessary.
    input_length = input_sample.shape[-1]
    if cam.shape[0] != input_length:
        x_old = np.linspace(0, 1, cam.shape[0])
        x_new = np.linspace(0, 1, input_length)
        cam = np.interp(x_new, x_old, cam)
    
    # Get the original ECG signal (assuming single channel).
    ecg_signal = input_sample[0, 0].cpu().detach().numpy()
    
    # Create a visualization: plot the ECG signal and overlay the Grad-CAM.
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_signal, label="ECG Signal", color="black", linewidth=1.5)
    # Overlay CAM as a heatmap: use fill_between with a colormap.
    plt.fill_between(np.arange(input_length), ecg_signal, ecg_signal + cam * (np.max(ecg_signal)-np.min(ecg_signal)),
                     color='red', alpha=0.5, label="Grad-CAM")
    plt.title(f"Grad-CAM Visualization (Predicted: {class_idx}, True: {label_sample.item()})")
    plt.xlabel("Time Steps")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.tight_layout()
    save_path = f'./plot/ecg_{index}_{use_amc_loss}_cam.png'
    plt.savefig(save_path)
    plt.show()

# Example usage:
# visualize_grad_cam_ecg(model, ecg_dataloader, device, index=0, use_amc_loss=False)

def visualize_feature_ablation(model, dataloader, device, index, use_amc_loss=False):

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            # Assume model returns a tuple and logits is the first element.
            _, logits = self.model(x)
            return logits

    wrapped_model = ModelWrapper(model)
    wrapped_model.to(device)
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    inputs = inputs.to(device)
    input_tensor = inputs[index].unsqueeze(0)
    label_sample = labels[index]
    true_label_idx = label_sample.item()

    # Create a FeatureAblation object
    fa = FeatureAblation(wrapped_model)

    feature_mask = torch.arange(196).reshape(14,14).repeat_interleave(repeats=2, dim=1).repeat_interleave(repeats=2, dim=0).reshape(1,1,28,28).to(device)
    # For pixel-level ablation, use a sliding window shape of (1, 1)
    # This will ablate one pixel at a time across the entire image.
    #attributions = fa.attribute(input_tensor, target=true_label_idx, sliding_window_shapes=(1, 1, 1))
    attributions = fa.attribute(input_tensor, target=true_label_idx, feature_mask=feature_mask)

    #metric_value = torch.sum(torch.abs(attributions))
    # Compute absolute values of attributions
    abs_attr = torch.abs(attributions)

    # Create a mask for values greater than 2
    mask = abs_attr > 2

    # Sum only the values that pass the threshold
    metric_value = torch.sum(abs_attr[mask])
    print(f"Robustness: {metric_value.item():.2f}")
    # attributions is a tensor of the same shape as input_tensor, where each element
    # indicates the effect on the model's output when that corresponding pixel is ablated.
    # You can visualize the attribution map to see which pixels, when dropped out, cause
    # the most significant drop in prediction confidence.

    # For visualization (assuming a single-channel image)
    input_img = input_tensor.squeeze().cpu().detach().numpy()  # shape: (28, 28)
    attr_map = attributions.squeeze().cpu().detach().numpy()

    plt.figure(figsize=(5,5))
    # First, display the input image in grayscale.
    plt.imshow(input_img, cmap='gray')
    # Then, overlay the attribution map with a colormap and some transparency.
    plt.imshow(attr_map, cmap='jet', alpha=0.9)
    #plt.imshow(attr_map, cmap='hot')
    plt.title("Feature Ablation Attribution Map")
    plt.colorbar()
    save_path = f'./plot/mnist_re/fa_{index}_{use_amc_loss}.png'
    plt.savefig(save_path)
    #plt.show()

    # Compute pixel attributions using FeatureAblation.
    #ablator = FeatureAblation(net)
    #attr = ablator.attribute(normalize(image), target=label, feature_mask=feature_mask)
    # For a single-channel image (or if all channels share the same attributions), select one channel.
    pixel_attr = attributions[:, 0:1]

    # Define the pixel dropout attack function.
    def pixel_dropout(image, dropout_pixels):
        """
        Drops pixels based on the pixel attribution scores.
        
        Parameters:
        image: Input image tensor of shape [1, C, H, W]
        dropout_pixels: The number of pixels to drop.
        
        Returns:
        The ablated image, where pixels with attribution values below the kth threshold are set to 0.
        """
        # Total number of pixels in one channel.
        total_pixels = image[0][0].numel()
        # Determine the number of pixels to keep.
        keep_pixels = total_pixels - int(dropout_pixels)
        # Find the kth highest attribution value as threshold.
        kth_val, _ = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
        # Create a mask: pixels with attribution below the threshold are dropped.
        # (Assuming that higher attribution means more important, so we drop the lower ones.)
        mask = (pixel_attr < kth_val.item()).float()
        return mask * image

    # Set up MinParamPerturbation using the pixel_dropout attack.
    min_pert_attr = MinParamPerturbation(
        forward_func=wrapped_model,
        attack=pixel_dropout,
        arg_name="dropout_pixels",
        mode="linear",
        arg_min=0,
        arg_max=784,  # adjust according to total number of pixels (e.g., 28*28=784 for MNIST)
        arg_step=16,
        #preproc_fn=normalize,         # normalization function applied before model inference
        apply_before_preproc=True     # apply the perturbation before normalization
    )

    # Evaluate to find the minimum number of pixels that need to be dropped for misclassification.
    pixel_dropout_im, pixels_dropped = min_pert_attr.evaluate(input_tensor, target=true_label_idx, perturbations_per_eval=10)
    print("Minimum Pixels Dropped:", pixels_dropped/784)
    plt.figure(figsize=(5,5))
    # First, display the input image in grayscale.
    plt.imshow(input_img, cmap='gray')
    plt.imshow(pixel_dropout_im.squeeze().cpu().detach().numpy(), alpha=0.6)
    plt.title(f"Minimum Pixels Dropped: {(pixels_dropped/784):.4f}")
    save_path = f'./plot/mnist_re/pixel_drop_{index}_{use_amc_loss}.png'
    plt.savefig(save_path)


# def compute_feature_ablation(model, dataloader, device, use_amc_loss=False):

#     class ModelWrapper(torch.nn.Module):
#         def __init__(self, model):
#             super(ModelWrapper, self).__init__()
#             self.model = model

#         def forward(self, x):
#             # Assume model returns a tuple and logits is the first element.
#             _, logits = self.model(x)
#             return logits

#     wrapped_model = ModelWrapper(model)
#     wrapped_model.to(device)

#     # Create a FeatureAblation object
#     fa = FeatureAblation(wrapped_model)

#     # robust_scores = []
#     # for inputs, labels in dataloader:
#     #     inputs, labels = inputs.to(device), labels.to(device)
#     #     # For pixel-level ablation, use a sliding window shape of (1, 1)
#     #     # This will ablate one pixel at a time across the entire image.
#     #     attributions = fa.attribute(inputs, target=labels, sliding_window_shapes=(1, 1, 1))

#     #     #metric_value = torch.sum(torch.abs(attributions))
#     #     # Compute absolute values of attributions
#     #     abs_attr = torch.abs(attributions)

#     #     # Create a mask for values greater than 2
#     #     mask = abs_attr > 2

#     #     # Sum only the values that pass the threshold
#     #     metric_value = torch.sum(abs_attr[mask])
#     #     #print(f"Robustness: {metric_value.item():.2f}")
#     #     robust_scores.append(metric_value.mean().item())
        

#     # # Compute average infidelity across all batches
#     # average_robust = sum(robust_scores) / len(robust_scores)
#     # print(f"Average Robustness: {average_robust}")

#     pixel_attr = attributions[:, 0:1]

#     # Define the pixel dropout attack function.
#     def pixel_dropout(image, dropout_pixels):
#         """
#         Drops pixels based on the pixel attribution scores.
        
#         Parameters:
#         image: Input image tensor of shape [1, C, H, W]
#         dropout_pixels: The number of pixels to drop.
        
#         Returns:
#         The ablated image, where pixels with attribution values below the kth threshold are set to 0.
#         """
#         # Total number of pixels in one channel.
#         total_pixels = image[0][0].numel()
#         # Determine the number of pixels to keep.
#         keep_pixels = total_pixels - int(dropout_pixels)
#         # Find the kth highest attribution value as threshold.
#         kth_val, _ = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
#         # Create a mask: pixels with attribution below the threshold are dropped.
#         # (Assuming that higher attribution means more important, so we drop the lower ones.)
#         mask = (pixel_attr < kth_val.item()).float()
#         return mask * image

#     # Set up MinParamPerturbation using the pixel_dropout attack.
#     min_pert_attr = MinParamPerturbation(
#         forward_func=wrapped_model,
#         attack=pixel_dropout,
#         arg_name="dropout_pixels",
#         mode="linear",
#         arg_min=0,
#         arg_max=784,  # adjust according to total number of pixels (e.g., 28*28=784 for MNIST)
#         arg_step=16,
#         #preproc_fn=normalize,         # normalization function applied before model inference
#         apply_before_preproc=True     # apply the perturbation before normalization
#     )

#     pixels_dropped_all = []
#     for inputs, labels in dataloader:
#         attributions = fa.attribute(inputs, target=labels, sliding_window_shapes=(1, 1, 1))
#         # Evaluate to find the minimum number of pixels that need to be dropped for misclassification.
#         pixel_dropout_im, pixels_dropped = min_pert_attr.evaluate(inputs, target=labels, perturbations_per_eval=10)
#         pixels_dropped_all.append(pixels_dropped.mean()/784)

#     average_drop = sum(pixels_dropped_all) / len(pixels_dropped_all)
#     print("Average Minimum Pixels Dropped:", pixels_dropped/784)



def compute_feature_ablation(model, dataloader, device, use_amc_loss=False):
    # Wrap the model so that forward returns logits.
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            # Assume model returns a tuple and logits is the second element.
            # Adjust the indexing if needed.
            _, logits = self.model(x)
            return logits

    wrapped_model = ModelWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # Create a FeatureAblation object using the wrapped model.
    fa = FeatureAblation(wrapped_model)

    # Define a helper function that creates a pixel dropout function
    # using the per-sample pixel attributions.
    def make_pixel_dropout(pixel_attr):
        def pixel_dropout(image, dropout_pixels):
            """
            Drops pixels based on the pixel attribution scores.
            Parameters:
              image: Input image tensor of shape [1, C, H, W]
              dropout_pixels: The number of pixels to drop.
            Returns:
              The ablated image, where pixels with attribution values below the kth threshold are set to 0.
            """
            # total_pixels for one channel (assume single-channel input)
            total_pixels = image[0, 0].numel()
            keep_pixels = total_pixels - int(dropout_pixels)
            # kthvalue returns the kth smallest value in the flattened tensor
            kth_val, _ = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
            # Create mask: drop pixels with attribution below kth_val
            mask = (pixel_attr < kth_val.item()).float()
            return mask * image
        return pixel_dropout

    pixels_dropped_all = []

    # Process each batch from the dataloader
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Compute attributions for this batch.
        # Expecting inputs shape: [B, C, H, W]. Using sliding window (1,1,1)
        attributions = fa.attribute(inputs, target=labels, sliding_window_shapes=(1, 1, 1))
        # For a single-channel image, use one channel's attribution.
        pixel_attr_batch = attributions[:, 0:1]  # shape: [B, 1, H, W]

        # If batch size > 1, process each sample individually.
        if inputs.shape[0] > 1:
            for i in range(inputs.shape[0]):
                img = inputs[i:i+1]      # shape: [1, C, H, W]
                lab = labels[i:i+1].item()
                pixel_attr = pixel_attr_batch[i:i+1]  # shape: [1, 1, H, W]

                # Build a pixel dropout function for this sample.
                pixel_dropout_fn = make_pixel_dropout(pixel_attr)
                # Create a new MinParamPerturbation instance for this sample.
                min_pert = MinParamPerturbation(
                    forward_func=wrapped_model,
                    attack=pixel_dropout_fn,
                    arg_name="dropout_pixels",
                    mode="linear",
                    arg_min=0,
                    arg_max=img.shape[-2] * img.shape[-1],  # e.g., 784 for 28x28
                    arg_step=16,
                    apply_before_preproc=True
                )
                # Evaluate the minimal dropout needed for misclassification.
                # Here we use the Misclassification criterion by default.
                _, pixels_dropped = min_pert.evaluate(img, target=lab, perturbations_per_eval=10)
                if pixels_dropped is None:
                    # If no perturbation found that causes misclassification, assume maximum dropout (i.e., no misclassification was achieved).
                    normalized_pixels = 0.0  # or (arg_max/total_pixels), where arg_max = img.shape[-2] * img.shape[-1]
                else:
                    normalized_pixels = pixels_dropped / (img.shape[-2] * img.shape[-1])
                # Normalize by total pixels.
                pixels_dropped_all.append(normalized_pixels)
        else:
            # Batch size == 1, process directly.
            pixel_dropout_fn = make_pixel_dropout(pixel_attr_batch)
            min_pert = MinParamPerturbation(
                forward_func=wrapped_model,
                attack=pixel_dropout_fn,
                arg_name="dropout_pixels",
                mode="linear",
                arg_min=0,
                arg_max=inputs.shape[-2] * inputs.shape[-1],
                arg_step=16,
                apply_before_preproc=True
            )
            _, pixels_dropped = min_pert.evaluate(inputs, target=labels, perturbations_per_eval=10)
            pixels_dropped_all.append(pixels_dropped / (inputs.shape[-2] * inputs.shape[-1]))

    average_drop = sum(pixels_dropped_all) / len(pixels_dropped_all)
    print("Average Minimum Pixels Dropped (fraction of total pixels):", average_drop)

# Example usage:
# compute_feature_ablation(model, test_loader, device, use_amc_loss=False)
