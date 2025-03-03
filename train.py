import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from evaluate import evaluate_model, compute_infidelity

class AMCLoss(nn.Module):
    def __init__(self, angular_margin=0.5, lambda_=0.1):
        """
        Args:
            angular_margin (float): Angular margin (mg) to separate non-neighbor points.
            lambda_ (float): Balancing parameter between cross-entropy and AMC loss.
        """
        super(AMCLoss, self).__init__()
        self.angular_margin = angular_margin
        self.lambda_ = lambda_

    def forward(self, embeddings, logits, labels):
        """
        Args:
            embeddings (torch.Tensor): Feature embeddings (N x D).
            labels (torch.Tensor): Ground truth labels (N).
        Returns:
            torch.Tensor: Combined loss (cross-entropy + AMC loss).
        """
        # Normalize embeddings to unit hypersphere
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise cosine similarities (dot product since normalized)
        cos_sim = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Compute pairwise labels
        labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # (N x N) pairwise label similarity

        # Compute geodesic distance (angular distance)
        geodesic_distances = torch.acos(torch.clamp(cos_sim, min=-0.99999, max=0.99999))
        #print(geodesic_distances)
        # Compute AMC-Loss
        positive_loss = (labels_eq * geodesic_distances**2).sum() / labels_eq.sum()
        #print(labels_eq.sum())
        #print('positive_loss ', positive_loss)
        negative_loss = (
            (~labels_eq) * torch.clamp(self.angular_margin - geodesic_distances, min=0)**2
        ).sum() / (~labels_eq).sum()
        #print((~labels_eq).sum())
        #print('negative_loss ', negative_loss)
        amc_loss = positive_loss + negative_loss

        # Cross-Entropy Loss
        #logits = torch.mm(embeddings, normalized_embeddings.t())
        cross_entropy_loss = F.cross_entropy(logits, labels)
        #print('cross_entropy_loss ', cross_entropy_loss)
        #print('amc_loss ', amc_loss)
        # Combined loss
        total_loss = cross_entropy_loss + self.lambda_ * amc_loss

        return total_loss

class AMCLossWithPairing(nn.Module):
    def __init__(self, angular_margin=0.5, lambda_=0.1):
        """
        Args:
            angular_margin (float): Angular margin (mg) to separate non-neighbor points.
            lambda_ (float): Balancing parameter between cross-entropy and AMC loss.
        """
        super(AMCLossWithPairing, self).__init__()
        self.angular_margin = angular_margin
        self.lambda_ = lambda_

    def forward(self, embeddings, logits, labels):
        """
        Args:
            embeddings (torch.Tensor): Feature embeddings (N x D).
            labels (torch.Tensor): Ground truth labels (N).
        Returns:
            torch.Tensor: Combined loss (cross-entropy + AMC loss).
        """
        # Ensure we have an even number of samples for pairing
        batch_size = embeddings.shape[0]
        #print('batch_size: ', batch_size)
        if batch_size % 2 == 1:
            # Drop the last sample if batch size is odd
            embeddings = embeddings[:-1]
            logits = logits[:-1]
            labels = labels[:-1]
            
        # Normalize embeddings to unit hypersphere
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        # Split embeddings into two halves for pairwise comparison
        half = embeddings.size(0) // 2
        emb1, emb2 = normalized_embeddings[:half], normalized_embeddings[half:]
        labels1, labels2 = labels[:half], labels[half:]

        # Compute pairwise cosine similarities (dot product since normalized)
        inner_product = (emb1 * emb2).sum(dim=1)

        # Compute geodesic distance (angular distance)
        geodesic_distances = torch.acos(torch.clamp(inner_product, -1.0 + 1e-5, 1.0 - 1e-5))

        # Define positive (same class) and negative (different class) masks
        positive_mask = (labels1 == labels2)
        negative_mask = ~positive_mask

        # Compute AMC-Loss
        if positive_mask.sum() == 0:
            positive_loss = torch.tensor(0.0, device=embeddings.device)
        else:
            positive_loss = (positive_mask * geodesic_distances**2).sum() / positive_mask.sum()

        if negative_mask.sum() == 0:
            negative_loss = torch.tensor(0.0, device=embeddings.device)
        else:
            negative_loss = (
                negative_mask * torch.clamp(self.angular_margin - geodesic_distances, min=0)**2
            ).sum() / negative_mask.sum()

        amc_loss = positive_loss + negative_loss

        # Cross-Entropy Loss
        #logits = torch.mm(embeddings, normalized_embeddings.t())
        cross_entropy_loss = F.cross_entropy(logits, labels)
        #print('cross_entropy_loss ', cross_entropy_loss)
        #print('amc_loss ', amc_loss)
        # Combined loss
        total_loss = cross_entropy_loss + self.lambda_ * amc_loss

        return total_loss

# Define training loop
def train_model(model, path, criterion, optimizer, dataloader, dataloader_val, num_epochs=10, use_amc_loss=False):
    # Check device compatibility (including Apple Silicon MPS support)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    #criterion = criterion.to(device)
    print('device: ', device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            embeddings, logits = model(inputs)
            if use_amc_loss:
                loss = criterion(embeddings, logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)

            loss.backward()
            # Clip gradients to a maximum norm (e.g., 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}", flush=True)

        if (epoch + 1) % 50 == 0 and epoch > 100:  
            path_ = path + '_ep_' + str(epoch+1) + '.pth'
            #print('------')
            #evaluate_model(model, dataloader_val, device)
            #torch.save(model.state_dict(), path_)
            #print("Model weights saved at epoch ", epoch + 1)
        elif (epoch + 1) % 5 == 0:
            #print('------')
            evaluate_model(model, dataloader_val, device)
            infidelity_score = compute_infidelity(model, dataloader_val, device, noise_std=0.003, perturbations=50)
            print('\n')



def train_model_resnet(model, path, criterion, optimizer, dataloader, dataloader_val, num_epochs=10, use_amc_loss=False, scheduler=None):
    # Check device compatibility (including Apple Silicon MPS support)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    #criterion = criterion.to(device)
    print('device: ', device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            embeddings, logits = model(inputs)
            if use_amc_loss:
                loss = criterion(embeddings, logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 150 == 0 and epoch > 100:  
            path_ = path + '_ep_' + str(epoch+1) + '.pth'
            print('------')
            evaluate_model(model, dataloader_val, device)
            torch.save(model.state_dict(), path_)
            print("Model weights saved at epoch ", epoch + 1)
        elif (epoch + 1) % 10 == 0:
            print('------')
            evaluate_model(model, dataloader_val, device)

        # Step the scheduler after each epoch if provided
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}", flush=True)


def collate_fn(batch):
    """
    Custom collate function to pad spectrograms in a batch.

    Args:
        batch: List of tuples (spectrogram, label). Each spectrogram is a tensor 
               of shape [1, n_mels, time], where 'time' may vary across samples.

    Returns:
        padded_specs: Tensor of shape [batch_size, 1, n_mels, max_time] with zero padding.
        labels: Tensor of shape [batch_size].
    """
    # Local mapping dictionary to convert string labels to integers.
    label_list = [
        'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four',
        'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off',
        'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
        'tree', 'two', 'up', 'wow', 'yes', 'zero', 'backward', 'forward',
        'follow', 'learn', 'visual'
    ]
    label_dict = {lab: i for i, lab in enumerate(label_list)}

    spectrograms = []
    labels_list_out = []

    for spec, label in batch:
        # Ensure the spectrogram has an explicit channel dimension.
        # If the spectrogram is of shape [n_mels, time], unsqueeze to [1, n_mels, time].
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        spectrograms.append(spec)
        
        # If label is a string, convert it using label_dict.
        if isinstance(label, str):
            labels_list_out.append(label_dict.get(label, 0))
        else:
            labels_list_out.append(label)

    # Find the maximum time dimension among the spectrograms.
    max_time = max(spec.shape[-1] for spec in spectrograms)
    
    padded_specs = []
    for spec in spectrograms:
        pad_amount = max_time - spec.shape[-1]
        # Pad only the time dimension (last dimension)
        padded_spec = F.pad(spec, (0, pad_amount))
        padded_specs.append(padded_spec)
    
    # Stack into a batch tensor.
    padded_specs = torch.stack(padded_specs)
    labels_tensor = torch.tensor(labels_list_out, dtype=torch.long)
    
    return padded_specs, labels_tensor

def collate_fn_us8k(batch):
    """
    Custom collate function to pad spectrograms in a batch for UrbanSound8K.

    Args:
        batch: List of tuples (spectrogram, label). Each spectrogram is a tensor 
               of shape [1, n_mels, time] or [n_mels, time] (if channel dimension is missing).

    Returns:
        padded_specs: Tensor of shape [batch_size, 1, n_mels, max_time] with zero padding.
        labels: Tensor of shape [batch_size].
    """
    spectrograms = []
    labels_list = []

    for spec, label in batch:
        # Ensure the spectrogram has an explicit channel dimension.
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        spectrograms.append(spec)
        # For UrbanSound8K, the label is already an integer.
        labels_list.append(label)

    # Find the maximum time dimension among the spectrograms.
    max_time = max(spec.shape[-1] for spec in spectrograms)

    padded_specs = []
    for spec in spectrograms:
        pad_amount = max_time - spec.shape[-1]
        # Pad only the time dimension (last dimension)
        padded_spec = F.pad(spec, (0, pad_amount))
        padded_specs.append(padded_spec)

    padded_specs = torch.stack(padded_specs)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    return padded_specs, labels_tensor