import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


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

        # Combined loss
        total_loss = cross_entropy_loss + self.lambda_ * amc_loss

        return total_loss

# Define training loop
def train_model(model, criterion, optimizer, dataloader, num_epochs=10, use_amc_loss=False):
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

        if epoch % 50 == 0:  
            if use_amc_loss:
                path = './weights/' +  'train/' + 'paper_' + 'amc-loss' + '_ep_' + str(epoch) + '.pth'
            else:
                path = './weights/' +  'train/' + 'paper_' + '_ep_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), path)
            print("Model weights saved at epoch ", epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}", flush=True)

