import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights
)
from resnets_cifar import ResNet18, ResNet50, ResNet10, ResNet8

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(128, 256)  # Input size matches pooled feature size
        self.fc2 = nn.Linear(128, num_classes)   # Final output layer for logits

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        
        # Global average pooling
        embeddings = self.global_pool(x).view(x.size(0), -1)  # Shape: (batch_size, 128)
        #embeddings = F.relu(self.fc1(embeddings))  # Optional, depending on the design
        
        # Final classification layer
        logits = self.fc2(embeddings)  # Shape: (batch_size, num_classes)
        
        return embeddings, logits

class PaperModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(PaperModel, self).__init__()
        
        self.gaussian_noise = 0.15

        # First block
        self.conv1_1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding='same')
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv1_3 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.5)

        # Second block
        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.conv2_3 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.5)

        # Third block
        self.conv3_1 = nn.Conv2d(256, 512, kernel_size=3, padding='valid')
        self.conv3_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv3_3 = nn.Conv2d(256, 128, kernel_size=1)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Add Gaussian noise
        if self.training:
            noise = torch.randn_like(x) * self.gaussian_noise
            x = x + noise

        # First block
        x = F.leaky_relu(self.conv1_1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv1_2(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv1_3(x), negative_slope=0.1)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = F.leaky_relu(self.conv2_1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2_2(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2_3(x), negative_slope=0.1)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third block
        x = F.leaky_relu(self.conv3_1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3_2(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3_3(x), negative_slope=0.1)

        # Global Average Pooling
        embeddings = self.global_avg_pool(x)
        embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten

        # Fully Connected Layer
        logits = self.fc(embeddings)

        return embeddings, logits

# Example usage
#model = PaperModel(num_classes=10)
#print(model)

class ResNetWithEmbedding(nn.Module):
    """
    Wraps a standard ResNet so that its forward pass returns (embeddings, logits)
    instead of just logits. The 'embeddings' come from the layer right before
    the final linear classifier.
    """
    def __init__(self, base_model: nn.Module, num_classes: int):
        super(ResNetWithEmbedding, self).__init__()
        
        # Extract all layers except the original 'fc'
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )
        self.avgpool = base_model.avgpool

        # The in_features of the final linear layer
        in_features = base_model.fc.in_features

        # We create our own fully-connected layer to control
        # the number of classes. The base_model.fc is unused here.
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Pass through all feature layers
        x = self.features(x)
        x = self.avgpool(x)              # (batch_size, channels, 1, 1)
        embeddings = torch.flatten(x, 1) # (batch_size, in_features)

        # Compute logits from embeddings
        logits = self.fc(embeddings)
        return embeddings, logits


# Function to get the chosen model
def get_model(model_type, input_channels, num_classes, pretrained=False):
    """
    Return the specified model, adjusted for the number of input channels and classes.
    """
    if model_type == 'paper_model':
        # Use the custom PaperModel
        model = PaperModel(input_channels=input_channels)
    else:
        if model_type == 'resnet18':
            if pretrained:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = resnet18(weights=None)

        elif model_type == 'resnet50':
            if pretrained:
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = resnet50(weights=None)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # If using MNIST (1 channel), adjust the first convolution layer
        if input_channels == 1:
            # Replace the first conv layer
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        model = ResNetWithEmbedding(model, num_classes=num_classes)

    return model

def define_tsnet(name, num_class, cuda=True):
	if name == 'resnet20':
		net = resnet20(num_class=num_class)
	elif name == 'resnet110':
		net = resnet110(num_class=num_class)
	elif name == 'resnet18':
		net = ResNet18(num_class=num_class)
	elif name == 'resnet10':
		net = ResNet10(num_class=num_class)
	elif name == 'resnet8':
		net = ResNet8(num_class=num_class)
	elif name == 'resnet50':
		net = ResNet50(num_class=num_class)
		# net = resnet50()        

	else:
		raise Exception('model name does not exist.')

	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	else:
		net = torch.nn.DataParallel(net)

	return net


class Conv1DModel(nn.Module):
    """
    A 1D CNN model for ECG time-series classification that extracts an embedding
    before the final classifier. The model consists of a feature extractor composed
    of several 1D convolutional layers followed by an adaptive global pooling layer.
    An embedding layer maps the pooled features to a lower-dimensional space, which
    is then used by the classifier.
    
    Args:
        input_channels (int): Number of input channels (e.g., 1 for univariate time-series).
        num_classes (int): Number of target classes.
        embedding_dim (int): Dimension of the embedding space (default: 128).
    """
    def __init__(self, input_channels, num_classes, embedding_dim=128):
        super(Conv1DModel, self).__init__()

        # Feature extractor: several 1D conv layers with BatchNorm and ReLU activations.
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Adaptive pooling to handle variable input lengths:
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Output shape: (batch_size, channels, 1)
        
        # Embedding layer to produce a fixed-length representation.
        self.embedding_layer = nn.Linear(64, embedding_dim)
        
        # Final classifier layer
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length)
        
        Returns:
            logits (torch.Tensor): Classification logits of shape (batch_size, num_classes)
            embedding (torch.Tensor): Embedding vector of shape (batch_size, embedding_dim)
        """
        # Extract features using convolutional layers.
        features = self.feature_extractor(x)
        # Global average pooling across the temporal dimension.
        pooled = self.global_pool(features).squeeze(-1)  # Shape: (batch_size, channels)
        # Project pooled features to an embedding.
        embedding = self.embedding_layer(pooled)
        embedding = F.relu(embedding)
        # Compute logits from the embedding.
        logits = self.classifier(embedding)
        
        return embedding, logits
