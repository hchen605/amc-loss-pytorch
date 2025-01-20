import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(128, 256)  # Input size matches pooled feature size
        self.fc2 = nn.Linear(128, 10)   # Final output layer for logits

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
    def __init__(self, num_classes=10):
        super(PaperModel, self).__init__()
        
        self.gaussian_noise = 0.15

        # First block
        self.conv1_1 = nn.Conv2d(3, 128, kernel_size=3, padding='same')
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


