import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNetfeats(nn.Module):
    def __init__(self, device):
        super(ResNetfeats, self).__init__()
        self.device = device
        # Using a pre-trained ResNet as the base model for feature extraction
        self.base_model = models.resnet50(pretrained=False).to(self.device)

        # Replace the final fully connected layer to match our output requirements
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity().to(self.device)  # Remove the final FC layer

        # Defining additional layers for processing extracted features
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU()
        ).to(self.device)

        # Separate heads for content authenticity and quality assessment
        self.content_auth_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.quality_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

    def forward(self, x):
        # Feature extraction
        features = self.base_model(x)

        # Feature processing
        processed_features = self.feature_processor(features)

        # Content authenticity and quality assessment
        content_authenticity = self.content_auth_head(processed_features)
        quality = self.quality_head(processed_features)

        # Combine both heads' outputs by averaging them
        combined_output = (content_authenticity + quality) / 2

        return combined_output
