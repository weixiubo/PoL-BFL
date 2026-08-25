"""
Model Definitions for Experiments

Provides various neural network models for FL experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST/FashionMNIST
    
    Architecture:
    - Conv1: 1x28x28 -> 32x24x24
    - Conv2: 32x24x24 -> 64x8x8
    - FC1: 64*8*8 -> 128
    - FC2: 128 -> 10
    """
    
    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # CIFAR10 (32x32) → 5x5 after two 5x5 conv (no padding) + two 2x2 pools
        # MNIST (28x28)  → 4x4 after same ops
        self.flatten_dim = 64 * (5 * 5 if input_channels == 3 else 4 * 4)
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FEMNISTTwoLayerCNN(nn.Module):
    """Two-convolution FEMNIST network with the paper's ~0.4M parameters."""

    def __init__(self, num_classes=62):
        super().__init__()
        if num_classes != 62:
            raise ValueError("the FEMNIST reference model requires 62 classes")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 320)
        self.fc2 = nn.Linear(320, 62)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ResNet18(nn.Module):
    """
    ResNet-18 for CIFAR-10/CIFAR-100

    Adapted from torchvision.models.resnet18 for 32x32 images.
    Supports variable number of classes via num_classes parameter.
    """
    
    def __init__(self, num_classes=10, layers=(2, 2, 2, 2)):
        super(ResNet18, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        
        # First block (may downsample)
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet34(ResNet18):
    """ResNet-34 for CIFAR-10/CIFAR-100."""

    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes, layers=(3, 4, 6, 3))


class VGG11(nn.Module):
    """
    VGG-11 for CIFAR-10
    
    Simplified VGG architecture for 32x32 images.
    """
    
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(model_name: str, num_classes: int = 10, input_channels: int = 3):
    """
    Factory function to create model instances
    
    Args:
        model_name: Name of model ('SimpleCNN', 'ResNet18', 'ResNet34', 'VGG11')
        num_classes: Number of output classes
        input_channels: Number of input channels
    
    Returns:
        model: PyTorch model
    """
    key = str(model_name).lower()
    if key == 'simplecnn':
        model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    elif key in {'twolayercnn', 'femnisttwolayercnn', 'femnistcnn'}:
        model = FEMNISTTwoLayerCNN(num_classes=num_classes)
    elif key == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif key == 'resnet34':
        model = ResNet34(num_classes=num_classes)
    elif key == 'vgg11':
        model = VGG11(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if str(os.getenv('POL_SUPPRESS_MODEL_INFO', '0')).strip().lower() not in ('1', 'true', 'yes', 'on'):
        logger.info(f"Created {model_name} model")
    return model


def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_info(model, model_name: str):
    """Print model information"""
    num_params = count_parameters(model)
    size_mb = get_model_size(model)
    
    print("\n" + "="*50)
    print(f"Model: {model_name}")
    print("="*50)
    print(f"Number of parameters: {num_params:,}")
    print(f"Model size: {size_mb:.2f} MB")
    print("="*50 + "\n")
