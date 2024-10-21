import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # Compute attention scores
        scores = self.v(torch.tanh(self.W(x)))
        weights = torch.softmax(scores, dim=1)
        # Compute weighted sum
        context = torch.sum(weights * x, dim=1)
        return context


class EfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes, attention_dim=1280):
        super(EfficientNetWithAttention, self).__init__()
        # Load the pre-trained EfficientNet-B0 model
        self.efficientnet = models.efficientnet_b7(pretrained=True)
        
        # Define the attention mechanism
        self.attention = Attention(input_dim=attention_dim)
        
        # Modify the classifier
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, attention_dim)
        
        # New classifier
        self.fc = nn.Linear(attention_dim, num_classes)

    def forward(self, x):
        # Extract features from EfficientNet
        features = self.efficientnet(x)
        
        # Apply attention
        attention_output = self.attention(features.unsqueeze(1))  # Add batch dimension
        
        # Classify using the new classifier
        out = self.fc(attention_output)
        return out
        
#model = EfficientNetWithAttention(num_classes=NUM_CLASSES)



class EfficientNetWithAttention_fusion(nn.Module):
    def __init__(self, num_classes, attention_dim=1280,attention_dim_1=1792):
        super(EfficientNetWithAttention_fusion, self).__init__()
        # Load the pre-trained EfficientNet-B0 model
        self.efficientnet = models.efficientnet_b7(pretrained=True)
        
        # Define the attention mechanism
        self.attention = Attention(input_dim=attention_dim_1)
        
        # Modify the classifier
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, attention_dim)
        
        # New classifier
        self.fc = nn.Linear(attention_dim_1, num_classes)

    def forward(self, x, x2):
        # Extract features from EfficientNet
        features = self.efficientnet(x)
        # Apply global average pooling to x2
        x2_pooled = F.adaptive_avg_pool2d(x2, (1, 1))  # Shape: (batch_size, 512, 1, 1)
        # Use repeat instead of expand to match the spatial dimensions
        encoder_output_reshaped = x2_pooled.squeeze(-1).squeeze(-1)

        # Concatenate encoder output with features
        combined_features = torch.cat((encoder_output_reshaped, features), dim=1)
        # Apply self-attention to the combined features
        attention_output = self.attention(combined_features.unsqueeze(1))  # Add batch dimension if needed for attention
        
        # Classify using the new classifier
        out = self.fc(attention_output)
        return out
        

class EfficientNet_NoAttention_fusion(nn.Module):
    def __init__(self, num_classes, attention_dim=1280,dim=1792):
        super(EfficientNet_NoAttention_fusion, self).__init__()
        # Load the pre-trained EfficientNet-B0 model
        self.efficientnet = models.efficientnet_b7(pretrained=True)
                
        # Modify the classifier
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, attention_dim)
        
        # New classifier
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x2):
        # Extract features from EfficientNet
        features = self.efficientnet(x)
        # Apply global average pooling to x2
        x2_pooled = F.adaptive_avg_pool2d(x2, (1, 1))  # Shape: (batch_size, 512, 1, 1)
        # Use repeat instead of expand to match the spatial dimensions
        encoder_output_reshaped = x2_pooled.squeeze(-1).squeeze(-1)

        # Concatenate encoder output with features
        combined_features = torch.cat((encoder_output_reshaped, features), dim=1)
        # Classify using the new classifier
        out = self.relu(self.fc1(combined_features))
        out = self.relu(self.fc2(out))     
        out = self.fc3(out)           
        return out



class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        # Contracting path (Encoder)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Expansive path (Decoder)
        self.upconv4 = self.upconv(512, 256)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.conv_block(64, 32)

        # Final layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        up4 = self.upconv4(bottleneck)
        dec4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        up3 = self.upconv3(dec4)
        dec3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        up2 = self.upconv2(dec3)
        dec2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        up1 = self.upconv1(dec2)
        dec1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.final_conv(dec1)
        return out,bottleneck



class SpatialAttention3D_CBAM(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention3D_CBAM, self).__init__()
        assert kernel_size in {3, 5, 7}, "kernel size must be 3, 5, or 7"
        padding = kernel_size // 2

        # Adjust channels for grayscale input
        in_channels = 1  # Assuming grayscale input

        self.conv1 = nn.Conv2d(in_channels * 2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute average and max pooling across feature channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat= torch.cat([avg_out, max_out], dim=1)

        # Apply 3D convolution for attention generation
        x_cat = self.conv1(x_cat)
        attention = self.sigmoid(x_cat)

        # Apply attention element-wise to input features
        return x * attention

class UNet_attention(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_attention, self).__init__()
        
        # Contracting path (Encoder)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Expansive path (Decoder)
        self.upconv4 = self.upconv(512, 256)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.conv_block(64, 32)

        # Final layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.spatial_attention_1 = SpatialAttention3D_CBAM()
        self.spatial_attention_2 = SpatialAttention3D_CBAM()
        self.spatial_attention_3 = SpatialAttention3D_CBAM()
        self.spatial_attention_4 = SpatialAttention3D_CBAM()
        
        self.spatial_attention_11 = SpatialAttention3D_CBAM()
        self.spatial_attention_22 = SpatialAttention3D_CBAM()
        self.spatial_attention_33 = SpatialAttention3D_CBAM()
        self.spatial_attention_44 = SpatialAttention3D_CBAM()

    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1 = self.spatial_attention_1(enc1)
        
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc2 = self.spatial_attention_1(enc2)
        
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc3 = self.spatial_attention_1(enc3)
        
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc4 = self.spatial_attention_1(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        up4 = self.upconv4(bottleneck)
        dec4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        dec4 = self.spatial_attention_1(dec4)
        
        up3 = self.upconv3(dec4)
        dec3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        dec3 = self.spatial_attention_1(dec3)

        up2 = self.upconv2(dec3)
        dec2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec2 = self.spatial_attention_1(dec2)
        
        up1 = self.upconv1(dec2)
        dec1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        dec1 = self.spatial_attention_1(dec1)
        
        out = self.final_conv(dec1)
        return out,bottleneck

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)  # Output: 256x14x14
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)  # Output: 128x14x14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size to 128x7x7

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 512)  # Input: 128x7x7, Output: 512
        self.fc2 = nn.Linear(512, 256)            # Input: 512, Output: 256
        self.fc3 = nn.Linear(256, num_classes)    # Input: 256, Output: num_classes
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        #x = self.pool(x)
        # Flatten the output for the dense layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x            