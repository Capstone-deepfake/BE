import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.cm as cm
import pandas as pd
import random
from pathlib import Path
from PIL import Image
from .gradcam import apply_gradcam_to_image

# Configuration matching the trained model
TRAINED_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'frame_count': 32,  # Updated from 20 to 32 to match trained model
    'image_size': 128,
    'lstm_hidden_size': 256,  # Updated from 128 to 256
    'lstm_layers': 2,
    'dropout_rate': 0.3,  # Updated from 0.5 to 0.3
    'weight_decay': 0.0001,
    'gradient_clip': 1.0,
    'name': 'base_config'
}

# Use trained config values
FRAME_COUNT = TRAINED_CONFIG['frame_count']
IMAGE_SIZE = TRAINED_CONFIG['image_size']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------
# Enhanced Model Architecture (from second code)
# ------------------------------
class EnhancedMesoNet(nn.Module):
    """Enhanced MesoNet with improved architecture"""
    def __init__(self, image_size=128):
        super(EnhancedMesoNet, self).__init__()
        self.image_size = image_size

        # First conv block
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second conv block
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third conv block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth conv block
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fifth conv block (additional for better feature extraction)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)

        # Calculate feature dimension
        feature_dim = 256 * (image_size // 32) * (image_size // 32)
        self.feature_size = feature_dim

    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Fourth block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        # Fifth block
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        return x


class EnhancedMesoNetLSTM(nn.Module):
    """Enhanced MesoNet + LSTM matching the trained architecture"""
    def __init__(self, config=None):
        super(EnhancedMesoNetLSTM, self).__init__()

        if config is None:
            config = TRAINED_CONFIG

        self.config = config

        # Enhanced MesoNet base model
        self.mesonet = EnhancedMesoNet(config['image_size'])
        feature_dim = self.mesonet.feature_size

        # Flatten features
        self.flatten = nn.Flatten()

        # Feature reduction layer
        self.feature_reducer = nn.Sequential(
            nn.Linear(feature_dim, config['lstm_hidden_size']),
            nn.ReLU(),
            nn.BatchNorm1d(config['lstm_hidden_size']),
            nn.Dropout(config['dropout_rate'] * 0.5)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config['lstm_hidden_size'],
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=config['dropout_rate'] if config['lstm_layers'] > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config['lstm_hidden_size'] * 2,
            num_heads=8,
            dropout=config['dropout_rate'],
            batch_first=True
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(config['dropout_rate']),
            nn.Linear(config['lstm_hidden_size'] * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(config['dropout_rate'] * 0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate'] * 0.3),
            nn.Linear(64, 1)
        )

        # Store frame features for analysis
        self.frame_activations = []

    def forward(self, x, save_activations=False):
        # Handle single frame input
        if len(x.shape) == 4:  # [batch, channels, height, width]
            features = self.mesonet(x)
            features = self.flatten(features)
            features = self.feature_reducer(features)
            features = features.unsqueeze(1)  # Add sequence dimension
            lstm_out, _ = self.lstm(features)
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            pooled_out = torch.mean(attended_out, dim=1)
            output = self.classifier(pooled_out)
            return torch.sigmoid(output)  # Return probability for single frame

        # Handle sequence input [batch, frames, channels, height, width]
        batch_size, seq_len = x.size(0), x.size(1)
        frame_features = []

        # Process each frame through MesoNet
        for t in range(seq_len):
            frame = x[:, t]  # No need for extra dimensions
            features = self.mesonet(frame)
            features = self.flatten(features)
            features = self.feature_reducer(features)
            frame_features.append(features)

        # Stack features for LSTM
        lstm_input = torch.stack(frame_features, dim=1)

        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)

        # Apply attention mechanism
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling over sequence dimension
        pooled_out = torch.mean(attended_out, dim=1)

        # Final classification
        output = self.classifier(pooled_out)
        return torch.sigmoid(output)  # Return probability

# ------------------------------
# Grad-CAM Implementation (updated for enhanced architecture)
# ------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register hooks on the target layer
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor, class_idx=None):
        # Set model to training mode for backward pass
        self.model.train()

        # Enable gradients
        input_tensor.requires_grad_(True)

        # Forward pass
        model_output = self.model(input_tensor)

        if class_idx is None:
            class_idx = model_output.argmax(dim=1)

        # Clear gradients
        self.model.zero_grad()

        # Backward pass - use the raw output for binary classification
        model_output[0, 0].backward(retain_graph=True)

        # Generate CAM
        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            print("Warning: No gradients or activations captured")
            return np.zeros((4, 4))  # Return empty heatmap for 32x reduction

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        # Set model back to eval mode
        self.model.eval()

        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# ------------------------------
# Utility Functions
# ------------------------------
def extract_frames(video_path, num_frames=FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frames.append(frame)

    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.array(frames)

def denormalize_frame(frame_tensor):
    """Denormalize frame for visualization"""
    device = frame_tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    frame = frame_tensor * std + mean
    frame = torch.clamp(frame, 0, 1)
    return frame.permute(1, 2, 0).cpu().numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]

    # Overlay
    overlayed = alpha * heatmap_colored + (1 - alpha) * image
    return overlayed

def load_trained_model(model_path, config=None):
    """Load the trained model with proper configuration"""
    if config is None:
        config = TRAINED_CONFIG

    # Initialize model with the same architecture
    model = EnhancedMesoNetLSTM(config).to(DEVICE)

    # Load the saved state
    try:
        # Try loading as checkpoint format first
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # It's a checkpoint with metadata
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint. Accuracy: {checkpoint.get('accuracy', 'N/A')}")
            print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        else:
            # It's just the state dict
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the model path and format")
        return None

    model.eval()
    return model

def visualize_enhanced_gradcam(video_path, model_path, num_frames_to_show=10, save_path=None):
    """Enhanced Grad-CAM visualization with the trained model"""
    print(f"Loading enhanced model from: {model_path}")

    # Load the trained model
    model = load_trained_model(model_path, TRAINED_CONFIG)
    if model is None:
        return

    # Transform matching training setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Extract frames
    print(f"Extracting {FRAME_COUNT} frames from video...")
    frames_np = extract_frames(video_path)
    frames_tensor = torch.stack([transform(frame) for frame in frames_np])
    frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE)

    # Get prediction from full model
    print("Getting model prediction...")
    with torch.no_grad():
        output = model(frames_tensor, save_activations=True)
        prob = torch.sigmoid(output).item()
        label = "FAKE" if prob > 0.5 else "REAL"

    print(f"Prediction: {label} (Confidence: {prob:.4f})")

    # Create a CNN-only model for Grad-CAM (using the enhanced architecture)
    class CNNOnlyEnhanced(nn.Module):
        """CNN-only version of the enhanced model for GradCAM visualization"""
        def __init__(self, base_model):
            super().__init__()
            self.mesonet = base_model.mesonet
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            features = self.mesonet(x)
            pooled = self.global_pool(features)
            pooled = pooled.view(pooled.size(0), -1)
            return self.classifier(pooled)

    # Create CNN model for Grad-CAM
    cnn_model = CNNOnlyEnhanced(model).to(DEVICE)
    cnn_model.eval()

    # Select frames to visualize
    frame_indices = np.linspace(0, FRAME_COUNT-1, num_frames_to_show, dtype=int)

    # Create subplots with better layout for frames
    fig, axes = plt.subplots(3, len(frame_indices), figsize=(25, 12))
    if len(frame_indices) == 1:
        axes = axes.reshape(3, 1)

    # List to store all heatmaps for aggregation
    all_heatmaps = []

    for i, frame_idx in enumerate(frame_indices):
        # Get single frame
        single_frame = frames_tensor[:, frame_idx, :, :, :]

        # Generate Grad-CAM using the enhanced architecture
        gradcam = GradCAM(cnn_model, cnn_model.mesonet.conv5)

        try:
            heatmap = gradcam.generate_cam(single_frame)
            all_heatmaps.append(heatmap)  # Save for aggregation

            # Original frame for visualization
            original_frame = denormalize_frame(frames_tensor[0, frame_idx])

            # Plot original frame
            axes[0, i].imshow(original_frame)
            axes[0, i].set_title(f'Frame {frame_idx+1}')
            axes[0, i].axis('off')

            # Plot heatmap
            axes[1, i].imshow(heatmap, cmap='jet')
            axes[1, i].set_title(f'Attention Map')
            axes[1, i].axis('off')

            # Plot overlay
            overlayed = overlay_heatmap(original_frame, heatmap)
            axes[2, i].imshow(overlayed)
            axes[2, i].set_title(f'Overlay')
            axes[2, i].axis('off')

        except Exception as e:
            print(f"Warning: Could not generate Grad-CAM for frame {frame_idx}: {e}")
            original_frame = denormalize_frame(frames_tensor[0, frame_idx])
            all_heatmaps.append(np.zeros((heatmap.shape if 'heatmap' in locals() else (4,4))))  # fallback zero heatmap
            for row in range(3):
                axes[row, i].imshow(original_frame)
                axes[row, i].set_title(f'Frame {frame_idx+1}')
                axes[row, i].axis('off')

        gradcam.remove_hooks()

    # AGGREGATED HEATMAP OVER REPRESENTATIVE FRAME
    if len(all_heatmaps) > 0:
        # Sum all heatmaps
        aggregated_heatmap = np.sum(np.stack(all_heatmaps), axis=0)

        # Normalize aggregated heatmap to [0,1]
        aggregated_heatmap -= aggregated_heatmap.min()
        if aggregated_heatmap.max() > 0:
            aggregated_heatmap /= aggregated_heatmap.max()

        # Choose representative frame index (middle frame)
        rep_frame_idx = frame_indices[len(frame_indices)//2]
        rep_frame = denormalize_frame(frames_tensor[0, rep_frame_idx])

        # Overlay aggregated heatmap on representative frame
        aggregated_overlay = overlay_heatmap(rep_frame, aggregated_heatmap)

        # Add new figure below or as a separate figure
        plt.figure(figsize=(6, 6))
        plt.suptitle(f'Aggregated Grad-CAM Overlay on Frame {rep_frame_idx+1}', fontsize=14, fontweight='bold')

        plt.subplot(1, 3, 1)
        plt.imshow(rep_frame)
        plt.title(f'Representative Frame {rep_frame_idx+1}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(aggregated_heatmap, cmap='jet')
        plt.title('Aggregated Heatmap')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(aggregated_overlay)
        plt.title('Overlay (Aggregated)')
        plt.axis('off')

        if save_path:
            # Save aggregated visualization as separate file
            base_save_path = Path(save_path)
            agg_save_path = base_save_path.with_name(base_save_path.stem + '_aggregated.png')
            plt.savefig(agg_save_path, dpi=300, bbox_inches='tight')
            print(f"Aggregated visualization saved to {agg_save_path}")

        plt.show()

    # Original title and plot layout for frame-wise visualization
    plt.suptitle(f'Enhanced Grad-CAM Analysis - {num_frames_to_show} Frame Demonstration\nModel: {TRAINED_CONFIG["name"]} | '
                f'Prediction: {label} (Conf: {prob:.3f}) | '
                f'Architecture: Enhanced MesoNet+LSTM',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()

def apply_gradcam_to_frame(model, frame_tensor, target_layer_name='mesonet.conv5'):
    """Apply Grad-CAM to a single frame"""
    # Ensure frame is the correct size (112x112)
    if frame_tensor.shape[-2:] != (112, 112):
        frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(112, 112), mode='bilinear', align_corners=False).squeeze(0)

    # Rest of the function remains the same
    model.eval()
    target_layer = None
    
    # Get the target layer
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        print(f"Could not find layer: {target_layer_name}")
        return None

    # Register hooks
    gradients = []
    activations = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
        output = model(frame_tensor)
        
        if isinstance(output, tuple):
            output = output[0]
        
        # Backward pass
        output.backward()
        
        # Generate Grad-CAM
        if len(gradients) > 0 and activations is not None:
            pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]
            
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            
            if torch.max(heatmap) > 0:
                heatmap = heatmap / torch.max(heatmap)
            
            return heatmap.cpu().numpy()
    finally:
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
    
    return None

class CNNOnlyEnhanced(nn.Module):
    """CNN-only version of the enhanced model for GradCAM visualization"""
    def __init__(self, base_model):
        super().__init__()
        self.mesonet = base_model.mesonet
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.mesonet(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        return self.classifier(pooled)

def create_gradcam_video(frames_dir, output_path, model, frame_files, aggregated_output_path=None):
    """
    Create a video visualization of GradCAM results and save a representative frame.
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Path to save the output video
        model: The model to use for GradCAM
        frame_files: List of frame filenames
        aggregated_output_path: Path to save the representative GradCAM frame
    """
    print("Using enhanced trained model:", model.__class__.__name__)
    print("Configuration:", TRAINED_CONFIG)

    # Create CNN model for Grad-CAM
    cnn_model = CNNOnlyEnhanced(model).to(DEVICE)
    cnn_model.eval()

    # Define standard size for all frames
    STANDARD_SIZE = (128, 128)  # Using model's input size as standard

    # Get original video properties
    original_video_path = output_path.replace('gradcam_', '')
    cap = cv2.VideoCapture(original_video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    gradcam_out = cv2.VideoWriter(output_path, fourcc, original_fps, STANDARD_SIZE)

    # Transform for input
    transform = transforms.Compose([
        transforms.Resize(STANDARD_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create frame mapping dictionary
    frame_mapping = {}
    for frame_file in frame_files:
        frame_num = int(frame_file.split('_')[1].split('.')[0])
        frame_mapping[frame_num] = frame_file

    # Variables for tracking best frame for representative image
    best_frame = None
    best_heatmap = None
    best_activation = -1
    best_frame_idx = None

    # Process original video frame by frame
    cap = cv2.VideoCapture(original_video_path)
    frame_idx = 0
    last_processed_frame = None
    last_heatmap = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            if frame_idx in frame_mapping:
                # Process frame with face
                frame_file = frame_mapping[frame_idx]
                frame_path = os.path.join(frames_dir, frame_file)
                
                # Read and preprocess frame
                face_frame = cv2.imread(frame_path)
                if face_frame is None:
                    continue
                
                # Resize frame to standard size
                face_frame = cv2.resize(face_frame, STANDARD_SIZE)
                
                # Convert to RGB for processing
                frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Transform for model input
                input_tensor = transform(frame_pil).unsqueeze(0).to(DEVICE)
                
                # Generate GradCAM
                gradcam = GradCAM(cnn_model, cnn_model.mesonet.conv5)
                heatmap = gradcam.generate_cam(input_tensor)
                gradcam.remove_hooks()
                
                # Resize heatmap to standard size
                heatmap_resized = cv2.resize(heatmap, STANDARD_SIZE)
                
                # Create colored heatmap
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                
                # Calculate activation strength
                activation_strength = np.mean(heatmap_resized)
                
                # Update best frame if this one has stronger activation
                if activation_strength > best_activation:
                    best_activation = activation_strength
                    best_frame = face_frame.copy()
                    best_heatmap = heatmap_colored.copy()
                    best_frame_idx = frame_idx
                
                # Overlay heatmap on frame
                overlay = cv2.addWeighted(face_frame, 0.6, heatmap_colored, 0.4, 0)
                
                # Save current processed frame and heatmap
                last_processed_frame = face_frame.copy()
                last_heatmap = heatmap_colored.copy()
                
                # Write frame to video
                gradcam_out.write(overlay)
            else:
                # For frames without faces, use the last known frame if available
                if last_processed_frame is not None and last_heatmap is not None:
                    # Create faded overlay for continuity
                    fade_factor = 0.4 * max(0.2, 1.0 - (frame_idx - max(frame_mapping.keys())) * 0.1)
                    overlay = cv2.addWeighted(last_processed_frame, 0.6, last_heatmap, fade_factor, 0)
                    gradcam_out.write(overlay)
                else:
                    # If no previous frame available, write a blank frame
                    blank_frame = np.zeros((STANDARD_SIZE[1], STANDARD_SIZE[0], 3), dtype=np.uint8)
                    gradcam_out.write(blank_frame)
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            # Write last known good frame or blank frame
            if last_processed_frame is not None:
                gradcam_out.write(last_processed_frame)
            else:
                blank_frame = np.zeros((STANDARD_SIZE[1], STANDARD_SIZE[0], 3), dtype=np.uint8)
                gradcam_out.write(blank_frame)

        frame_idx += 1

    # Release video resources
    cap.release()
    gradcam_out.release()

    # Save representative frame if path provided
    if aggregated_output_path and best_frame is not None and best_heatmap is not None:
        try:
            # Create overlay of best frame
            overlay = cv2.addWeighted(best_frame, 0.6, best_heatmap, 0.4, 0)
            
            # Save representative frame
            cv2.imwrite(aggregated_output_path, overlay)
            print(f"Saved representative frame (from frame {best_frame_idx}) to {aggregated_output_path}")
            
        except Exception as e:
            print(f"Error saving representative frame: {str(e)}")
            return False

    return True

def analyze_video_enhanced(img_path, model_path, save_path=None):
    """Analyze a single frame using the enhanced model"""
    print(f"Using enhanced trained model: {model_path}")
    print(f"Configuration: {TRAINED_CONFIG}")

    # Load model
    model = load_trained_model(model_path)
    if model is None:
        return None

    # Load and preprocess image
    try:
        image = Image.open(img_path).convert('RGB')
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Get prediction
        print("Getting model prediction...")
        with torch.no_grad():
            output = model(input_tensor)
            prob = output.item()
            prediction = "FAKE" if prob > 0.5 else "REAL"
            confidence = prob * 100 if prediction == "FAKE" else (1 - prob) * 100
            
        print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")
        
        return {
            'prediction': prediction,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return None 