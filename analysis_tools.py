import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        # Support single layer or list of layers (optionally with names)
        self.target_layers = []
        self.layer_names = {}
        if isinstance(target_layers, list):
            for item in target_layers:
                if isinstance(item, tuple) and len(item) == 2:
                    name, layer = item
                    self.target_layers.append(layer)
                    self.layer_names[layer] = name
                else:
                    self.target_layers.append(item)
        else:
            self.target_layers = [target_layers]
        self.gradients = {}
        self.activations = {}
        
        # Register hooks for each layer
        for layer in self.target_layers:
            layer.register_forward_hook(self.save_activation(layer))
            layer.register_full_backward_hook(self.save_gradient(layer))

    def save_activation(self, layer):
        def hook(module, input, output):
            self.activations[layer] = output
        return hook

    def save_gradient(self, layer):
        def hook(module, grad_input, grad_output):
            # grad_output is a tuple, we take the first element
            self.gradients[layer] = grad_output[0]
        return hook

    def __call__(
        self,
        images,
        input_ids,
        attention_mask,
        target_class_idx=None,
        return_layer_cams=False,
        return_avg=True,
        **model_kwargs,
    ):
        # Forward pass
        self.model.zero_grad()
        logits = self.model(images, input_ids, attention_mask, **model_kwargs)
        
        if target_class_idx is None:
            target_class_idx = logits.argmax(dim=1)
        
        # Create one-hot encoding for the target class
        one_hot = torch.zeros_like(logits)
        for i in range(logits.size(0)):
            one_hot[i][target_class_idx[i]] = 1
            
        # Backward pass
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAMs
        final_cams = []
        per_layer_cams = []
        img_h, img_w = images.shape[2], images.shape[3]
        
        for i in range(images.size(0)):
            layer_cams = []
            layer_cam_map = {}
            for layer_idx, layer in enumerate(self.target_layers):
                grad = self.gradients[layer][i].cpu().data.numpy() # (C, H, W)
                act = self.activations[layer][i].cpu().data.numpy() # (C, H, W)
                
                # Global Average Pooling on gradients to get weights
                weights = np.mean(grad, axis=(1, 2)) # (C,)
                
                # Weighted combination of activations
                cam = np.zeros(act.shape[1:], dtype=np.float32)
                for j, w in enumerate(weights):
                    cam += w * act[j]
                    
                # ReLU
                cam = np.maximum(cam, 0)
                
                # Normalize per layer
                if np.max(cam) > 0:
                    cam = cam / np.max(cam)
                
                # Resize to original image size
                cam = cv2.resize(cam, (img_w, img_h))
                layer_cams.append(cam)
                layer_name = self.layer_names.get(layer, f"layer_{layer_idx}")
                layer_cam_map[layer_name] = cam
            
            # Average across all layers
            if return_avg:
                if layer_cams:
                    avg_cam = np.mean(layer_cams, axis=0)
                    # Re-normalize the averaged CAM
                    if np.max(avg_cam) > 0:
                        avg_cam = avg_cam / np.max(avg_cam)
                    final_cams.append(avg_cam)
                else:
                    final_cams.append(np.zeros((img_h, img_w), dtype=np.float32))

            if return_layer_cams:
                per_layer_cams.append(layer_cam_map)

        if return_layer_cams and return_avg:
            return final_cams, per_layer_cams, target_class_idx
        if return_layer_cams:
            return per_layer_cams, target_class_idx
        return final_cams, target_class_idx

def visualize_cam(image_tensor, cam, save_path, alpha=0.5):
    """
    image_tensor: (C, H, W) normalized tensor
    cam: (H, W) numpy array
    """
    # Denormalize image for visualization (assuming ImageNet mean/std)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Superimpose
    result = heatmap * alpha + img * (1 - alpha)
    cv2.imwrite(save_path, result)

class FeatureRankAnalyzer:
    def __init__(self, model):
        self.model = model
        self.features = []
        
        # Hook into the fusion output (input to classifier)
        # We need to find where the fusion output is. 
        # In MultimodalBaselineModel.forward:
        # fused_features = self.fusion(...)
        # logits = self.classifier(fused_features)
        # So we can hook the classifier's first layer or the fusion module.
        # Let's hook the fusion module's forward.
        self.model.fusion.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        # output is (B, hidden_dim) or similar
        self.features.append(output.detach().cpu())

    def compute_rank(self):
        if not self.features:
            return None, None
            
        # Concatenate all batches
        all_features = torch.cat(self.features, dim=0) # (N, D)
        
        # Compute SVD
        # Center the features? Usually yes for covariance analysis, but for raw rank SVD is fine.
        # Let's center it.
        all_features = all_features - all_features.mean(dim=0, keepdim=True)
        
        U, S, V = torch.linalg.svd(all_features)
        
        # Effective Rank (e.g., number of singular values > threshold)
        # Or just return singular values distribution
        singular_values = S.numpy()
        
        # Normalize singular values
        singular_values_norm = singular_values / singular_values.max()
        
        return all_features, singular_values_norm

    def plot_singular_values(self, singular_values, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(singular_values, marker='o')
        plt.title("Singular Value Distribution (Log Scale)")
        plt.yscale('log')
        plt.xlabel("Index")
        plt.ylabel("Normalized Singular Value")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def clear(self):
        self.features = []
