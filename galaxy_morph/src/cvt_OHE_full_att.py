from torch._tensor import Tensor
import pandas as pd
import numpy as np
import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import cv2
import os

class conv_embedd(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, stride, padding):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x)
        H_out, W_out = x.shape[2], x.shape[3]
        x = einops.rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = self.norm(x)
        return x, H_out, W_out

class CoarseLabelEmbed(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.norm(x)
        return x

class multi_head_attention(nn.Module):
    def __init__(self, in_dim, num_heads, kernel_size=3, with_cls_token=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.num_heads = num_heads
        self.with_cls_token = with_cls_token
        self.head_dim = in_dim // num_heads
        
        # Separate convolutions for Q, K, V
        self.q_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, stride=1, groups=num_heads),
            nn.BatchNorm2d(in_dim),
            Rearrange('b c h w -> b (h w) c')
        )
        self.k_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, stride=1, groups=num_heads),
            nn.BatchNorm2d(in_dim),
            Rearrange('b c h w -> b (h w) c')
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, stride=1, groups=num_heads),
            nn.BatchNorm2d(in_dim),
            Rearrange('b c h w -> b (h w) c')
        )
        
        self.att_drop = nn.Dropout(0.1)
        self.attention_map = None

    def forward_conv(self, x):
        B, hw, C = x.shape

        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, hw-1], 1)
        else:
            cls_token = None

        H = W = int(x.shape[1]**0.5)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        if self.with_cls_token:
            assert cls_token is not None
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        
        q = einops.rearrange(q, 'b t (H d) -> b H t d', H=self.num_heads, d=self.head_dim)
        k = einops.rearrange(k, 'b t (H d) -> b H t d', H=self.num_heads, d=self.head_dim)
        v = einops.rearrange(v, 'b t (H d) -> b H t d', H=self.num_heads, d=self.head_dim)

        att_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att_score = torch.softmax(att_score, dim=-1)
        
        self.attention_map = att_score.detach().clone()
        
        att_score = self.att_drop(att_score)
        x = att_score @ v
        x = einops.rearrange(x, 'b H t d -> b t (H d)')

        return x

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*dim, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return self.ff(x)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, with_cls_token):
        super().__init__()
        self.mhsa = multi_head_attention(embed_dim, num_heads, with_cls_token=with_cls_token)
        self.ff = MLP(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.mhsa(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        self.attention_map = self.mhsa.attention_map
        return x

class VisionTransformer(nn.Module):
    def __init__(self, depth, embed_dim, num_heads, patch_size, stride, padding, in_ch=3, cls_token=False, coarse_labels=None):
        super().__init__()
        self.stride = stride
        self.cls_token = cls_token
        self.coarse_labels = coarse_labels
        
        self.layers = nn.ModuleList([Block(embed_dim, num_heads, self.cls_token) for _ in range(depth)])
        self.embedding = conv_embedd(in_ch, embed_dim, patch_size, stride, padding)
        
        if self.coarse_labels is not None:
            self.coarse_embed = CoarseLabelEmbed(len(self.coarse_labels), embed_dim)
            self.merge_coarse = nn.Linear(2*embed_dim, embed_dim)

        if self.cls_token:
            self.cls_token_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, coarse_label=None, output_attentions=False, output_hidden_states=False):
        B = x.shape[0]
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        rgb = x
        rgb, h_out, w_out = self.embedding(rgb)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (rgb,)

        if coarse_label is not None and hasattr(self, 'coarse_embed'):
            coarse_emb = self.coarse_embed(coarse_label)
            coarse_emb_expanded = coarse_emb.unsqueeze(1).expand(-1, rgb.shape[1], -1)
            fused = torch.cat([rgb, coarse_emb_expanded], dim=-1)
            rgb = self.merge_coarse(fused)

        if self.cls_token:
            cls_token = einops.repeat(self.cls_token_embed, '1 1 e -> b 1 e', b=B).contiguous()
            rgb = torch.cat([cls_token, rgb], dim=1)
        
        for layer in self.layers:
            rgb = layer(rgb)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (rgb,)
            if output_attentions and hasattr(layer, 'attention_map'):
                all_attentions = all_attentions + (layer.mhsa.attention_map,)
        
        if not self.cls_token:
            rgb = einops.rearrange(rgb, 'b (h w) c -> b c h w', h=h_out, w=w_out).contiguous()

        return VisionTransformerOutput(
            last_hidden_state=rgb,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

class VisionTransformerOutput:
    def __init__(self, last_hidden_state, hidden_states=None, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions

class CvT_cyclic(nn.Module):
    def __init__(self, embed_dim, num_class, hint=False, coarse_labels=None):
        super().__init__()
        self.hint = hint
        self.coarse_labels = coarse_labels
        
        self.stage1 = VisionTransformer(depth=1,
                                     embed_dim=64,
                                     num_heads=1,
                                     patch_size=7,
                                     stride=4,
                                     padding=2,
                                     in_ch=3,
                                     coarse_labels=coarse_labels
                                     )
        self.stage2 = VisionTransformer(depth=2,
                                     embed_dim=192,
                                     num_heads=3,
                                     patch_size=3,
                                     stride=2,
                                     padding=1,
                                     in_ch=64,
                                     coarse_labels=coarse_labels)
        self.stage3 = VisionTransformer(depth=10,
                                     embed_dim=384,
                                     num_heads=6,
                                     patch_size=3,
                                     stride=2,
                                     padding=1,
                                     in_ch=192,
                                     cls_token=True,
                                     coarse_labels=coarse_labels)
        
        self.ff = nn.Sequential(
            nn.Linear(6*embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_class)
        )

    def forward(self, x, coarse_label=None, output_attentions=False, output_hidden_states=False):
        stage1_output = self.stage1(x, coarse_label, output_attentions, output_hidden_states)
        stage2_output = self.stage2(stage1_output.last_hidden_state, coarse_label, output_attentions, output_hidden_states)
        stage3_output = self.stage3(stage2_output.last_hidden_state, coarse_label, output_attentions, output_hidden_states)

        x = stage3_output.last_hidden_state[:, 0, :]
        logits = self.ff(x)

        return CvTOutput(
            logits=logits,
            hidden_states=(
                stage1_output.hidden_states,
                stage2_output.hidden_states,
                stage3_output.hidden_states
            ) if output_hidden_states else None,
            attentions=(
                stage1_output.attentions,
                stage2_output.attentions,
                stage3_output.attentions
            ) if output_attentions else None
        )

class CvTOutput:
    def __init__(self, logits, hidden_states=None, attentions=None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions 

def cvt_attention_map(model, data_loader, device, dest_dir=None, sel_gal_ids=None):
    """
    Extract and visualize attention maps from CvT model
    Args:
        model: CvT model
        data_loader: DALI data loader
        device: gpu or cpu
        dest_dir: destination directory to save the maps
        sel_gal_ids: list of galaxy IDs to visualize (optional)
    """
    model = model.to(device)
    model.eval()

    # Galaxy categories
    gxy_labels = ['Er', 'Ei', 'Ec', 'Seb', 'Sen', 'S', 'SB']

    def process_attention(attention_maps, batch_idx, stage_idx):
        """Helper function to process attention maps from a stage"""
        if not attention_maps:
            return None
        
        # Calculate spatial dimensions based on stage
        # Stage 1: 224/4 = 56, Stage 2: 56/2 = 28, Stage 3: 28/2 = 14
        if stage_idx == 0:
            h = w = 56  # Stage 1
        elif stage_idx == 1:
            h = w = 28  # Stage 2
        else:
            h = w = 14  # Stage 3
            
        # Average over all layers in the stage
        stage_attention = torch.stack([attn[batch_idx] for attn in attention_maps])
        # Average over heads and layers
        avg_attention = stage_attention.mean(dim=(0, 1))  # Now shape (seq_len, seq_len)
        
        # For Stage 3, we need to handle the CLS token
        if stage_idx == 2:
            # Remove CLS token attention (first row and column)
            avg_attention = avg_attention[1:, 1:]
        
        # Calculate attention rollout
        # First, ensure we're working with proper attention scores
        attention_probs = torch.softmax(avg_attention, dim=-1)
        
        # For visualization, we want to know how much each token attends to others
        # We can sum over the rows to get the total attention received by each token
        attention_scores = attention_probs.sum(dim=0)
        
        # Reshape to spatial dimensions
        try:
            spatial_attention = attention_scores.reshape(h, w)
        except RuntimeError as e:
            print(f"Debug - Stage {stage_idx + 1}:")
            print(f"Expected shape: ({h}, {w})")
            print(f"Actual tensor size: {attention_scores.numel()}")
            print(f"Attention tensor shape: {attention_scores.shape}")
            raise e
        
        # Normalize attention weights
        spatial_attention = (spatial_attention - spatial_attention.min()) / (spatial_attention.max() - spatial_attention.min() + 1e-8)
        
        return spatial_attention.cpu().numpy()

    for data in data_loader:
        images = data[0]['data'].to(device)
        labels = data[0]['label'].to(device).view(-1).long()
        galaxy_ids = data[0]['galaxy_id']

        with torch.no_grad():
            # Get model outputs with attention
            outputs = model(images, output_attentions=True)
            stage1_attns, stage2_attns, stage3_attns = outputs.attentions

            # Process each image in the batch
            for idx in range(images.shape[0]):
                img = images[idx].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                
                gal_id = galaxy_ids[idx]
                if sel_gal_ids and gal_id not in sel_gal_ids:
                    continue

                # Create figure for visualization
                fig, axs = plt.subplots(3, 3, figsize=(15, 15))
                
                # Original image
                axs[0, 0].imshow(img)
                axs[0, 0].set_title(f"ID: {gal_id}\nTrue: {gxy_labels[labels[idx]]}", fontsize=12)
                axs[0, 0].axis('off')

                # Process attention maps from each stage
                attn_map1 = process_attention(stage1_attns, idx, 0)  # Stage 1
                attn_map2 = process_attention(stage2_attns, idx, 1)  # Stage 2
                attn_map3 = process_attention(stage3_attns, idx, 2)  # Stage 3

                # Stage 1 attention
                if attn_map1 is not None:
                    attn_map1_resized = cv2.resize(attn_map1, (224, 224))
                    axs[0, 1].imshow(attn_map1_resized, cmap='jet')
                    axs[0, 1].set_title("Stage 1 Attention", fontsize=12)
                    axs[0, 1].axis('off')

                    # Stage 1 attention overlay
                    attn_mask1 = np.stack([attn_map1_resized]*3, axis=2)
                    overlay1 = img * attn_mask1
                    axs[0, 2].imshow(overlay1)
                    axs[0, 2].set_title("Stage 1 Overlay", fontsize=12)
                    axs[0, 2].axis('off')

                # Stage 2 attention
                if attn_map2 is not None:
                    attn_map2_resized = cv2.resize(attn_map2, (224, 224))
                    axs[1, 0].imshow(attn_map2_resized, cmap='jet')
                    axs[1, 0].set_title("Stage 2 Attention", fontsize=12)
                    axs[1, 0].axis('off')

                    # Stage 2 attention overlay
                    attn_mask2 = np.stack([attn_map2_resized]*3, axis=2)
                    overlay2 = img * attn_mask2
                    axs[1, 1].imshow(overlay2)
                    axs[1, 1].set_title("Stage 2 Overlay", fontsize=12)
                    axs[1, 1].axis('off')

                # Stage 3 attention
                if attn_map3 is not None:
                    attn_map3_resized = cv2.resize(attn_map3, (224, 224))
                    axs[1, 2].imshow(attn_map3_resized, cmap='jet')
                    axs[1, 2].set_title("Stage 3 Attention", fontsize=12)
                    axs[1, 2].axis('off')

                    # Stage 3 attention overlay
                    attn_mask3 = np.stack([attn_map3_resized]*3, axis=2)
                    overlay3 = img * attn_mask3
                    axs[2, 0].imshow(overlay3)
                    axs[2, 0].set_title("Stage 3 Overlay", fontsize=12)
                    axs[2, 0].axis('off')

                # Combined attention
                if all(m is not None for m in [attn_map1, attn_map2, attn_map3]):
                    attn_map1_resized = cv2.resize(attn_map1, (224, 224))
                    attn_map2_resized = cv2.resize(attn_map2, (224, 224))
                    attn_map3_resized = cv2.resize(attn_map3, (224, 224))
                    combined_attn = (attn_map1_resized + attn_map2_resized + attn_map3_resized) / 3
                    axs[2, 1].imshow(combined_attn, cmap='jet')
                    axs[2, 1].set_title("Combined Attention", fontsize=12)
                    axs[2, 1].axis('off')

                    # Combined overlay
                    combined_mask = np.stack([combined_attn]*3, axis=2)
                    combined_overlay = img * combined_mask
                    axs[2, 2].imshow(combined_overlay)
                    axs[2, 2].set_title("Combined Overlay", fontsize=12)
                    axs[2, 2].axis('off')

                plt.tight_layout()

                if dest_dir:
                    os.makedirs(dest_dir, exist_ok=True)
                    plt.savefig(os.path.join(dest_dir, f"G{gal_id}_AttnMaps.png"))
                    plt.close(fig)
                else:
                    plt.show()

                if not sel_gal_ids and not dest_dir:
                    break
                break
            break