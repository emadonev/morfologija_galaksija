from torch._tensor import Tensor
from torch._tensor import Tensor
import pandas as pd
import numpy as np

import einops
from einops.layers.torch import Rearrange

np.random.seed(42)

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import os
import cv2
import math
# ---------------------

class conv_embedd(nn.Module):
    '''
    This class handles the convolutional embedding needed for the CvT architecture
    '''
    def __init__(self, in_ch, embed_dim, patch_size, stride, padding):
        super().__init__() # initalization
        # creating a conv layer which will embed the input image into tokens
        self.embed = nn.Sequential(
        nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
    )
        # normalization layer
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x) #x has the shape of B C H W
        H_out, W_out = x.shape[2], x.shape[3]
        x = einops.rearrange(x, 'b c h w -> b (h w) c').contiguous() # rearrange to token shape
        x = self.norm(x) # normalization layer
        return x, H_out, W_out

class multi_head_attention(nn.Module):
    '''
    This class handles the attention part of the network
    '''
    def __init__(self, in_dim, num_heads, kernel_size=3, with_cls_token=False):
        super().__init__() # initialization
        padding = (kernel_size - 1) // 2 # define padding
        self.num_heads = num_heads # number of heads for attention
        self.with_cls_token = with_cls_token # if there is a class token
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
        
        # define the dropout rate for attention not to overfit
        self.att_drop = nn.Dropout(0.1)
        self.attention_map = None  # Initialize attention_map as None

    def forward_conv(self, x):
        B, hw, C = x.shape # assigning the batch size, image dimensions and number of channels

        if self.with_cls_token: # if there is a class token in the image
            cls_token, x = torch.split(x, [1, hw-1], 1)
        else:
            cls_token = None

        H = W = int(x.shape[1]**0.5) # getting the height and width of the image
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous() # rearranging the token into an image

        # Apply separate convolutions
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        if self.with_cls_token: # if a class token is present, concatenate that with the q, k, v matrices
            assert cls_token is not None
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x):
        # go through the process described above
        q, k, v = self.forward_conv(x)
        
        # Split heads while preserving spatial structure
        q = einops.rearrange(q, 'b t (H d) -> b H t d', H=self.num_heads, d=self.head_dim)
        k = einops.rearrange(k, 'b t (H d) -> b H t d', H=self.num_heads, d=self.head_dim)
        v = einops.rearrange(v, 'b t (H d) -> b H t d', H=self.num_heads, d=self.head_dim)

        # Scale dot-product attention
        att_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply softmax
        att_score = torch.softmax(att_score, dim=-1)
        
        # Store attention map before dropout
        self.attention_map = att_score.detach().clone()
        
        # Apply dropout
        att_score = self.att_drop(att_score)

        # Apply attention to values
        x = att_score @ v
        
        # Merge heads
        x = einops.rearrange(x, 'b H t d -> b t (H d)')

        return x

class MLP(nn.Module):
    '''
    This class is the multi layered perceptron (fancy work for linear network) of the model
    '''
    def __init__(self, dim):
        super().__init__() # init
        # fully connected layers
        self.ff = nn.Sequential(
        nn.Linear(dim, 4*dim), # linear layer
        nn.GELU(), # special ReLU function 
        nn.Dropout(0.1), # dropout to prevent overfitting
        nn.Linear(4*dim, dim), # another linear layer
        nn.Dropout(0.1) # dropout to prevent overfitting
        )
        
    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
  '''
  This class creates the block of the network - the transformer block
  '''
  def __init__(self, embed_dim, num_heads, with_cls_token):
    super().__init__() # init

    # multi head attention
    self.mhsa = multi_head_attention(embed_dim, num_heads, with_cls_token=with_cls_token)
    # followed by the multi layered perceptron
    self.ff = MLP(embed_dim)
    # one norm layer
    self.norm1 = nn.LayerNorm(embed_dim)
    # second norm layer
    self.norm2 = nn.LayerNorm(embed_dim)
    # dropout to prevent overfitting
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    # first part of the block is the attention process and then layer normalization
    x = x + self.dropout(self.mhsa(self.norm1(x)))
    # the second part of the block is the MLP and layer of normalization
    x = x + self.dropout(self.ff(self.norm2(x)))

    self.attention_map = self.mhsa.attention_map

    return x
  
class VisionTransformer(nn.Module):
  '''
  This class creates the actual vision transformer component
  '''
  def __init__(self, depth, embed_dim, num_heads, patch_size, stride, padding, in_ch=3, cls_token=False):
    super().__init__() # init

    self.stride = stride # stride init
    self.cls_token = cls_token # token init 
    # creating the layers of the network - creates the number of transformer blocks depending on how deep we want to go into the image
    self.layers = nn.ModuleList([Block(embed_dim, num_heads, self.cls_token) for _ in range(depth)])
    # defined the convolutional embedding of the rgb component of the image
    self.embedding = conv_embedd(in_ch, embed_dim, patch_size, stride, padding)

    if self.cls_token: # if we have a cls token, create an embedding for it
       self.cls_token_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

  def forward(self, x, output_attentions=False, output_hidden_states=False):
    B = x.shape[0]
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    rgb = x
    rgb, h_out, w_out = self.embedding(rgb)  # conv embedding

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (rgb,)

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

class CvT_bench(nn.Module):

  '''
  This class combines everything together for the final CvT architecture
  '''
  def __init__(self, embed_dim, num_class, hint=False):
    super().__init__()
    self.hint = hint
    # first stage of the network
    self.stage1 = VisionTransformer(depth=1,
                                     embed_dim=64,
                                     num_heads=1,
                                     patch_size=7,
                                     stride=4,
                                     padding=2,
                                     in_ch=3
                                     )
    # second stage of the network
    self.stage2 = VisionTransformer(depth=2,
                                     embed_dim=192,
                                     num_heads=3,
                                     patch_size=3,
                                     stride=2,
                                     padding = 1,
                                     in_ch = 64)
    # third stage of the network
    self.stage3 = VisionTransformer(depth=10,
                                     embed_dim=384,
                                     num_heads=6,
                                     patch_size=3,
                                     stride=2,
                                     padding=1,
                                     in_ch=192,
                                     cls_token=True)
    # final linear layer to create an output
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

  def forward(self, x, output_attentions=False, output_hidden_states=False):
    # Process through each stage
    stage1_output = self.stage1(x, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    stage2_output = self.stage2(stage1_output.last_hidden_state, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    stage3_output = self.stage3(stage2_output.last_hidden_state, output_attentions=output_attentions, output_hidden_states=output_hidden_states)

    # Get the [CLS] token output from stage3
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

def cvt_attention_map(model, data_loader, device, gxy_labels, dest_dir=None, sel_gal_ids=None):
    """
    Extract and visualize attention maps from CvT model
    Args:
        model: CvT model
        data_loader: DALI data loader
        device: gpu or cpu
        dest_dir: destination directory to save the maps
        sel_gal_ids: list of galaxy IDs to visualize (optional)
    """
    print(f"Starting attention map visualization for {len(sel_gal_ids) if sel_gal_ids else 'all'} galaxies")
    print(f"Destination directory: {dest_dir}")
    
    # Ensure model is in half precision and on the correct device
    model = model.half().to(device)
    model.eval()

    # Galaxy categories

    def process_attention(attention_maps, batch_idx, stage_idx):
        """Helper function to process attention maps from a stage"""
        if not attention_maps:
            print(f"No attention maps for stage {stage_idx + 1}")
            return None
        
        try:
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
            
            # Convert to numpy and ensure float32
            spatial_attention = spatial_attention.cpu().numpy().astype(np.float32)
            
            return spatial_attention
        except Exception as e:
            print(f"Error processing attention maps for stage {stage_idx + 1}: {str(e)}")
            return None

    processed_galaxies = set()
    for data in data_loader:
        # Convert inputs to half precision
        images = data[0]['data'].to(device).half()
        labels = data[0]['label'].to(device).view(-1).long()
        galaxy_ids = data[0]['galaxy_id']

        print(f"Processing batch with galaxy IDs: {galaxy_ids}")

        with torch.no_grad():
            # Get model outputs with attention
            outputs = model(images, output_attentions=True)
            stage1_attns, stage2_attns, stage3_attns = outputs.attentions

            # Process each image in the batch
            for idx in range(images.shape[0]):
                try:
                    # Convert image to float32 and normalize to [0,1]
                    img = images[idx].cpu().numpy()
                    img = np.transpose(img, (1, 2, 0))
                    img = img.astype(np.float32)
                    
                    gal_id = galaxy_ids[idx]
                    print(f"Processing galaxy ID: {gal_id}")
                    
                    if sel_gal_ids and gal_id not in sel_gal_ids:
                        print(f"Skipping galaxy {gal_id} - not in selected list")
                        continue
                    
                    if gal_id in processed_galaxies:
                        print(f"Skipping galaxy {gal_id} - already processed")
                        continue
                        
                    processed_galaxies.add(gal_id)
                    print(f"Generating attention maps for galaxy {gal_id}")

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
                        try:
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
                        except Exception as e:
                            print(f"Error processing stage 1 attention: {str(e)}")

                    # Stage 2 attention
                    if attn_map2 is not None:
                        try:
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
                        except Exception as e:
                            print(f"Error processing stage 2 attention: {str(e)}")

                    # Stage 3 attention
                    if attn_map3 is not None:
                        try:
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
                        except Exception as e:
                            print(f"Error processing stage 3 attention: {str(e)}")

                    # Combined attention
                    if all(m is not None for m in [attn_map1, attn_map2, attn_map3]):
                        try:
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
                        except Exception as e:
                            print(f"Error processing combined attention: {str(e)}")

                    plt.tight_layout()

                    if dest_dir:
                        os.makedirs(dest_dir, exist_ok=True)
                        output_path = os.path.join(dest_dir, f"G{gal_id}_AttnMaps.png")
                        print(f"Saving attention maps to: {output_path}")
                        plt.savefig(output_path)
                        plt.close(fig)
                    else:
                        plt.show()

                    # Check if we've processed all selected galaxies
                    if sel_gal_ids and len(processed_galaxies) >= len(sel_gal_ids):
                        print("All selected galaxies processed")
                        return
                except Exception as e:
                    print(f"Error processing galaxy {gal_id}: {str(e)}")
                    continue