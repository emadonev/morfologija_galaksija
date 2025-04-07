# CREATING A CVT ARCHITECTURE

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
import torch.nn.functional as F

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
    
# class chint_embed(nn.Module):
#     '''
#     This class handles the auxiliary channel information for prolonged classification
#     '''
#     def __init__(self, c, embed_dim):
#         super().__init__() # initalization
#         # creating a simple linear layer to embed the auxiliary channel information
#         self.embed = nn.Sequential(
#             nn.Linear(c, embed_dim)
#         )
#         # layer norm
#         self.norm = nn.LayerNorm(embed_dim)
    
#     def forward(self, x):
#         # feeding it forward
#         x = self.embed(x)
#         x = self.norm(x)
#         return x

class CoarseLabelEmbed(nn.Module):
    '''
    This class handles embedding one-hot encoded coarse labels into feature space
    '''
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        # Linear layer to project one-hot encoded vector to embedding space
        self.embed = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.norm(x)
        return x

class multi_head_attention(nn.Module):
    '''
    This class handles the attention part of the network
    '''
    def __init__(self, in_dim, num_heads, kernel_size=3, with_cls_token=False):
        super().__init__() # initialization
        padding = (kernel_size - 1) // 2 # define padding
        self.num_heads = num_heads # number of heads for attention
        self.with_cls_token = with_cls_token # if there is a class token
        # defining how the process of attention with q, k, v matrices will be done
        self.conv = nn.Sequential(
           # conv layer
            nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, stride=1),
            # batch norm
            nn.BatchNorm2d(in_dim),
            # rearranging the image back into token form
            Rearrange('b c h w -> b (h w) c')
        )
        # define the dropout rate for attention not to overfit
        self.att_drop = nn.Dropout(0.11)

    def forward_conv(self, x):
        # taking in the image and applying convolutions to the query, key and value matrices in order to gain information
        # for the attention process

        B, hw, C = x.shape # assigning the batch size, image dimensions and number of channels

        if self.with_cls_token: # if there is a class token in the image
            cls_token, x = torch.split(x, [1, hw-1], 1)
        else:
            cls_token = None

        H = W = int(x.shape[1]**0.5) # getting the height and width of the image
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous() # rearranging the token into an image

        q = self.conv(x) # query matrix application
        k = self.conv(x) # key matrix application
        v = self.conv(x) # value matrix application

        if self.with_cls_token: # if a class token is present, concatenate that with the q, k, v matrices
            assert cls_token is not None
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x):
        # go through the process described above
        q, k, v = self.forward_conv(x)
        # rearrange the q, k, v matrices into the form of the attention mechanism
        q = einops.rearrange(q, 'b t (d H) -> b H t d', H=self.num_heads)
        k = einops.rearrange(k, 'b t (d H) -> b H t d', H=self.num_heads)
        v = einops.rearrange(v, 'b t (d H) -> b H t d', H=self.num_heads)

        att_score = q@k.transpose(2, 3)/self.num_heads**0.5 # (q_j * k_iT)/sqrt(d_k)
        # softmax the attention scores
        att_score = torch.softmax(att_score, dim=-1)
        # dropout for attention not to overfit
        att_score = self.att_drop(att_score)

        x = att_score@v # multiply every attention value as a certain weight to the value matrix
        x = einops.rearrange(x, 'b H t d -> b t (H d)') # rearrange back to the token form

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
    self.dropout = nn.Dropout(0.11)

  def forward(self, x):
    # first part of the block is the attention process and then layer normalization
    x = x + self.dropout(self.mhsa(self.norm1(x)))
    # the second part of the block is the MLP and layer of normalization
    x = x + self.dropout(self.ff(self.norm2(x)))
    return x
  
class VisionTransformer(nn.Module):
  '''
  This class creates the actual vision transformer component
  '''
  def __init__(self, depth, embed_dim, num_heads, patch_size, stride, padding, in_ch=3, cls_token=False, use_coarse_labels=False, num_coarse_classes=0):
    super().__init__() # init

    self.stride = stride # stride init
    self.cls_token = cls_token # token init
    self.use_coarse_labels = use_coarse_labels
    
    # creating the layers of the network - creates the number of transformer blocks depending on how deep we want to go into the image
    self.layers = nn.Sequential(*[Block(embed_dim, num_heads, self.cls_token) for _ in range(depth)])
    # defined the convolutional embedding of the rgb component of the image
    self.embedding = conv_embedd(in_ch, embed_dim, patch_size, stride, padding)
    # # defined the linear embedding of the auxiliary component of the image - class hint
    # self.hint_embed = chint_embed(1, embed_dim*6)
    # # defining how to merge the cls token and aux token
    # self.merge_lin = nn.Linear(2*embed_dim, embed_dim)
    
    # Add coarse label embedding if needed
    if use_coarse_labels and num_coarse_classes > 0:
        self.coarse_embed = CoarseLabelEmbed(num_coarse_classes, embed_dim)
        self.merge_coarse = nn.Linear(2*embed_dim, embed_dim)

    if self.cls_token: # if we have a cls token, create an embedding for it
       self.cls_token_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

  def forward(self, x, aux_layer=None, coarse_labels=None, ch_out=False):
    B = x.shape[0]
    # aux = None
    
    # if x.shape[1] == 4:
    #   # split the rgb component of the image and the auxiliary component of the image
    #   rgb = x[:, :3, :, :]
    #   aux = x[:, 3:, :, :]

    #   # assign the shape of the rgb part
    #   # reshape the aux layer to be a single channel
    #   aux = aux.mean(dim=[2, 3])  # shape: [B, 1]
    #   # embed the aux layer
    #   aux = self.hint_embed(aux)  # shape: [B, embed_dim]
    #   # reshape the aux layer so we can concatenate with cls_token
    #   aux = aux.unsqueeze(1)      # shape: [B, 1, embed_dim]
    # else:
    rgb = x

    rgb, h_out, w_out = self.embedding(rgb)  # conv embedding
    
    if self.cls_token:
      cls_token = einops.repeat(self.cls_token_embed, '1 1 e -> b 1 e', b=B).contiguous()
      
      # Handle coarse label embedding if provided
      if self.use_coarse_labels and coarse_labels is not None:
          # Embed the coarse labels
          coarse_embed = self.coarse_embed(coarse_labels)  # [B, embed_dim]
          coarse_embed = coarse_embed.unsqueeze(1)  # [B, 1, embed_dim]
          
          # Merge cls token with coarse label embedding
          merged = torch.cat([cls_token, coarse_embed], dim=-1)
          cls_token = self.merge_coarse(merged)
      
      rgb = torch.cat([cls_token, rgb], dim=1)
      
      # If aux_layer is provided, merge it with the cls token.
      if aux_layer is not None:
          merged = torch.cat([cls_token, aux_layer], dim=-1)
          cls_aux = self.merge_lin(merged)
          # Replace the cls token with the merged version.
          rgb[:, 0, :] = cls_aux.squeeze(1)
    
    rgb = self.layers(rgb)
    
    if not ch_out:
        if self.cls_token:
            rgb = rgb[:, 1:, :]  
        rgb = einops.rearrange(rgb, 'b (h w) c -> b c h w', h=h_out, w=w_out).contiguous()
    
    # if aux is not None:
    #   return rgb, aux
    # else:
    return rgb

class CvT(nn.Module):
  '''
  This class combines everything together for the final CvT architecture
  '''
  def __init__(self, embed_dim, num_class, use_coarse_labels=False, num_coarse_classes=0):
    super().__init__()
    # self.hint = hint
    self.use_coarse_labels = use_coarse_labels
    self.num_coarse_classes = num_coarse_classes
    
    # first stage of the network
    self.stage1 = VisionTransformer(depth=1,
                                     embed_dim=64,
                                     num_heads=1,
                                     patch_size=7,
                                     stride=4,
                                     padding=2
                                     )
    # second stage of the network
    self.stage2 = VisionTransformer(depth=4,
                                     embed_dim=192,
                                     num_heads=3,
                                     patch_size=3,
                                     stride=2,
                                     padding = 1,
                                     in_ch = 64)
    # third stage of the network
    self.stage3 = VisionTransformer(depth=16,
                                     embed_dim=384,
                                     num_heads=6,
                                     patch_size=3,
                                     stride=2,
                                     padding=1,
                                     in_ch=192,
                                     cls_token=True,
                                     use_coarse_labels=use_coarse_labels,
                                     num_coarse_classes=num_coarse_classes)
    # final linear layer to create an output
    self.ff = nn.Sequential(
        nn.Linear(6*embed_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, num_class)
    )

  def forward(self, x, coarse_labels=None):
    x, aux = self.stage1(x)
    x = self.stage2(x)
    
    # if self.hint:
    #   x = self.stage3(x, aux_layer=aux, coarse_labels=coarse_labels, ch_out=True)
    # else:
    x = self.stage3(x, coarse_labels=coarse_labels, ch_out=True)
    
    x = x[:, 0, :]
    x = self.ff(x)
    return x

class TwoStageCvT(nn.Module):
    '''
    This class combines the two CvT models where the first predicts coarse labels
    and the second predicts fine labels using the coarse predictions
    '''
    def __init__(self, embed_dim, coarse_classes=3, fine_classes=7):
        super().__init__()
        
        # First model for coarse classification
        self.coarse_model = CvT(embed_dim, coarse_classes)
        
        # Second model for fine classification that takes coarse predictions
        self.fine_model = CvT(embed_dim, fine_classes, use_coarse_labels=True, num_coarse_classes=coarse_classes)
        
    def forward(self, x, use_true_labels=False, true_coarse_labels=None):
        # Get predictions from first model
        coarse_preds = self.coarse_model(x)
        
        # Convert predictions to one-hot encoding
        if use_true_labels and true_coarse_labels is not None:
            # Use ground truth for training
            coarse_one_hot = true_coarse_labels
        else:
            # Use model predictions at inference time
            coarse_one_hot = F.one_hot(torch.argmax(coarse_preds, dim=1), num_classes=coarse_preds.shape[1]).float()
        
        # Pass image and one-hot encoded coarse labels to fine model
        fine_preds = self.fine_model(x, coarse_labels=coarse_one_hot)
        
        return coarse_preds, fine_preds 