import torch
import torch.nn as nn
from self_attention_cv.transunet import TransUnet

def TransUNet():
    """
    Returns the exact TransUNet architecture skeleton used during training.
    We don't need optimizers or loss functions for inference!
    """
    model = TransUnet(
        in_channels=3, 
        classes=2, 
        img_dim=224, 
        vit_blocks=8, 
        vit_dim_linear_mhsa_block=512
    )
    
    return model