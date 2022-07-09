import clip
import torch
import torch.nn as nn

class ImageEncoder(nn.Module):

    def __init__(self, device):
        super(ImageEncoder, self).__init__()
        self.encoder, _ = clip.load("ViT-B/16", device=device)   # loads already in eval mode
        
    def forward(self, x):
        """
        Expects a tensor of size (batch_size, 3, 224, 224)
        """
        with torch.no_grad():
            x = x.type(self.encoder.visual.conv1.weight.dtype)
            x = self.encoder.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.encoder.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.encoder.visual.positional_embedding.to(x.dtype)
            x = self.encoder.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.encoder.visual.transformer(x)
            grid_feats = x.permute(1, 0, 2)  # LND -> NLD    (N, 197, 768)
            grid_feats = self.encoder.visual.ln_post(grid_feats[:,1:])  
                
        return grid_feats.float()
