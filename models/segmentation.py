import torch.nn as nn
import torch

class SegmentationHead(nn.Module):
    def __init__(self, encoder, in_channels = 5, embedding_size = 256, num_classes = 32, hidden_dim = 16):
        super(SegmentationHead, self).__init__()
        self.encoder = encoder
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.mlp_in = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(embedding_size + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        # append embedding to all x
        x = self.mlp_in(x)
        x = torch.cat([x, embedding.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)
        x = self.mlp_out(x)
        return x
        