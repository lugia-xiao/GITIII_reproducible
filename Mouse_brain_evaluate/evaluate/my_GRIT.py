import torch
import torch.nn as nn

from attention import GRIT_encoder,GRIT_encoder_last_layer
from embedding import Embedding

class My_GRIT(nn.Module):
    def __init__(self,genes,ligands_info,node_dim,edge_dim,num_heads,n_layers,node_dim_small=16,att_dim=8):
        super().__init__()
        self.embeddings=Embedding(genes,ligands_info,node_dim,edge_dim)
        self.encoders=nn.ModuleList([
            GRIT_encoder(node_dim,edge_dim,num_heads,att_dim) for i in range(n_layers-1)
        ])
        self.last_layer=GRIT_encoder_last_layer(node_dim, len(genes), edge_dim, node_dim_small, att_dim)

    def forward(self,x):
        x=self.embeddings(x)
        for encoderi in self.encoders:
            x=encoderi(x)
        x=self.last_layer(x)
        return x
