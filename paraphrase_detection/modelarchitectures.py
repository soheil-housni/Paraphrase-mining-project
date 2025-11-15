import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer
import numpy as np


class SBERTPairClassifier(nn.Module):
    def __init__(self, model : SentenceTransformer, fc_sizes : list, device, dropout : float = 0.1):
        super().__init__()
        self.sbert = model
        self.dropout = dropout
        self.device = device
        OUPUT_DIM = 2
        EMB_DIM = self.sbert.get_sentence_embedding_dimension()
        layers = [nn.Linear(3 * EMB_DIM, fc_sizes[0])]
        for layer in range(len(fc_sizes)):
            if layer == len(fc_sizes) - 1:
                layers.append(nn.Linear(fc_sizes[layer], OUPUT_DIM))
            else:
                layers.append(nn.Linear(fc_sizes[layer], fc_sizes[layer + 1]))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x0 : np.ndarray, x1 : np.ndarray):
        emb1 = self.sbert.encode(sentences = x0, convert_to_tensor = True).to(self.device)
        emb2 = self.sbert.encode(sentences = x1, convert_to_tensor = True).to(self.device)
        abs_diff = torch.abs(emb1 - emb2)
        x = torch.cat([emb1, emb2, abs_diff], dim=1)
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
            x = nn.functional.dropout(x, p = self.dropout)
        logits = self.layers[-1](x)
        return logits