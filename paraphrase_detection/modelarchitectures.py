import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer

class PairClassifier():
    class CrossEntropy(nn.Module):
        def __init__(self, model : SentenceTransformer, fc_sizes : list, use_n_layers : int, device, fixed : bool, dropout : float = 0.1):
            super().__init__()
            self.sbert = model
            self.dropout = dropout
            self.device = device
            self.fixed = fixed
            
            OUTPUT_DIM = 2
            EMB_DIM = model.get_sentence_embedding_dimension()

            hidden_sizes = fc_sizes[:use_n_layers]
            layers = []
            layers.append(nn.Linear(3 * EMB_DIM, hidden_sizes[0]))
            norm_layers = [nn.LayerNorm(fc_sizes[0])]
            for i in range(use_n_layers - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                norm_layers.append(nn.LayerNorm(fc_sizes[i + 1]))
            layers.append(nn.Linear(hidden_sizes[-1], OUTPUT_DIM))

            self.layers = nn.ModuleList(layers)
            self.norm_layers = nn.ModuleList(norm_layers)

        def forward(self, x0 : np.ndarray, x1 : np.ndarray):
            if self.fixed:
                emb_x0 = self.sbert.encode(sentences = x0, convert_to_tensor = True).to(self.device)
                emb_x1 = self.sbert.encode(sentences=x1, convert_to_tensor = True).to(self.device)
            else:
                emb_x0 = self.sbert(x0)["sentence_embedding"].to(self.device)
                emb_x1 = self.sbert(x1)["sentence_embedding"].to(self.device)
                
            emb_x0 = nn.functional.normalize(emb_x0)
            emb_x1 = nn.functional.normalize(emb_x1)
            abs_diff = torch.abs(emb_x0 - emb_x1)
            x = torch.cat([emb_x0, emb_x1, abs_diff], dim=1)
            for layer, norm in zip(self.layers[:-1], self.norm_layers):
                x = layer(x)
                x = norm(x)
                x = nn.functional.gelu(x)
                x = nn.functional.dropout(x, p = self.dropout, training = self.training)
            logits = self.layers[-1](x)
            return logits

    class CosSimilarity(nn.Module):
        def __init__(self, model : SentenceTransformer, fc_sizes : list, use_n_layers : int, device, fixed : bool, dropout : float = 0.1):
            super().__init__()
            self.sbert = model
            self.dropout = dropout
            self.device = device
            self.fixed = fixed
            
            OUTPUT_DIM = 2
            EMB_DIM = model.get_sentence_embedding_dimension()

            hidden_sizes = fc_sizes[:use_n_layers]
            layers = []
            layers.append(nn.Linear(3 * EMB_DIM, hidden_sizes[0]))
            norm_layers = [nn.LayerNorm(fc_sizes[0])]
            for i in range(use_n_layers - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                norm_layers.append(nn.LayerNorm(fc_sizes[i + 1]))
            layers.append(nn.Linear(hidden_sizes[-1], OUTPUT_DIM))
            self.layers = nn.ModuleList(layers)
            self.norm_layers = nn.ModuleList(norm_layers)
            self.cosine_similarity = nn.CosineSimilarity()
            
        def forward(self, x0 : np.ndarray, x1 : np.ndarray):
            if self.fixed:
                emb_x0 = self.sbert.encode(sentences = x0, convert_to_tensor = True).to(self.device)
                emb_x1 = self.sbert.encode(sentences = x1, convert_to_tensor = True).to(self.device)
            else:
                emb_x0=self.sbert(x0)["sentence_embedding"].to(self.device)
                emb_x1=self.sbert(x1)["sentence_embedding"].to(self.device)
            
            emb_x0 = nn.functional.normalize(emb_x0)
            emb_x1 = nn.functional.normalize(emb_x1)
            cos_sim = self.cosine_similarity(emb_x0, emb_x1).unsqueeze(1)
            abs_diff = torch.abs(emb_x0 - emb_x1)
            x = torch.cat([emb_x0, emb_x1, abs_diff], dim=1)
            for layer, norm in zip(self.layers[:-1], self.norm_layers):
                x = layer(x)
                x = norm(x)
                x = nn.functional.gelu(x)
                x = nn.functional.dropout(x, p = self.dropout, training = self.training)
            output = self.layers[-1](x)
            threshold = nn.functional.tanh(threshold)
            output = nn.functional.sigmoid(cos_sim - threshold)
            return (output, threshold, cos_sim)