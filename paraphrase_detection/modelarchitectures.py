import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer



class SBERTPairClassifier(nn.Module):
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

        for i in range(use_n_layers - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(nn.Linear(hidden_sizes[-1], OUTPUT_DIM))
        
    def forward(self, x0 : np.ndarray, x1 : np.ndarray):
        if self.fixed:
            emb_x0 = self.sbert(x0)["sentence_embedding"]
            emb_x1 = self.sbert(x1)["sentence_embedding"]
            abs_diff = torch.abs(emb_x0 - emb_x1)
            x = torch.cat([emb_x0, emb_x1, abs_diff], dim=1)
            for layer in self.layers[:-1]:
                x = nn.functional.gelu(layer(x))
                x = nn.functional.dropout(x, p = self.dropout, training=self.training)
            logits = self.layers[-1](x)
            return logits
        else:
            emb_x0=self.sbert.encode(sentences = x0,convert_to_tensor=True).to(self.device)
            emb_x1=self.sbert.encode(sentences=x1,convert_to_tensor=True).to(self.device)
            abs_diff = torch.abs(emb_x0 - emb_x1)
            x = torch.cat([emb_x0, emb_x1, abs_diff], dim=1)
            for layer in self.layers[:-1]:
                x = nn.functional.gelu(layer(x))
                x = nn.functional.dropout(x, p = self.dropout, training=self.training)
            logits = self.layers[-1](x)
            return logits


# def __init__(self, model : SentenceTransformer, fc_sizes : list, use_n_layers : int, device, fixed : bool, dropout : float = 0.1):
#             super().__init__()
#             self.sbert = model
#             self.dropout = dropout
#             self.device = device
#             self.fixed = fixed
#             OUPUT_DIM = 2
#             EMB_DIM = model.get_sentence_embedding_dimension()
#             layers = [nn.Linear(3 * EMB_DIM, fc_sizes[0])]
#             for layer in range(len(fc_sizes)):
#                 if layer == len(fc_sizes) - 1:
#                     layers.append(nn.Linear(fc_sizes[layer], OUPUT_DIM))
#                 else:
#                     layers.append(nn.Linear(fc_sizes[layer], fc_sizes[layer + 1]))
#             self.layers = nn.ModuleList(layers)
    

# class fixed_sbert_SBERTPairClassifier(nn.Module):
#     def __init__(self, model : SentenceTransformer, fc_sizes : list, device, dropout : float = 0.1):
#         super().__init__()
#         self.sbert = model
#         self.dropout = dropout
#         self.device = device
#         OUPUT_DIM = 2
#         EMB_DIM = model.get_sentence_embedding_dimension()
#         layers = [nn.Linear(3 * EMB_DIM, fc_sizes[0])]
#         for layer in range(len(fc_sizes)):
#             if layer == len(fc_sizes) - 1:
#                 layers.append(nn.Linear(fc_sizes[layer], OUPUT_DIM))
#             else:
#                 layers.append(nn.Linear(fc_sizes[layer], fc_sizes[layer + 1]))
#         self.layers = nn.ModuleList(layers)
        
#     def forward(self, x0 : np.ndarray, x1 : np.ndarray):
#         emb_x0=self.sbert.encode(sentences = x0,convert_to_tensor=True).to(self.device)
#         emb_x1=self.sbert.encode(sentences=x1,convert_to_tensor=True).to(self.device)
#         abs_diff = torch.abs(emb_x0 - emb_x1)
#         x = torch.cat([emb_x0, emb_x1, abs_diff], dim=1)
#         for layer in self.layers[:-1]:
#             x = nn.functional.gelu(layer(x))
#             x = nn.functional.dropout(x, p = self.dropout, training=self.training)
#         logits = self.layers[-1](x)
#         return logits
    
    
# def SBERTPairClassifier_model_selection(model : SentenceTransformer, fc_sizes : list, device, dropout : float = 0.1, trainable:bool=False):
#     if trainable:
#         return trainable_sbert_SBERTPairClassifier(model,fc_sizes,device,dropout)
#     else:
#         return fixed_sbert_SBERTPairClassifier(model,fc_sizes,device,dropout)


    
    


    