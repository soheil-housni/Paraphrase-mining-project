import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer
import math

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
        
    
    class CrossAttention(nn.Module):
        def __init__(self,
                     model : SentenceTransformer,
                     fc_sizes : list,
                     use_n_layers : int,
                     device,
                     fixed : bool,
                     use_n_layers_cross_att : int,
                     fc_sizes_cross_att : list[int],
                     dropout : float = 0.1
                     ):
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
            self.cross_attention = CrossAttention(dmodel=self.sbert[0].auto_model.config.hidden_size, use_n_layers_cross_att = use_n_layers_cross_att, fc_sizes_cross_att = )
            
        def forward(self, x0 : np.ndarray, x1 : np.ndarray):
            transformed_x0 = self.sbert[0].auto_model(**x0)
            transformed_x1 = self.sbert[0].auto_model(**x1)

            cross_attention_x0 = self.cross_attention(transformed_x0[0],transformed_x1[0],x1["attention_mask"], dropout_p=self.dropout)
            cross_attention_x1 = self.cross_attention(transformed_x1[0],transformed_x0[0],x0["attention_mask"], dropout_p=self.dropout)

            pooled_x0=self.sbert[1]({"token_embeddings":cross_attention_x0, "attention_mask":x0["attention_mask"]})
            pooled_x1=self.sbert[1]({"token_embeddings":cross_attention_x1, "attention_mask":x1["attention_mask"]})

            normalized_x0 = self.sbert[2](pooled_x0)
            normalized_x1 = self.sbert[2](pooled_x1)

            emb_x0 = normalized_x0["sentence_embedding"].to(self.device)
            emb_x1 = normalized_x1["sentence_embedding"].to(self.device)

            emb_x0 = nn.functional.normalize(emb_x0)
            emb_x1 = nn.functional.normalize(emb_x1)

            abs_diff = torch.abs(emb_x0 - emb_x1)
            x = torch.cat([emb_x0, emb_x1, abs_diff], dim=1)
            for layer in range(len(self.layers)-1):
                x=self.layers[layer](x)
                x=self.norm_layers[layer](x)
                x = nn.functional.gelu(x)
                x = nn.functional.dropout(x, p = self.dropout, training=self.training)
            logits = self.layers[-1](x)
            return logits

class CrossAttention(nn.Module):
    def __init__(self, use_n_layers_cross_att : int, dmodel = 384, h = 8, fc_sizes_cross_att = [768]):
        super().__init__()
        self.dmodel=dmodel
        self.h=h
        self.dk=dmodel//self.h
        WQ_list=[]
        WK_list=[]
        WV_list=[]
        for h in range(self.h):
            WQ_list.append(nn.Linear(self.dmodel,self.dk, bias=False))
            WK_list.append(nn.Linear(self.dmodel,self.dk, bias=False))
            WV_list.append(nn.Linear(self.dmodel,self.dk, bias=False))

        self.W0 = nn.Linear(self.h*self.dk,dmodel, bias=False)

        self.WQ_list = nn.ModuleList(WQ_list)
        self.WK_list = nn.ModuleList(WK_list)
        self.WV_list = nn.ModuleList(WV_list)

        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)

        layers = []
        layers.append(nn.Linear(dmodel, fc_sizes_cross_att[0]))
        for i in range(use_n_layers_cross_att - 1):
            layers.append(nn.Linear(fc_sizes_cross_att[i], fc_sizes_cross_att[i + 1]))
        layers.append(nn.Linear(fc_sizes_cross_att[-1], dmodel))
        self.layers = nn.ModuleList(layers)

    def forward(self, A, B, attention_maskB, dropout_p=0.1) -> torch.Tensor:
        head_attention=[]
        for i in range(self.h):
            Q=self.WQ_list[i](A)
            K=self.WK_list[i](B)
            V=self.WV_list[i](B)
            scores=torch.bmm(Q,K.transpose(1,2))/math.sqrt(self.dk)
            scores.masked_fill_(attention_maskB.unsqueeze(1)==0,float("-inf"))
            attention_weights=nn.functional.softmax(scores,dim=-1)
            attention_weights=nn.functional.dropout(attention_weights,p=dropout_p, training=self.training)
            attention_score=torch.bmm(attention_weights,V)
            head_attention.append(attention_score)
        concat=torch.concat(head_attention, dim=-1)
        projection=self.W0(concat)
        projection=nn.functional.dropout(projection,p=dropout_p, training=self.training)
        output=self.norm1(A+projection)

        x=output
        for layer in range(len(self.layers)-1):
            x=self.layers[layer](x)
            x = nn.functional.gelu(x)
            x = nn.functional.dropout(x, p = dropout_p, training=self.training)
        x=self.layers[-1](x)
        output=self.norm2(output+x)

        return output