import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from datasets import load_dataset
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
import torch.nn as nn
from transformers import AutoModel
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from evaluate import load
from transformers import EarlyStoppingCallback
from transformers import get_linear_schedule_with_warmup,get_constant_schedule
from functools import partial
from transformers.modeling_outputs import SequenceClassifierOutput
from math import ceil
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import ConstantLR
from sklearn.model_selection import train_test_split

df = pd.read_parquet("hf://datasets/sentence-transformers/quora-duplicates/pair-class/train-00000-of-00001.parquet")

array=np.array(df)
array=array[:2000]




class CustomSbertModel(nn.Module):
    def __init__(self, pretrained_sbert_model_name, n_classes):
        super().__init__()
        self.n_classes=n_classes
        self.sbert=SentenceTransformer(pretrained_sbert_model_name)
        self.embedding_dimension=self.sbert.get_sentence_embedding_dimension()
        self.linear=nn.Linear(in_features=3*self.embedding_dimension, out_features=n_classes)
        
    def forward(self,sentence1, sentence2,batch_size):
        embedding1=self.sbert.encode(sentences=sentence1,batch_size=batch_size)
        embedding2=self.sbert.encode(sentences=sentence2,batch_size=batch_size)
        embedding1_torch=torch.tensor(embedding1)
        embedding2_torch=torch.tensor(embedding2)
        diff=torch.abs(embedding1_torch-embedding2_torch)
        characteristic_vector=torch.cat([embedding1_torch,embedding2_torch,diff],dim=1)
        logits=self.linear(characteristic_vector)
        return logits
        

def freezing(model,n_first_encoder_layers_to_freeze):
    if n_first_encoder_layers_to_freeze>0:
        for i in range(n_first_encoder_layers_to_freeze):
            for parameter in model.sbert[0].auto_model.encoder.layer[i].parameters():
                parameter.requires_grad=False
    return model


def dataset_preparation(dataset):
    sentences=array[:,:2]
    labels=array[:,-1]
    X_temp,X_test,y_temp,y_test=train_test_split(sentences,labels, test_size=0.15)
    X_train,X_val, y_train,y_val=train_test_split(X_temp,y_temp, test_size=0.12)
    y_train,y_val,y_test=y_train.astype(np.int64),y_val.astype(np.int64),y_test.astype(np.int64)
    return X_train,y_train,X_val,y_val,X_test,y_test


def training_modules(model,n_epochs,batch_size,X_train):
    optimizer=torch.optim.AdamW(model.parameters())
    n_batches=X_train.shape[0]//batch_size
    n_steps=n_epochs*n_batches
    scheduler=get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=n_steps*0.1, num_training_steps=n_steps)
    loss_fn=torch.nn.CrossEntropyLoss()
    return optimizer,scheduler,loss_fn

def training(model,X_train,y_train,X_val,y_val,n_epochs=5,batch_size=32):
    optimizer,scheduler,loss_fn=training_modules(model,n_epochs,batch_size,X_train)
    y_val=torch.tensor(y_val,dtype=torch.long)
    for epoch in range(n_epochs):
        batch_losses=[]
        model.train()
        for i in range(0,len(X_train),batch_size):
            batch_X_train=X_train[i:i+batch_size]
            batch_y_train=y_train[i:i+batch_size]
            batch_y_train=torch.tensor(batch_y_train, dtype=torch.long)
            logits=model(list(batch_X_train[:,0]),list(batch_X_train[:,1]),batch_size=len(batch_X_train))
            loss=loss_fn(logits,batch_y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss.item())
        epoch_loss=np.mean(batch_losses)  
        print(f"Epoch {epoch} :")
        print(f"The training loss is {epoch_loss}")
        
        with torch.no_grad():
            model.eval()
            logits_val=model(list(X_val[:,0]),list(X_val[:,1]),batch_size=len(X_val))
            val_loss=loss_fn(logits_val,y_val)
        
        print(f"The validation loss is {val_loss}")
        print("------------------------")
        

model=CustomSbertModel("all-MiniLM-L6-v2",2)
model=freezing(model,6)
X_train,y_train,X_val,y_val,X_test,y_test=dataset_preparation(array)
training(model,X_train,y_train,X_val,y_val,n_epochs=5,batch_size=32)         
        
        
        
        
        
    
                
            
    