# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 17:27:11 2025

@author: sh032
"""

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
from ignite.handlers.early_stopping import EarlyStopping
from ignite.engine.engine import Engine
from ignite.metrics import Accuracy,Loss
from ignite.engine.events import Events
from ignite.handlers.early_stopping import EarlyStopping



df = pd.read_parquet("hf://datasets/sentence-transformers/quora-duplicates/pair-class/train-00000-of-00001.parquet")

array=np.array(df)
array=array[:5000]

class CustomSbertModel(nn.Module):
    def __init__(self, pretrained_sbert_model_name, n_classes):
        super().__init__()
        self.n_classes=n_classes
        self.sbert=SentenceTransformer(pretrained_sbert_model_name)
        self.embedding_dimension=self.sbert.get_sentence_embedding_dimension()
        self.linear1=nn.Linear(in_features=3*self.embedding_dimension, out_features=3*self.embedding_dimension//2)
        self.last_linear=nn.Linear(in_features=3*self.embedding_dimension//2, out_features=n_classes)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.1)
    def forward(self,sentence1, sentence2,batch_size):
        embedding1=self.sbert.encode(sentences=sentence1,batch_size=batch_size)
        embedding2=self.sbert.encode(sentences=sentence2,batch_size=batch_size)
        embedding1_torch=torch.tensor(embedding1)
        embedding2_torch=torch.tensor(embedding2)
        diff=torch.abs(embedding1_torch-embedding2_torch)
        characteristic_vector=torch.cat([embedding1_torch,embedding2_torch,diff],dim=1)
        logits=self.linear1(characteristic_vector)
        logits=self.relu(logits)
        logits=self.dropout(logits)
        logits=self.last_linear(logits)
        return logits
        

def freezing(model, n_first_encoder_layers_to_freeze):
    if n_first_encoder_layers_to_freeze > 0:
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


def iterable_batches(X_set,y_set,batch_size):
    iterable=[]
    for i in range(0,len(X_set),batch_size):
        upper_indexation=min(len(X_set)-1,i+batch_size)
        batch_X=X_set[i:upper_indexation]
        batch_y=y_set[i:upper_indexation]
        batch=(batch_X,batch_y)
        iterable.append(batch)
    return iterable
        


def training_modules(model,n_epochs,batch_size,X_train):
    optimizer=torch.optim.AdamW(model.parameters())
    n_batches=X_train.shape[0]//batch_size
    n_steps=n_epochs*n_batches
    scheduler=get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=n_steps*0.1, num_training_steps=n_steps)
    loss_fn=torch.nn.CrossEntropyLoss()
    return optimizer,scheduler,loss_fn



def lauch_training(X_train,y_train,X_val,y_val,X_test,y_test,pretrained_sbert_model_name,n_first_encoder_layers_to_freeze,num_classes=2,batch_size=32,n_epochs=5):
    train_set=iterable_batches(X_train,y_train,batch_size=batch_size)
    val_set=iterable_batches(X_val,y_val,batch_size=len(X_val))
    
    n_epochs=n_epochs
    batch_size=batch_size
    model=CustomSbertModel(pretrained_sbert_model_name,num_classes)
    model = freezing(model,n_first_encoder_layers_to_freeze)
    optimizer,scheduler,loss_fn=training_modules(model,n_epochs,batch_size,X_train)

    
    accuracy=Accuracy()
    loss=Loss(loss_fn)
    
    
    def train_step(engine,batch):
        model.train()
        X,y=batch
        y=torch.tensor(y)
        logits=model(list(X[:,0]),list(X[:,1]),batch_size=len(X))
        loss=loss_fn(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return logits,y
    
    trainer=Engine(train_step)
    accuracy.attach(trainer,"training_accuracy")
    loss.attach(trainer,"training_loss")
    
    
    def val_step(engine,batch):
        X,y=batch
        y=torch.tensor(y)
        model.eval()
        with torch.no_grad():
            logits=model(list(X[:,0]),list(X[:,1]),batch_size=len(X))
            loss=loss_fn(logits,y)
        return logits,y
    
    def score_function(engine):
        val_loss=engine.state.metrics['validation_loss']
        return -val_loss
    
    evaluator=Engine(val_step)
    early_stopping=EarlyStopping(patience=3, score_function=score_function, trainer=trainer, min_delta=0.01)
    evaluator.add_event_handler(Events.COMPLETED,early_stopping)
    accuracy.attach(evaluator, "validation_accuracy")
    loss.attach(evaluator,"validation_loss")
    
    
    def run_validation(engine):
        evaluator.run(val_set,max_epochs=1)
        training_metrics=engine.state.metrics
        val_metrics=evaluator.state.metrics
        print(f"Epoch number : {engine.state.epoch} :")
        print(f"Training loss : {training_metrics['training_loss']}")
        print(f"Training accuracy : {training_metrics['training_accuracy']}")
        print(f"Validation loss : {val_metrics['validation_loss']}")
        print(f"Validation accuracy : {val_metrics['validation_accuracy']}")
        print("----------------------------------------------------------------------")
    
    def message(engine):
        epoch=engine.state.epoch
        print(f"Training stops at epoch {epoch}")
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED,run_validation)
    trainer.add_event_handler(Events.COMPLETED,message)
    trainer.run(train_set,max_epochs=n_epochs)
        
        


X_train,y_train,X_val,y_val,X_test,y_test=dataset_preparation(array)
pretrained_sbert_model_name="all-MiniLM-L6-v2"
n_first_encoder_layers_to_freeze=3
n_epochs=10
lauch_training(X_train,y_train,X_val,y_val,X_test,y_test,pretrained_sbert_model_name,n_first_encoder_layers_to_freeze,n_epochs=10)

        
        