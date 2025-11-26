import torch.nn
import torch
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy, Loss
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from loguru import logger
from torch.optim.lr_scheduler import _LRScheduler
from .modelarchitectures import PairClassifier

class Train():
    MIN_DELTA_EARLY_STOP = 0.002
    def __init__(self,
                 model : torch.nn.Module,
                 optimizer : torch.optim,
                 scheduler : _LRScheduler,
                 criterion : torch.nn, 
                 device : torch.device,
                 n_freeze : int,
                 epochs : int,
                 train_dataloader : DataLoader,
                 val_dataloader : DataLoader,
                 patience : int
                 ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device 
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.fixed = model.fixed
        self.patience = patience
        self.model.to(self.device)
        self.n_freeze = n_freeze

        # Determine from model if we're using CrossEntropy or CosSimilarity as a model architecture
        if isinstance(model, PairClassifier.CrossEntropy) | isinstance(model, PairClassifier.CrossAttention):
            self.is_cos_sim = False
        elif isinstance(model, PairClassifier.CosSimilarity):
            self.is_cos_sim = True
        else:
            raise TypeError(f'Unsupported model type {type(model)}')
        
        if not self.fixed:
            self.freeze_layers(self.n_freeze)

    def freeze_layers(self, n_freeze : int) -> None:
        for param in self.model.sbert[0].auto_model.embeddings.parameters():
            param.requires_grad = False

        for i in range(n_freeze):
            for param in self.model.sbert[0].auto_model.encoder.layer[i].parameters():
                param.requires_grad = False

    def run_training_loop(self) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        avg_batch_train_loss = np.zeros(self.epochs)
        avg_batch_val_loss = np.zeros(self.epochs)
        epoch_train_acc = np.zeros(self.epochs)
        epoch_val_acc = np.zeros(self.epochs)
        epoch_val_f1 = np.zeros(self.epochs)

        if self.is_cos_sim:
            cosines_train_set = []
            thresholds_train_set = []

        best_model_f1 = 0
        best_params = None
        best_val = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            print(f'EPOCH: {epoch}')

            self.model.train()
            train_batch_loss = []
            all_train_preds = []
            all_train_labels = []
            for train_X_batch, train_y_batch in self.train_dataloader:
                if self.fixed:
                    train_X_batch = np.array(train_X_batch).T
                    train_x0 = train_X_batch[:,0].tolist()
                    train_x1 = train_X_batch[:,1].tolist()
                    
                else:
                    train_x0 = train_X_batch[0]
                    train_x0 = {k : v.to(self.device) for k, v in train_x0.items()}
                    train_x1 = train_X_batch[1]
                    train_x1 = {k : v.to(self.device) for k, v in train_x1.items()}

                train_y_batch = train_y_batch.to(self.device)

                if self.is_cos_sim:
                    outputs, thresholds, cos_sims = self.model(train_x0, train_x1)
                    outputs = outputs.view(-1)
                else:
                    outputs = self.model(train_x0, train_x1)
                    
                loss = self.criterion(outputs, train_y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_batch_loss.append(loss)

                if self.is_cos_sim:
                    preds = (outputs>=0.5).long()
                    all_train_preds.append(preds.cpu())
                    all_train_labels.append(train_y_batch.cpu())
                else:
                    all_train_preds.append(outputs.argmax(dim=1).cpu())
                    all_train_labels.append(train_y_batch.cpu())

            self.model.eval()
            val_batch_loss = []
            all_val_preds = []
            all_val_labels = []
            for val_X_batch, val_y_batch in self.val_dataloader:
                
                with torch.no_grad():
                    if self.fixed:
                        val_X_batch = np.array(val_X_batch).T
                        val_x0 = val_X_batch[:,0].tolist()
                        val_x1 = val_X_batch[:,1].tolist()
                        
                    else:
                        val_x0 = val_X_batch[0]
                        val_x0 = {k : v.to(self.device) for k, v in val_x0.items()}
                        val_x1 = val_X_batch[1]
                        val_x1 = {k : v.to(self.device) for k, v in val_x1.items()}
                        
                    val_y_batch = val_y_batch.to(self.device)

                    if self.is_cos_sim:
                        outputs, thresholds, cos_sims = self.model(val_x0, val_x1)
                        outputs = outputs.view(-1)
                    else:
                        outputs = self.model(val_x0, val_x1)

                    loss = self.criterion(outputs, val_y_batch)
                    val_batch_loss.append(loss)

                    if self.is_cos_sim:
                        preds = (outputs>=0.5).long()
                        all_val_preds.append(preds.cpu())
                        all_val_labels.append(val_y_batch.cpu())
                    else:
                        all_val_preds.append(outputs.argmax(dim=1).cpu())
                        all_val_labels.append(val_y_batch.cpu())
                    
            
            all_train_preds = torch.cat(all_train_preds)
            all_train_labels = torch.cat(all_train_labels)
            all_val_preds = torch.cat(all_val_preds)
            all_val_labels = torch.cat(all_val_labels)
            
            avg_batch_train_loss[epoch] = np.mean([loss.cpu().item() for loss in train_batch_loss])
            avg_batch_val_loss[epoch] = np.mean([loss.cpu().item() for loss in val_batch_loss])

            epoch_train_acc[epoch] = accuracy_score(all_train_labels, all_train_preds.detach())
            epoch_val_acc[epoch] = accuracy_score(all_val_labels, all_val_preds.detach())
            epoch_val_f1[epoch] = f1_score(all_val_labels, all_val_preds, average = 'macro')

            if  epoch_val_f1[epoch] >= best_model_f1:
                best_model_f1 = epoch_val_f1[epoch]
                best_params = self.model.state_dict()

            logger.info(f'Epoch {epoch}: train loss = {avg_batch_train_loss[epoch]}')
            logger.info(f'Epoch {epoch}: validation loss = {avg_batch_val_loss[epoch]}')
            logger.info(f'Epoch {epoch}: train acc = {epoch_train_acc[epoch]}')
            logger.info(f'Epoch {epoch}: validation acc = {epoch_val_acc[epoch]}')
            logger.info(f'Epoch {epoch}: validation f1 = {epoch_val_f1[epoch]}')

            avg_val_loss = np.mean([loss.cpu().item() for loss in val_batch_loss])

            if avg_val_loss < best_val - Train.MIN_DELTA_EARLY_STOP:
                best_val = avg_val_loss
                patience_counter = 0
            else:
                patience_counter +=1
            
            if patience_counter >= self.patience:
                logger.warning(f'EARLY STOP during training at epoch {epoch}')
                slice_log = lambda x : x[: epoch + 1]
                avg_batch_train_loss = slice_log(avg_batch_train_loss)
                epoch_train_acc = slice_log(epoch_train_acc)
                avg_batch_val_loss = slice_log(avg_batch_val_loss)
                epoch_val_acc = slice_log(epoch_val_acc)
                epoch_val_f1 = slice_log(epoch_val_f1)
                break

        return best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1