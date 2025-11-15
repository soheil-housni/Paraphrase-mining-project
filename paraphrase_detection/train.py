import torch.nn
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy, Loss
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from loguru import logger
from torch.optim.lr_scheduler import _LRScheduler

class Train():
    def __init__(self,
                 model : torch.nn.Module,
                 optimizer : torch.optim,
                 scheduler : _LRScheduler,
                 criterion : torch.nn, 
                 device : torch.device,
                 n_freeze : int,
                 epochs : int,
                 train_dataloader : DataLoader,
                 val_dataloader : DataLoader
                 ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device 
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Freeze first n layer of model during training
        self.freeze_layers(n_freeze)

    def freeze_layers(self, n_freeze : int) -> None:
        for i in range(n_freeze):
            for param in self.model.sbert[0].auto_model.encoder.layer[i].parameters():
                param.requires_grad = True

    def run_training_loop(self) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        avg_batch_train_loss = np.zeros(self.epochs)
        avg_batch_val_loss = np.zeros(self.epochs)
        epoch_train_acc = np.zeros(self.epochs)
        epoch_val_acc = np.zeros(self.epochs)
        epoch_val_f1 = np.zeros(self.epochs)

        best_model_f1 = 0
        best_params = None

        for epoch in range(self.epochs):
            print(f'EPOCH: {epoch}')
            self.model.to(self.device)

            self.model.train()
            train_batch_loss = []
            all_train_preds = []
            all_train_labels = []
            for train_X_batch, train_y_batch in self.train_dataloader:
                train_X_batch = np.array(train_X_batch).T
                train_y_batch = train_y_batch.to(self.device)
                train_x0 = train_X_batch[:,0]
                train_x1 = train_X_batch[:,1]
                logits = self.model(train_x0, train_x1)
                loss = self.criterion(logits, train_y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_batch_loss.append(loss)
                all_train_preds.append(logits.argmax(dim=1).cpu())
                all_train_labels.append(train_y_batch.cpu())

            self.model.eval()
            val_batch_loss = []
            all_val_preds = []
            all_val_labels = []
            for val_X_batch, val_y_batch in self.val_dataloader:
                val_X_batch = np.array(val_X_batch).T
                val_y_batch = val_y_batch.to(self.device)
                with torch.no_grad():
                    val_x0 = val_X_batch[:,0]
                    val_x1 = val_X_batch[:,1]
                    logits = self.model(val_x0, val_x1)
                    loss = self.criterion(logits, val_y_batch)

                    val_batch_loss.append(loss)
                    all_val_preds.append(logits.argmax(dim=1).cpu())
                    all_val_labels.append(val_y_batch.cpu())
            
            all_train_preds = torch.cat(all_train_preds)
            all_train_labels = torch.cat(all_train_labels)
            all_val_preds = torch.cat(all_val_preds)
            all_val_labels = torch.cat(all_val_labels)
            
            avg_batch_train_loss[epoch] = np.mean([loss.cpu().item() for loss in train_batch_loss])
            avg_batch_val_loss[epoch] = np.mean([loss.cpu().item() for loss in val_batch_loss])

            epoch_train_acc[epoch] = accuracy_score(all_train_labels, all_train_preds)
            epoch_val_acc[epoch] = accuracy_score(all_val_labels, all_val_preds)
            epoch_val_f1[epoch] = f1_score(all_val_labels, all_val_preds, average = 'macro')

            if  epoch_val_f1[epoch] >= best_model_f1:
                best_model_f1 = epoch_val_f1[epoch]
                best_params = self.model.state_dict()

            logger.info(f'Epoch {epoch}: train loss = {avg_batch_train_loss[epoch]}')
            logger.info(f'Epoch {epoch}: validation loss = {avg_batch_val_loss[epoch]}')
            logger.info(f'Epoch {epoch}: train acc = {epoch_train_acc[epoch]}')
            logger.info(f'Epoch {epoch}: validation acc = {epoch_val_acc[epoch]}')
            logger.info(f'Epoch {epoch}: validation f1 = {epoch_val_f1[epoch]}')

        return best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1