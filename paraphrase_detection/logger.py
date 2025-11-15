import torch
import numpy as np
import pandas as pd

def log_metrics_and_model(model_name : str,
                          avg_batch_train_loss : np.ndarray,
                          epoch_train_acc : np.ndarray,
                          avg_batch_val_loss : np.ndarray,
                          epoch_val_acc : np.ndarray,
                          epoch_val_f1 : np.ndarray,
                          params : None | dict = None,
                          ):
    if params:
        torch.save(params, f'models/{model_name}.pth')
    epochs = np.arange(1, len(avg_batch_train_loss) + 1)
    col_values = [epochs, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1]
    col_values = [x.reshape(-1, 1) for x in col_values]
    col_names = ['epoch','avg_batch_train_loss', 'epoch_train_acc', 'avg_batch_val_loss', 'epoch_val_acc', 'epoch_val_f1']
    df = pd.DataFrame(np.hstack(col_values), columns = col_names)
    df.to_csv(f'log/{model_name}.txt', sep = '\t', index = False)    