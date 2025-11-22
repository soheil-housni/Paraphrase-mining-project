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
    col_values = [epochs.astype(np.int8), avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1]
    col_values = [x.reshape(-1, 1) for x in col_values]
    col_names = ['epoch','avg_batch_train_loss', 'epoch_train_acc', 'avg_batch_val_loss', 'epoch_val_acc', 'epoch_val_f1']
    df = pd.DataFrame(np.hstack(col_values), columns = col_names)
    best_f1 = epoch_val_f1.max()
    best = df[df['epoch_val_f1'] == best_f1]
    with open(f'log/{model_name}.txt', 'w') as log:
        log.write(df.to_string(index = False))
        log.write('\n \n')
        log.write('Model with highest F1 score: \n')
        log.write(best.to_string(index=False))
        log.write('\n')
    
def log_bo_results(model_name : str,
                   X_observed : np.ndarray,
                   Y_observed : np.ndarray,
                   col_names : list[str],
                   all_metrics : tuple,
                   params : None | dict = None,
                   ):
    avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1 = all_metrics
    if params:
        torch.save(params, f'models/{model_name}.pth')
    n_iterations = np.arange(1, X_observed.shape[0] + 1).T
    col_values = np.hstack((n_iterations, X_observed, Y_observed.T))
    df_hp = pd.DataFrame(col_values, columns = col_names.insert(0, 'n_iter'))

    best_f1 = Y_observed.max()
    best = df_hp[df_hp['epoch_val_f1'] == best_f1]

    with open(f'log/{model_name}_hp.txt', 'w') as log:
        log.write(df_hp.to_string(index = False))
        log.write('\n \n')
        log.write('Model with highest F1 score: \n')
        log.write(best.to_string(index = False))

    epochs = np.arange(1, len(avg_batch_train_loss) + 1)
    col_metrics = [epochs.astype(np.int8), avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1]
    col_metrics = [x.reshape(-1, 1) for x in col_metrics]
    col_names = ['epoch','avg_batch_train_loss', 'epoch_train_acc', 'avg_batch_val_loss', 'epoch_val_acc', 'epoch_val_f1']
    df = pd.DataFrame(np.hstack(col_metrics), columns = col_names)
    best_f1 = epoch_val_f1.max()
    best = df[df['epoch_val_f1'] == best_f1]
    with open(f'log/{model_name}_metrics.txt', 'w') as log:
        log.write(df.to_string(index = False))
        log.write('\n \n')
        log.write('Model with highest F1 score: \n')
        log.write(best.to_string(index=False))
        log.write('\n')