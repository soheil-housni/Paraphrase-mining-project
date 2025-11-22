from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataloader import TextPairDataset
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import torch
from loguru import logger
from modelarchitectures import PairClassifier
from transformers import get_linear_schedule_with_warmup
from enums import Optimizer, Scheduler
from train import Train
from dataclasses import dataclass
from logger import log_bo_results

@dataclass
class HyperParameters:
    lr: float
    weight_decay: float
    dropout: float
    batch_size: int
    n_freeze : int
    use_n_layers: int
    fc1: int
    fc2: int
    fc3: int
    fc4: int

class BOSearchTrain():

    NUM_RESTARTS = 5
    RAW_SAMPLES = 50
    WARMUP_STEPS_RATIO = 0.1

    def __init__(self,
                 bounds : torch.Tensor,
                 names : list,
                 sbert_model : SentenceTransformer,
                 X_train : np.ndarray,
                 y_train : np.ndarray,
                 X_val : np.ndarray,
                 y_val : np.ndarray,
                 optimizer : Optimizer,
                 scheduler : Scheduler,
                 criterion : torch.nn,
                 device : torch.device,
                 n_init_samples : int,
                 n_iterations : int,
                 fixed : bool,
                 cos_similarity : bool,
                 epochs : int
                 ) -> None:
        
        self.bounds = bounds
        self.names = names
        self.n_init_samples = n_init_samples
        self.sbert_model = sbert_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_iterations = n_iterations
        self.criterion = criterion
        self.device = device
        self.fixed = fixed
        self.cos_similarity = cos_similarity
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.X_observed = None
        self.Y_observed = None

        self.hp_index = {name: i for i, name in enumerate(names)}

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tensor_to_hparams(self, hp : torch.Tensor) -> HyperParameters:
        hyperparams = HyperParameters(
            lr = float(hp[self.hp_index["lr"]]),
            weight_decay = float(hp[self.hp_index["weight_decay"]]),
            dropout = float(hp[self.hp_index["dropout"]]),
            n_freeze = int(hp[self.hp_index["n_freeze"]]),
            batch_size = int(hp[self.hp_index["batch_size"]]),
            use_n_layers = int(hp[self.hp_index["use_n_layers"]]),
            fc1 = int(hp[self.hp_index["fc1"]]),
            fc2 = int(hp[self.hp_index["fc2"]]),
            fc3 = int(hp[self.hp_index["fc3"]]),
            fc4 = int(hp[self.hp_index["fc4"]]),
        )
        return hyperparams
    

    def init_hp_samples(self):
        return self.bounds[0, :] + (self.bounds[1, :] - self.bounds[0, :]) * torch.rand(self.n_init_samples, self.bounds.shape[1])

    def init_fit_samples(self, init_X : torch.Tensor):
        logger.info(f'Initial fit of GP for {self.n_init_samples} samples of hyperparameter sets \n')
        self.X_observed = init_X
        Y_observed = torch.zeros(self.n_init_samples)
        hp = self.init_hp_samples()
        for n in range(self.n_init_samples):
            logger.trace(f'Starting INIT iteration {n}:\n" f"{[(name, hp[n, idx]) for idx, name in enumerate(self.names)]}')
            hp_n = hp[n, :]
            results = self.train_for_hp_set(hp_n, True)
            if self.cos_similarity:
                best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1 = results
            else:
                best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1, cosines_train_set, thresholds_train_set = results
            Y_observed[n] = np.max(epoch_val_f1)
            logger.trace(f'Maximum validation F1 score: {np.max(epoch_val_f1)}')
        logger.success(f'Finished initial fit of samples')
        self.Y_observed = Y_observed 
    
    def fit_GP(self) -> SingleTaskGP:
        model = SingleTaskGP(self.X_observed.double(), self.Y_observed.double())
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def train_for_hp_set(self, hyperparams : torch.Tensor):
        hp = self.tensor_to_hparams(hyperparams)

        fc_sizes = [hp.fc1, hp.fc2, hp.fc3, hp.fc4]
        model, optimizer, scheduler, train_loader, val_loader = self.train_config(hp.lr, hp.weight_decay, fc_sizes, hp.use_n_layers, hp.dropout, hp.batch_size)
        if self.cos_similarity:
            trainer = Train.CosSimClassifier(
                model,
                optimizer,
                scheduler,
                self.criterion,
                self.device,
                hp.n_freeze,
                self.epochs,
                train_loader,
                val_loader,
                self.sbert_model
            )
            results = trainer.run_training_loop()

        else:
            trainer = Train.Classifier(
                model,
                optimizer,
                scheduler,
                self.criterion,
                self.device,
                hp.n_freeze,
                self.epochs,
                train_loader,
                val_loader,
                self.sbert_model
            )
            results = trainer.run_training_loop()

        return results
        
    def sample_set(self, model : SingleTaskGP, n_candidates = 1) -> torch.Tensor:
        best_f = self.Y_observed.min()
        acq_fct = qExpectedImprovement(model = model, best_f = best_f)

        acq_bounds = self.bounds.T.double()

        canditates, _ = optimize_acqf(acq_function = acq_fct,
                                   bounds = acq_bounds,
                                   q = n_candidates,
                                   num_restarts = BOSearchTrain.NUM_RESTARTS,
                                   raw_samples = BOSearchTrain.RAW_SAMPLES
                                   )
        return canditates
    
    def train_config(self,
                     lr : float,
                     weight_decay : float,
                     fc_sizes : int,
                     use_n_layers : int,
                     dropout : float,
                     batch_size : int
                     ) -> tuple[PairClassifier, Optimizer, Scheduler, DataLoader, DataLoader]:
        steps = self.epochs * self.X_train.shape[0] // batch_size
        n_warmup_steps = steps * BOSearchTrain.WARMUP_STEPS_RATIO

        if self.cos_similarity:
            model = PairClassifier.CosSimilarity(self.sbert_model, fc_sizes, use_n_layers, self.device, self.fixed, dropout)
        else:
            model = PairClassifier.CrossEntropy(self.sbert_model, fc_sizes, use_n_layers, self.device, self.fixed, dropout)

        match self.optimizer:
            case Optimizer.ADAMW:
                optimizer = torch.optim.AdamW(lr = lr, weight_decay = weight_decay)
            case _:
                raise(TypeError(f'{self.optimizer} is not a valid Optimizer'))

        match self.scheduler:
            case Scheduler.LINEAR:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = n_warmup_steps, num_training_steps = steps)
            case _:
                raise(TypeError(f'{self.scheduler} is not a valid Scheduler'))

        train_loader = DataLoader(dataset = TextPairDataset(self.X_train, self.y_train, tokenization = True, tokenizer = self.tokenizer), batch_size = batch_size, shuffle = True, num_workers = 4)
        val_loader = DataLoader(dataset = TextPairDataset(self.X_val, self.y_val, tokenization = True, tokenizer = self.tokenizer), batch_size = batch_size, shuffle = True, num_workers = 4)
        return model, optimizer, scheduler, train_loader, val_loader

    def search_loop(self):
        self.init_fit_samples()
        logger.info(f"Starting BO hyperparameter search \n")
        gp_model = self.fit_GP()
        best_params = None
        best_f1 = 0
        best_results = None
        for n in range(self.n_iterations):
            hp_set = self.sample_set(gp_model)
            logger.info(f"Starting iteration {n}:\n" f"{[(name, hp_set[idx]) for idx, name in enumerate(self.names)]}")
            results = self.train_for_hp_set()

            if self.cos_similarity:
                params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1 = results
            else:
                params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1, cosines_train_set, thresholds_train_set = results
            
            y = torch.tensor(np.max(epoch_val_f1))
            self.X_observed = torch.vstack((self.X_observed, hp_set))
            self.Y_observed = torch.hstack((self.Y_observed, y))

            logger.trace(f'\n Maximum validation F1 score: {np.max(epoch_val_f1)} \n')
            
            if np.max(epoch_val_f1) >= best_f1:
                best_results = results
        
        log_bo_results(model_name = f'BO_cos{self.cos_similarity}',
                       X_observed = self.X_observed.numpy(),
                       Y_observed = self.Y_observed.numpy(),
                       col_names = self.names,
                       all_metrics = results[1:6],
                       params = best_results[0]
                       )


        
        

                