# Relative imports
from .modelarchitectures import PairClassifier
from .logger import log_bo_results
from enums import Optimizer, Scheduler
from .train import Train
from .dataloader import TextPairDataset

# Libraries
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import torch
from loguru import logger
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LRScheduler
from dataclasses import dataclass, fields
from typing import Any

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

class BOSearchTrain():
    """
    Bayesian Optimization hyperparameter tuning with expected improvement as acquisition function and gaussian process 
    as surrogate model. The criterion is the validation F1 score.
    """    
    NUM_RESTARTS = 5
    RAW_SAMPLES = 50
    WARMUP_STEPS_RATIO = 0.1
    EARLY_STOP_THRESHOLD = 0.002
    EARLY_STOP_WINDOW = 5

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
                 criterion : torch.nn.Module,
                 device : torch.device,
                 n_init_samples : int,
                 n_iterations : int,
                 fixed : bool,
                 cos_similarity : bool,
                 epochs : int,
                 patience : int
                 ) -> None:
        """
        Inializes the arguements required for the BO loop.

        Args:
            bounds (torch.Tensor): Bounds of the hyperparameters search space
            names (list): Name of the hyperparameter to be tuned in BO
            sbert_model (SentenceTransformer): SBERT model used from SentenceTransformer
            X_train (np.ndarray): Design matrix of training data
            y_train (np.ndarray): Label array of training data
            X_val (np.ndarray): Design matrix of validation data
            y_val (np.ndarray): Label array of training data
            optimizer (Optimizer): Optimizer for training
            scheduler (Scheduler): Learning rate scheduler
            criterion (torch.nn): Criterion for loss calculations
            device (torch.device): Device used during training
            n_init_samples (int): Number of samples to initialize the surrogate model
            n_iterations (int): Maximal number of iterations of BO
            fixed (bool): Determines if the SBERT layers are to be kept fixed (fixed = True) or trainable (fixed = False)
            cos_similarity (bool): Determinines if it should inialize the SBERTPairClassifier with cosine similarity or 
            the cross entropy classifier
            epochs (int): Number of epochs to run during training
        """        
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
        self.patience = patience

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Helper to get the index of the hyperparameter based on the name
        self.hp_index = {name: i for i, name in enumerate(names)}
    
    @staticmethod
    def normalize_X(X : torch.Tensor, bounds : torch.Tensor) -> torch.Tensor:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        return (X - lower) / (upper - lower)

    @staticmethod
    def unnormalize_X(X_norm : torch.Tensor, bounds : torch.Tensor) -> torch.Tensor:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        return lower + X_norm * (upper - lower)
    
    def tensor_to_hparams(self, hp : torch.Tensor) -> HyperParameters:
        """
        Transforms the tensor hyperparameters that BoTorch outputs into arguments of the HyperParameters class.
        Helps obtain the value of the candidate hyperparameters.

        Args:
            hp (torch.Tensor): Candidate hyperparameters

        Returns:
            HyperParameters: Dataclass containing values of hyperparameters
        """        
        hyperparams = HyperParameters(
            lr = float(hp[self.hp_index["lr"]]),
            weight_decay = float(hp[self.hp_index["weight_decay"]]),
            dropout = float(hp[self.hp_index["dropout_p"]]),
            n_freeze = int(hp[self.hp_index["n_freeze"]]),
            batch_size = int(hp[self.hp_index["batch_size"]]),
            use_n_layers = int(hp[self.hp_index["use_n_layers"]]),
            fc1 = int(hp[self.hp_index["fc1"]]),
            fc2 = int(hp[self.hp_index["fc2"]]),
            fc3 = int(hp[self.hp_index["fc3"]])
        )
        return hyperparams
    
    def init_hp_samples(self) -> None:
        """
        Initializes self.n_init_samples set of uniformly sampled hyperparameters
        """        
        self.X_observed = self.bounds[0, :] + (self.bounds[1, :] - self.bounds[0, :]) * torch.rand(self.n_init_samples, self.bounds.shape[1])

    def init_fit_samples(self) -> tuple[float, tuple]:
        """
        Fits the set of initial samples that were drawn from the self.init_hp_samples to obtain an initial set of Y.
        """
        logger.info(f'Initial fit of GP for {self.n_init_samples} samples of hyperparameter sets \n')

        Y_observed = torch.zeros((self.n_init_samples, 1))
        self.init_hp_samples()
        hp_init = self.X_observed
        best_f1 = 0
        best_results = None
        for n in range(self.n_init_samples):
            hp_n = hp_init[n, :]
            hp = self.tensor_to_hparams(hp_n)
            logger.info(f"Starting iteration {n}:\n"
                        f"{[(f.name, getattr(hp, f.name)) for f in fields(hp)]}")

            results = self.train_for_hp_set(hp)
            if self.cos_similarity:
                best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1, cosines_train_set, thresholds_train_set = results
            else:
                best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1 = results
            Y_observed[n] = np.max(epoch_val_f1)
            
            logger.info(f'Maximum validation F1 score: {np.max(epoch_val_f1)} \n')

            if np.max(epoch_val_f1) >= best_f1:
                best_f1 = np.max(epoch_val_f1)
                best_results = results

        logger.success(f'Finished initial fit of samples')
        self.Y_observed = Y_observed
        return best_f1, best_results
    
    def fit_GP(self) -> SingleTaskGP:
        """
        Fits a Gaussian Process using the Exact Marginal LL

        Returns:
            SingleTaskGP: The GP that was fit on collected data so far
        """

        assert self.X_observed.dim() == 2, 'X_observed has to be of dim = 2 for GP fit'
        assert self.Y_observed.dim() == 2, 'Y_observed has to be of dim = 2 for GP fit'

        X_norm = BOSearchTrain.normalize_X(self.X_observed, self.bounds.T)
        
        model = SingleTaskGP(X_norm.double(), self.Y_observed.double())
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def sample_set(self, model : SingleTaskGP, n_candidates : int = 1) -> torch.Tensor:
        """
        Samples a candidate hyperparameter set based on the best Expected Improvement.

        Args:
            model (SingleTaskGP): Gaussian Process fit so far
            n_candidates (int, optional): Number of candidate sets. Defaults to 1.

        Returns:
            torch.Tensor: Candidate set of hyperparameters
        """        
        best_f = self.Y_observed.max()
        acq_fct = qLogExpectedImprovement(model = model, best_f = best_f)

        d = self.X_observed.shape[1]

        acq_bounds = torch.stack([
            torch.zeros(d),
            torch.ones(d)
        ]).double()

        candidates_norm, _ = optimize_acqf(acq_function = acq_fct,
                                      bounds = acq_bounds,
                                      q = n_candidates,
                                      num_restarts = BOSearchTrain.NUM_RESTARTS,
                                      raw_samples = BOSearchTrain.RAW_SAMPLES,
                                      return_best_only = True
                                      )
        candidates = BOSearchTrain.unnormalize_X(candidates_norm, self.bounds.T)
        return candidates
    
    def train_for_hp_set(self, hp : HyperParameters) -> tuple[Any, ...]:
        """
        Trains a model based on a candidate set of hyperparameters.

        Args:
            hyperparams (torch.Tensor): Candidate hyperparameters
        Returns:
            tuple[Any, ...]: Outputs the best params across epochs as well as all metrics that Train class outputs. It is of variable
            length depending if we are using the cosine similarity trainer or not
        """        

        fc_sizes = [hp.fc1, hp.fc2, hp.fc3]
        model, optimizer, scheduler, train_loader, val_loader = self.train_config(hp.lr, hp.weight_decay, fc_sizes, hp.use_n_layers, hp.dropout, hp.batch_size)

        assert isinstance(hp.n_freeze, int), 'n_freeze must be int'

        try:
            if self.cos_similarity:
                trainer = Train(
                    model,
                    optimizer,
                    scheduler,
                    self.criterion,
                    self.device,
                    hp.n_freeze,
                    self.epochs,
                    train_loader,
                    val_loader,
                    self.patience
                )
                results = trainer.run_training_loop()

            else:
                trainer = Train(
                    model,
                    optimizer,
                    scheduler,
                    self.criterion,
                    self.device,
                    hp.n_freeze,
                    self.epochs,
                    train_loader,
                    val_loader,
                    self.patience
                )
                results = trainer.run_training_loop()
        finally:
            del trainer, model, optimizer, scheduler

            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return results
    
    def train_config(self,
                     lr : float,
                     weight_decay : float,
                     fc_sizes : list[int],
                     use_n_layers : int,
                     dropout : float,
                     batch_size : int
                     ) -> tuple[PairClassifier, torch.optim.Optimizer, LRScheduler, DataLoader, DataLoader]:
        """
        Inializes the training configurations that are dependent on hyperparameters.

        Args:
            lr (float): Learning rate.
            weight_decay (float): Weight decay in optimizer.
            fc_sizes (list[int]): Contains the number of fc layers (length of fc_sizes) and the number of neurons per hidden
            layer.
            use_n_layers (int): Number of the first n layers to use.
            dropout (float): Neuron dropout rate for the model definitions
            batch_size (int): Batch size to intialize the data loaders of training and validation sets

        Returns:
            tuple[PairClassifier, Optimizer, Scheduler, DataLoader, DataLoader]: Outputs the classifier, optimizer, learning rate scheduler,
            training data loader and validation data loader.
        """        
        steps = self.epochs * self.X_train.shape[0] // batch_size
        n_warmup_steps = int(steps * BOSearchTrain.WARMUP_STEPS_RATIO)

        assert isinstance(fc_sizes, list), 'fc_sizes must be list'
        assert isinstance(use_n_layers, int), 'use_n_layers must be int'
        assert isinstance(dropout, float), 'dropout must be float'
        assert isinstance(dropout, float), 'dropout must be float'
        assert isinstance(lr, float), 'dropout must be float'
        assert isinstance(weight_decay, float), 'weight_decay must be float'
        assert isinstance(batch_size, int), 'batch_size must be int'

        if self.cos_similarity:
            model = PairClassifier.CosSimilarity(self.sbert_model, fc_sizes, use_n_layers, self.device, self.fixed, dropout)
        else:
            model = PairClassifier.CrossEntropy(self.sbert_model, fc_sizes, use_n_layers, self.device, self.fixed, dropout)

        match self.optimizer:
            case Optimizer.ADAMW:
                optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
            case _:
                raise(TypeError(f'{self.optimizer} is not a valid Optimizer'))

        match self.scheduler:
            case Scheduler.LINEAR:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = n_warmup_steps, num_training_steps = steps)
            case _:
                raise(TypeError(f'{self.scheduler} is not a valid Scheduler'))
        if self.fixed:
            train_loader = DataLoader(dataset = TextPairDataset(self.X_train, self.y_train), batch_size = batch_size, shuffle = True, num_workers = 4)
            val_loader = DataLoader(dataset = TextPairDataset(self.X_val, self.y_val), batch_size = batch_size, shuffle = True, num_workers = 4)
        else:
            train_loader = DataLoader(dataset = TextPairDataset(self.X_train, self.y_train, tokenization = True, tokenizer = self.tokenizer), batch_size = batch_size, shuffle = True, num_workers = 4)
            val_loader = DataLoader(dataset = TextPairDataset(self.X_val, self.y_val, tokenization = True, tokenizer = self.tokenizer), batch_size = batch_size, shuffle = True, num_workers = 4)
        return model, optimizer, scheduler, train_loader, val_loader

    def search_loop(self) -> None:
        """
        Computes the hyperparameter search iteratively.
        """        
        # Sample an initial set of X observations, evaluate it and fit the GP
        best_f1, best_results = self.init_fit_samples()        
        gp_model = self.fit_GP()

        best_vals = []

        logger.success(f"Starting BO hyperparameter search \n")
        
        for n in range(self.n_iterations):
            # Obtain candidate hyperparameter set by sampling from GP
            hp_set = self.sample_set(gp_model)
            hp_set = hp_set.squeeze(0)
            
            hp = self.tensor_to_hparams(hp_set)

            logger.info(f"Starting iteration {n}:\n"
                        f"{[(f.name, getattr(hp, f.name)) for f in fields(hp)]}")
            
            # Train and evaluate the hyperparameter set
            results = self.train_for_hp_set(hp)
            if self.cos_similarity:
                params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1, cosines_train_set, thresholds_train_set = results
            else:
                params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1 = results
            
            # Update current knowledge of the function
            y = torch.tensor([[np.max(epoch_val_f1)]])
            self.X_observed = torch.vstack((self.X_observed, hp_set.unsqueeze(0)))
            self.Y_observed = torch.vstack((self.Y_observed, y))

            logger.info(f'\n Maximum validation F1 score: {np.max(epoch_val_f1)} \n')
            
            # Fit Surrogate Model to newly added data
            gp_model = self.fit_GP()

            best_vals.append(np.max(epoch_val_f1))

            # Save metrics of best performing model
            if np.max(epoch_val_f1) >= best_f1:
                best_f1 = np.max(epoch_val_f1)
                best_results = results
            
            if len(best_vals) >= BOSearchTrain.EARLY_STOP_WINDOW:
                window = best_vals[-BOSearchTrain.EARLY_STOP_WINDOW:]
                if max(window) - min(window) < BOSearchTrain.EARLY_STOP_THRESHOLD:
                    logger.warning(f'EARLY STOP in Bayesian Optimization loop after {n} loop')
                    break
        return best_results
          