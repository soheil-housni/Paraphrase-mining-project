# Relative imports
from .modelarchitectures import PairClassifier
from .logger import log_bo_results
from enums import Optimizer, Scheduler, Model
from .train import Train
from .dataloader import TextPairDataset
from .hyperparametersets import HP

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


class BOSearchTrain():
    """
    Bayesian Optimization hyperparameter tuning with expected improvement as acquisition function and gaussian process 
    as surrogate model. The criterion is the validation F1 score.
    """    
    NUM_RESTARTS = 5
    RAW_SAMPLES = 50
    WARMUP_STEPS_RATIO = 0.1
    EARLY_STOP_THRESHOLD = 0.01
    EARLY_STOP_WINDOW = 6

    def __init__(self,
                 bounds : torch.Tensor,
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
                 epochs : int,
                 patience : int,
                 model_arch : Model
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
            the cross entropy classifier
            epochs (int): Number of epochs to run during training
        """        
        self.bounds = bounds
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
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.model_arch = model_arch

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        match model_arch:
            case Model.CROSSENTROPY:
                self.hp_dataclass = HP.CrossEntropyHP
            case Model.CROSSATTENTION:
                self.hp_dataclass = HP.CrossAttentionHP
            case Model.COSINETHRESHOLD:
                self.hp_dataclass = HP.CosineSimilarityHP
            
        if len(fields(self.hp_dataclass)) != bounds.shape[1]:
                    raise ValueError(f'Number of bounds is different from length of fields in hyperparameter set data class')

        hp_fields = fields(self.hp_dataclass)

        # Helper to get the index of the hyperparameter based on the name
        self.hp_index = {f.name: i for i, f in enumerate(hp_fields)}
    
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
    
    def tensor_to_hparams(self, hp : torch.Tensor):
        """
        Transforms the tensor hyperparameters that BoTorch outputs into arguments of the HyperParameters class.
        Helps obtain the value of the candidate hyperparameters.

        Args:
            hp (torch.Tensor): Candidate hyperparameters

        Returns:
            HyperParameters: Dataclass containing values of hyperparameters
        """        
        hp_kwargs = {}

        for field in fields(self.hp_dataclass):
            name = field.name
            index = self.hp_index.get(name)

            value = hp[index].item()

            if field.type is int:
                hp_kwargs[name] = int(value)
            elif field.type is float:
                hp_kwargs[name] = float(value)
            else:
                hp_kwargs[name] = value
            
        hyperparams = self.hp_dataclass(**hp_kwargs)

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

            match self.model_arch:
                case Model.CROSSENTROPY | Model.CROSSATTENTION:
                    best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1 = results
                case Model.COSINETHRESHOLD:
                    best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1, cosines_train_set, thresholds_train_set = results
                    
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
    
    def train_for_hp_set(self, hp : HP) -> tuple[Any, ...]:
        """
        Trains a model based on a candidate set of hyperparameters.

        Args:
            hyperparams (torch.Tensor): Candidate hyperparameters
        Returns:
            tuple[Any, ...]: Outputs the best params across epochs as well as all metrics that Train class outputs. It is of variable
            length depending if we are using the cosine similarity trainer or not
        """        

        fc_sizes = [hp.fc1, hp.fc2, hp.fc3]

        if self.model_arch == Model.CROSSATTENTION:
            extra_hp = {
                'use_n_layers_cross_att' : hp.use_n_layers_cross_att,
                'fc_cross_att_sizes' : [hp.fc1_cross_att, hp.fc2_cross_att]
            }

        model, optimizer, scheduler, train_loader, val_loader = self.train_config(hp.lr, hp.weight_decay, fc_sizes, hp.use_n_layers, hp.dropout, hp.batch_size, **extra_hp)

        assert isinstance(hp.n_freeze, int), 'n_freeze must be int'

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

        del trainer, model, optimizer, scheduler, train_loader, val_loader

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
                     batch_size : int,
                     use_n_layers_cross_att : int = 0,
                     fc_cross_att_sizes : list[int] = None
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

        match self.model_arch:
            case Model.CROSSENTROPY:
                model = PairClassifier.CrossEntropy(self.sbert_model, fc_sizes, use_n_layers, self.device, self.fixed, dropout)
            case Model.COSINETHRESHOLD:
                model = PairClassifier.CosSimilarity(self.sbert_model, fc_sizes, use_n_layers, self.device, self.fixed, dropout)
            case Model.CROSSATTENTION:
                model = PairClassifier.CrossAttention(self.sbert_model, fc_sizes, use_n_layers, self.device, False, use_n_layers_cross_att, fc_cross_att_sizes, dropout)
            case _:
                raise(TypeError(f'{self.model_arch} is not a valid Model'))

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
            
        if self.fixed and self.model_arch != Model.CROSSATTENTION:
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

            match self.model_arch:
                case Model.CROSSENTROPY | Model.CROSSATTENTION:
                    best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1 = results
                case Model.COSINETHRESHOLD:
                    best_params, avg_batch_train_loss, epoch_train_acc, avg_batch_val_loss, epoch_val_acc, epoch_val_f1, cosines_train_set, thresholds_train_set = results
            
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
          