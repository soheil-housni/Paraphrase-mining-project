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


class BOSearchTrain():

    NUM_RESTARTS = 5
    RAW_SAMPLES = 50
    WARMUP_STEPS_RATIO = 0.1

    def __init__(self,
                 bounds : torch,
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

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def fit_GP(self) -> SingleTaskGP:
        model = SingleTaskGP(self.X_observed.double(), self.Y_observed.double())
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def init_hp_samples(self):
        return self.bounds[0, :] + (self.bounds[1, :] - self.bounds[2, :]) * torch.rand(self.n_init_samples, self.bounds.shape[1])

    def init_fit_samples(self, init_X : torch.Tensor):
        logger.info(f'Intial fit of GP for {self.n_init_samples} samples of hyperparameter sets')
        self.X_observed = init_X
        Y_observed = torch.zeros(self.n_init_samples)
        for n in range(self.n_init_samples):
            # TODO ADD TRAINING FUNCTION TO FIT INITIAL X AND GET INITIAL Y
            Y_observed[n] = 
        self.Y_observed = Y_observed 

    def sample_set(self, model : SingleTaskGP, Y_observed : torch.Tensor, n_candidates = 1) -> torch.Tensor:
        best_f = Y_observed.min()
        acq_fct = qExpectedImprovement(model = model, best_f = best_f)

        acq_bounds = self.bounds.T.double()

        canditates = optimize_acqf(acq_function = acq_fct,
                                   bounds = acq_bounds,
                                   q = n_candidates,
                                   num_restarts = BOSearchTrain.NUM_RESTARTS,
                                   raw_samples = BOSearchTrain.RAW_SAMPLES
                                   )
        return canditates
    
    def train_config(self, lr, weight_decay, fc_sizes, use_n_layers, dropout, batch_size):
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

        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
        train_loader = DataLoader(dataset = TextPairDataset(self.X_train, self.y_train, tokenization=True, tokenizer=tokenizer), batch_size = batch_size, shuffle = True, num_workers = 4)
        val_loader = DataLoader(dataset = TextPairDataset(self.X_val, self.y_val, tokenization=True, tokenizer=tokenizer), batch_size = batch_size, shuffle = True, num_workers = 4)
        return model, optimizer, scheduler

    def search_loop(self):
        hp_set = self.init_hp_samples()
        self.init_fit_samples()
        gp_model = self.fit_GP()
        for n in range(self.n_iterations):
            logger.info(f"Starting iteration {n}:\n" f"{[(name, hp_set[idx]) for idx, name in enumerate(self.names)]}")
        
        