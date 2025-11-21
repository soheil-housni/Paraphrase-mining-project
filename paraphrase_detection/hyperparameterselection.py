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

class BOSearchTrain():
    def __init__(self,
                 bounds : torch,
                 sbert_model : SentenceTransformer,
                 X_train : np.ndarray,
                 y_train : np.ndarray,
                 X_val : np.ndarray,
                 y_val : np.ndarray,
                 optimizer : torch.optim,
                 scheduler,
                 criterion : torch.nn,
                 device : torch.device,
                 n_init_samples : int
                 ) -> None:
        self.bounds = bounds
        self.n_init_samples = n_init_samples
        self.sbert_model = sbert_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def fit_GP(self, X : torch.Tensor, Y : torch.Tensor):
        X = X.double()
        Y = Y.double()

        model = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def sample_from_space(self):
        return self.bounds[0, :] + (self.bounds[1, :] - self.bounds[2, :]) * torch.rand(self.n_init_samples, self.bounds.shape[1])

    def train_eval(self):
        