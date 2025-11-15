from torch.utils.data import Dataset
import numpy as np

class TextPairDataset(Dataset):
    def __init__(self, sentences : np.ndarray, labels : np.ndarray):
        self.sentences = sentences
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx : int):
        s1, s2 = self.sentences[idx]
        label = self.labels[idx][0]
        return (s1, s2), label