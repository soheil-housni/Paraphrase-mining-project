from torch.utils.data import Dataset
import numpy as np

class TextPairDataset(Dataset):
    def __init__(self, sentences : np.ndarray, labels : np.ndarray, tokenization:bool = False, tokenizer=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenization=tokenization
        if tokenizer:
            self.tokenizer=tokenizer
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx : int):
        s1, s2 = self.sentences[idx]
        label = self.labels[idx][0]
        if self.tokenization:
            enc1 = self.tokenizer(
            s1,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
            )
            enc2 = self.tokenizer(
            s2,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
            )
            enc1 = {k: v.squeeze(0) for k, v in enc1.items()}
            enc2 = {k: v.squeeze(0) for k, v in enc2.items()}
            return (enc1,enc2),label

        return (s1, s2), label
    