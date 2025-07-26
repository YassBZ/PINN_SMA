import pickle
import torch
from torch.utils.data import Dataset

class NeuralDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.X = data["X"]
        self.Y = data["Y"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]