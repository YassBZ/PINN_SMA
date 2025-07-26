import h5py
import torch
import pickle

def convert_h5_to_pickle(h5_path, pkl_path):
    print(f"[INFO] Loading {h5_path}...")
    with h5py.File(h5_path, "r") as h5f:
        x_np = h5f["X"][:]
        y_np = h5f["Y"][:]

    print(f"[INFO] Converting to torch tensors...")
    data = {
        "X": torch.tensor(x_np, dtype=torch.float32),
        "Y": torch.tensor(y_np, dtype=torch.float32)
    }

    print(f"[INFO] Saving to {pkl_path}...")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[DONE] Saved {len(x_np)} samples.")

convert_h5_to_pickle("/Users/yassinebz/Research/Independent Research/PINN SMA/PINN SMA/subset5M.h5", "/Users/yassinebz/Research/Independent Research/PINN SMA/PINN SMA/subset5M.pkl")