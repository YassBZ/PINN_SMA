import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from NeuralNets.MLP import MLPConstitutiveModel
from NeuralNets.DataStructure import NeuralDataset  # or NeuralDataset for HDF5
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from src.PlotParams import plotParams


def main():
    # --- Hyperparameters ---
    hyperparam_hist = []
    EPOCHS = 20
    dataset = NeuralDataset("/Users/yassinebz/Research/Independent Research/PINN SMA/PINN SMA/subset5M.pkl")
    train_len = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
    criterion = nn.MSELoss()
    for BATCH_SIZE in Batchsize_vals:
        for LEARNING_RATE in lr_vals:
            for HIDDEN_DIM in hidden_dim_vals:
                for DEPTH in depth_vals:
                    for DROPOUT_RATE in dropout_vals:
                        for WEIGHT_DECAY in decay_vals:
                            print("[INFO] Parameters of the current model:")
                            print(f"Batch size: {BATCH_SIZE}")
                            print(f"Learning rate: {LEARNING_RATE}")
                            print(f"Hidden dimension: {HIDDEN_DIM}")
                            print(f"Depth: {DEPTH}")
                            print(f"Dropout rate: {DROPOUT_RATE}")
                            print(f"Weight decay: {WEIGHT_DECAY}")

                            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
                            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=10)
                            model = MLPConstitutiveModel(input_dim=42, output_dim=6, hidden_dim=HIDDEN_DIM, depth=DEPTH, dropout_rate=DROPOUT_RATE).to(device)
                            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

                            train_loss_hist = []
                            val_loss_hist = []

                            for epoch in range(1, EPOCHS + 1):
                                print(f"[INFO] Epoch {epoch} / {EPOCHS}")
                                model.train()
                                total_loss = 0.0
                                for x, y in train_loader:
                                    x = x.to(device, dtype=torch.float32)
                                    y = y.to(device, dtype=torch.float32)

                                    pred = model(x)
                                    loss = criterion(pred, y)

                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()
                                    total_loss += loss.item()

                                avg_train_loss = total_loss / len(train_loader)
                                train_loss_hist.append(avg_train_loss)

                                model.eval()
                                val_loss = 0.0
                                with torch.no_grad():
                                    for x, y in val_loader:
                                        x = x.to(device, dtype=torch.float32)
                                        y = y.to(device, dtype=torch.float32)
                                        pred = model(x)
                                        loss = criterion(pred, y)
                                        val_loss += loss.item()

                                avg_val_loss = val_loss / len(val_loader)
                                val_loss_hist.append(avg_val_loss)

                                print(f"Epoch {epoch:03d} | Train MSE: {avg_train_loss} | Val MSE: {avg_val_loss:.6f}")
                                scheduler.step(avg_val_loss)

                            os.makedirs("Result", exist_ok=True)
                            torch.save(model.state_dict(), f"Result/b{BATCH_SIZE}_l{LEARNING_RATE}_H{HIDDEN_DIM}_D{DEPTH}_E{EPOCHS}_Dropout{DROPOUT_RATE}_Decay{WEIGHT_DECAY}.pt")
                            history = {
                                        "train_loss": train_loss_hist,
                                        "val_loss": val_loss_hist
                            }
                            with open(f"Result/history_b{BATCH_SIZE}_l{LEARNING_RATE}_H{HIDDEN_DIM}_D{DEPTH}_E{EPOCHS}_Dropout{DROPOUT_RATE}_Decay{WEIGHT_DECAY}.pkl", "wb") as f:
                                pickle.dump(history, f)

                            print("[INFO] Model Saved Successfully")

                            with torch.no_grad():
                                yhat = model(xs_tensor)
                                output = yhat.cpu().numpy()

                            y_pred = np.zeros((output.shape[0] + 1, output.shape[1]))
                            y_pred[1:, :] = output

                            abs_diff = np.abs(y_pred[:, 0] - teststress[:, 0])
                            area_diff = simpson(abs_diff, teststrain[:, 0])
                            denominator = simpson(np.abs(teststress[:, 0]), teststrain[:, 0])
                            print("[INFO] Integrated Relative Error", area_diff / denominator)

                            hyperparam_hist.append((BATCH_SIZE, LEARNING_RATE, HIDDEN_DIM, DEPTH, DROPOUT_RATE, WEIGHT_DECAY, area_diff / denominator))
    return hyperparam_hist



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    Batchsize_vals = [32, 64, 128, 256, 512, 1024]
    lr_vals = [1e-6, 1e-4, 1e-2, 1e-1]
    hidden_dim_vals = [32, 64, 128, 256, 512, 1024]
    depth_vals = [2, 3, 4, 5, 6]
    dropout_vals = [0.01, 0.1, 0.2, 0.3]
    decay_vals = [1e-5, 1e-4, 1e-3, 1e-2]

    print(f"[INFO] Evaluating {len(Batchsize_vals) * len(lr_vals) * len(hidden_dim_vals) * len(depth_vals) * len(dropout_vals) * len(decay_vals)} Models")

    # --- Device selection ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    datasetpath = "/Users/yassinebz/Research/Independent Research/PINN SMA/PINN SMA/merged_dataset.npz"
    data = np.load(datasetpath, allow_pickle=True)

    randidx = 339146
    teststrain = data["strain"][randidx]
    testprops = data["Props"][randidx]
    testinputs = data["Input"][randidx]
    teststress = data["stress"][randidx]

    xs = []

    for t in range(1, len(teststrain)):
        eps_t = teststrain[t - 1]
        delta_eps = teststrain[t] - eps_t
        sigma_t = teststress[t - 1]

        x = np.concatenate([testprops, testinputs, eps_t, delta_eps, sigma_t]).astype(np.float32)
        xs.append(x)

    xs_tensor = torch.tensor(xs, dtype=torch.float32).to(device)

    hist = main()
    with open("Result/hyperparam_hist.pkl", "wb") as f:
        pickle.dump(hist, f)
