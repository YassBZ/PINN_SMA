import torch
import numpy as np
from NeuralNets.MLP import MLPConstitutiveModel
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import simpson
from src.PlotParams import plotParams
from torchviz import make_dot

matplotlib.use('MacOSX')

dataset = "/Users/yassinebz/Research/Independent Research/PINN SMA/PINN SMA/merged_dataset.npz"
model_path = "/Users/yassinebz/Research/Independent Research/PINN SMA/PINN SMA/BackUp/b512_l1e-3_H256_D5_E200_Dropout0.2_Decay1E-4.pt"
model_params = model_path.split("/")[-1].split("_")
data = np.load(dataset, allow_pickle=True)
randidx = np.random.randint(0, len(data["strain"]))
#randidx = 21696
#randidx = 269625
teststrain = data["strain"][randidx]
testprops = data["Props"][randidx]
testinputs = data["Input"][randidx]
teststress = data["stress"][randidx]

print(randidx)
xs = []

for t in range(1, len(teststrain)):
    eps_t = teststrain[t-1]
    delta_eps = teststrain[t] - eps_t
    sigma_t = teststress[t-1]

    x = np.concatenate([testprops, testinputs, eps_t, delta_eps, sigma_t]).astype(np.float32)
    xs.append(x)

# --- Device selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

# --- Hyperparameters ---
HIDDEN_DIM = int(model_params[2][1:])
DEPTH = int(model_params[3][1:])
DROPOUT_RATE = float(model_params[5][7:])

model = MLPConstitutiveModel(input_dim=42, output_dim=6, hidden_dim=HIDDEN_DIM, depth=DEPTH, dropout_rate=DROPOUT_RATE).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

xs_tensor = torch.tensor(xs, dtype=torch.float32).to(device)
with torch.no_grad():
    yhat = model(xs_tensor)
    output = yhat.cpu().numpy()

y_pred = np.zeros((output.shape[0]+1, output.shape[1]))
y_pred[1:, :] = output

abs_diff = np.abs(y_pred[:, 0] - teststress[:, 0])
area_diff = simpson(abs_diff, teststrain[:, 0])
print(f"[INFO] Area difference: {area_diff}")
denominator = simpson(np.abs(teststress[:, 0]), teststrain[:, 0])
print(f"[INFO] Denominator: {denominator}")
print("[INFO] Integrated Relative Error", area_diff / denominator)
make_dot(yhat, params=dict(list(model.named_parameters()))).render("nn_torchviz", format="png")
plotParams()
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel(r"$\varepsilon_{11}$")
ax.set_ylabel(r"$\sigma_{11}$")
ax.text(0.83, 0.05, f"IARE: {np.round(100 * area_diff / denominator, 2)} \\%", transform=ax.transAxes)
ax.plot(teststrain[:, 0], teststress[:, 0], label=f"Ground Truth")
ax.plot(teststrain[:, 0], y_pred[:, 0], label=f"Prediction", color="red", linestyle="dashed")
ax.legend()
plt.tight_layout()
plt.show()
