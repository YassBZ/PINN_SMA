import numpy as np
import h5py
import multiprocessing as mp
import random

def extract_sample_batch(sample_batch, props, inputs, strains, stresses):
    xs, ys = [], []
    for i, t in sample_batch:
        try:
            prop = props[i]
            input_ = inputs[i]
            eps_t = strains[i][t - 1]
            delta_eps = strains[i][t] - eps_t
            sigma_t = stresses[i][t - 1]
            sigma_tp1 = stresses[i][t]

            x = np.concatenate([prop, input_, eps_t, delta_eps, sigma_t]).astype(np.float32)
            y = sigma_tp1.astype(np.float32)

            xs.append(x)
            ys.append(y)
        except Exception as e:
            print(f"Skipped (i={i}, t={t}) due to error: {e}")
            continue
    return xs, ys

def process_wrapper(args):
    return extract_sample_batch(*args)

def preprocess_sampled_npz_to_h5_parallel(npz_path, h5_path, target_sample_count=2_000_000, num_workers=mp.cpu_count(), seed=42):
    np.random.seed(seed)
    random.seed(seed)

    print(f"[INFO] Loading metadata from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    props = data["Props"]
    inputs = data["Input"]
    strains = data["strain"]
    stresses = data["stress"]

    # Step 1: Build global index list
    print("[INFO] Indexing all (i, t) pairs...")
    global_index = []
    for i in range(len(strains)):
        T = strains[i].shape[0]
        global_index.extend([(i, t) for t in range(1, T)])

    print(f"[INFO] Total available samples: {len(global_index):,}")
    print(f"[INFO] Randomly selecting {target_sample_count:,} samples...")

    selected_samples = random.sample(global_index, k=target_sample_count)

    # Step 2: Split into chunks
    chunks = np.array_split(selected_samples, num_workers)
    args = [(chunk.tolist(), props, inputs, strains, stresses) for chunk in chunks]

    # Step 3: Process in parallel
    all_x, all_y = [], []
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(process_wrapper, args):
            xs, ys = result
            all_x.extend(xs)
            all_y.extend(ys)

    # Step 4: Write to HDF5
    print(f"[INFO] Writing {len(all_x):,} samples to {h5_path}...")
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("X", data=np.stack(all_x), compression="gzip")
        h5f.create_dataset("Y", data=np.stack(all_y), compression="gzip")

    print(f"[DONE] Saved {len(all_x):,} samples to {h5_path}")
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Safe on macOS
    preprocess_sampled_npz_to_h5_parallel(
        npz_path="/Users/yassinebz/Research/Independent Research/PINN SMA/PINN SMA/merged_dataset.npz",
        h5_path="/Users/yassinebz/Research/Independent Research/PINN SMA/PINN SMA/subset20M.h5",
        target_sample_count=50000000,
        num_workers=10
    )