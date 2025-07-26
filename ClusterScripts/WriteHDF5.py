import h5py
import numpy as np

def write_to_hdf5(h5_path, group_path, tempfile):
    data = np.load(tempfile)
    with h5py.File(h5_path, 'a') as f:
        grp = f.require_group(group_path)
        grp.create_dataset("strain", data=data["strain"])  # shape: [n_frames, 6]
        grp.create_dataset("stress", data=data["stress"])
        grp.create_dataset("time", data=data["times"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Writes results in HDF5.")
    parser.add_argument("h5_path", help="Path to HDF5 file")
    parser.add_argument("group_path", help="Path to HDF5 group")
    parser.add_argument("tempfilepath", help="Strain")
    args = parser.parse_args()

    write_to_hdf5(args.h5_path, args.group_path, args.tempfilepath)