import os
import subprocess
from multiprocessing import Pool

def run_folder(cpu_folder):
    umat_path = "/storage/hive/scratch/6/yzineb3/PINN_SMA/Cluster/AMF3D.o"

    command = [
        "abaqus", "python", os.path.abspath("RunandExtractCaseAbaqus.py"),
        cpu_folder,
        umat_path,
        "--part_instance", "CUBE-1",
        "--element_id", "1",
        "--ip_index", "0"
    ]
    print(f"Launching: {' '.join(command)}")
    subprocess.run(command, check=True)

def main(base_dir, NCPU):
    cpu_folders = [os.path.join(base_dir, f"CPU_{i}") for i in range(NCPU)]
    for folder in cpu_folders:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")

    with Pool(processes=NCPU) as pool:
        pool.map(run_folder, cpu_folders)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run folders in parallel (one per CPU), each writing its own HDF5.")
    parser.add_argument("base_dir", help="Base directory containing NCPU folders")
    parser.add_argument("NCPU", type=int, help="Number of CPUs")
    args = parser.parse_args()

    main(args.base_dir, args.NCPU)
