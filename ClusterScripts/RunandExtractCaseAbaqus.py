import os
import re
import numpy as np
import subprocess
from odbAccess import openOdb, INTEGRATION_POINT

def run_abaqus_inp(inp_path, umat_path):
    job_name = os.path.splitext(os.path.basename(inp_path))[0]
    command = [
        "abaqus", "job={}".format(job_name), "input={}".format(inp_path),
        "user={}".format(umat_path), "cpus=1", "interactive"
    ]
    ret = subprocess.call(command)

    if ret != 0:
        print("Abaqus run failed with code {}".format(ret))

def extract_history_from_odb(inp_path, odb_path, part_instance='CUBE-1', element_id=1, ip_index=0):
    job_name = os.path.splitext(os.path.basename(inp_path))[0]
    odb = openOdb(odb_path)
    dirpath = os.path.dirname(inp_path)
    npzpath = os.path.join(dirpath, 'NPZFiles')

    stress_history = []
    strain_history = []
    frame_times = []

    region = odb.rootAssembly.ElementSetFromElementLabels(
        name='TEMPSET_{}'.format(job_name),
        elementLabels=((part_instance, (element_id,)),)
    )

    for stepname in odb.steps.keys():
        step = odb.steps[stepname]
        frames = step.frames

        for frame in frames:
            try:
                stress_field = frame.fieldOutputs['S']
                strain_field = frame.fieldOutputs['LE']
                s_vals = stress_field.getSubset(region=region, position=INTEGRATION_POINT).values
                le_vals = strain_field.getSubset(region=region, position=INTEGRATION_POINT).values

                stress_history.append(s_vals[ip_index].data)
                strain_history.append(le_vals[ip_index].data)
                frame_times.append(frame.frameValue)

            except (IndexError, ValueError):
                stress_history.append([np.nan] * 6)
                strain_history.append([np.nan] * 6)
                frame_times.append(np.nan)

    odb.close()
    np.savez(os.path.join(npzpath, "temp_{}.npz".format(job_name)), strain=np.array(strain_history), stress=np.array(stress_history), time=np.array(frame_times))
    print(os.path.join(npzpath, "temp_{}.npz".format(job_name)))
    return os.path.join(npzpath, "temp_{}.npz".format(job_name))

def process_folder(inp_folder, umat_path, part_instance='CUBE-1', element_id=1, ip_index=0):
    inp_files = [f for f in os.listdir(inp_folder) if f.endswith(".inp")]
    NPZfolder = os.path.join(inp_folder, "NPZFiles")
    if not os.path.isdir(NPZfolder):
        os.mkdir(NPZfolder)
    for inp_file in inp_files:
        match = re.match(r"(Case\d+)_Input(\d+)\.inp", inp_file)
        if not match:
            print("Skipping invalid file: {}".format(inp_file))
            continue

        case_name = match.group(1)
        input_name = "Input{}".format(match.group(2))

        inp_path = os.path.join(inp_folder, inp_file)
        print("Running: {}".format(inp_file))
        run_abaqus_inp(inp_path, umat_path)

        odb_path = os.path.join(inp_folder, os.path.splitext(inp_file)[0] + ".odb")
        print("Extracting: {}".format(odb_path))

        tempdatapath = extract_history_from_odb(
            inp_path,
            odb_path,
            part_instance=part_instance,
            element_id=element_id,
            ip_index=ip_index
        )

        command = [
            "bash",
            "CleanFiles.sh",
            inp_folder,
            "{}_{}".format(case_name, input_name)
        ]

        ret = subprocess.call(command)
        if ret != 0:
            print("Cleaning File failed with code {}".format(ret))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Abaqus + UMAT and extract tensor histories to HDF5.")
    parser.add_argument("inp_folder", help="Folder containing .inp and .odb files for this job")
    parser.add_argument("umat_path", help="Path to your UMAT file (Fortran)")
    parser.add_argument("--element_id", type=int, default=1, help="Element ID to extract from")
    parser.add_argument("--ip_index", type=int, default=0, help="Integration point index")
    parser.add_argument("--part_instance", type=str, default="CUBE-1", help="Name of part instance in odb")
    args = parser.parse_args()

    process_folder(
        args.inp_folder,
        args.umat_path,
        part_instance=args.part_instance,
        element_id=args.element_id,
        ip_index=args.ip_index
    )