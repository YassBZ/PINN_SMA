import h5py
import os
import yaml
import itertools
import re

import numpy as np

PropsIdx = {
    "YoungModulus": 0,
    "PoissonRatio": 1,
    "DilationFactor": 2,
    "EpsilonTracT": 3,
    "EpsilonTracTFA": 4,
    "EpsilonCompT": 5,
    "Bf": 6,
    "Br": 7,
    "Ms": 8,
    "Af": 9,
    "rf": 10,
    "Fepsilon": 11,
    "Hf": 12,
    "Hepsilon": 13,
    "Htwin": 14,
    "Hs": 15,
    "alpha0": 16,
    "alpha1": 17,
    "alpha2": 18,
    "n": 19
}

InputIdxPressure = {
    "Px": 0,
    "Py": 1,
    "Pz": 2,
    "T": 3
}

def LoadGridInput(yamlfile):
    with open(yamlfile, 'r') as stream:
        input_dict = yaml.full_load(stream)

    assert input_dict["Mode"] == "Grid"
    PropsDict = input_dict["PropsGrid"]
    InputDict = input_dict["InputGrid"]
    ConfigDict = input_dict["Config"]
    TotalCases = 0
    print(PropsDict)
    print(InputDict)
    for key, item in PropsDict.items():
        if item["Nsteps"] is None and item["Steps"] is not None:
            numbercases = int(np.abs(item["Start"] - item["End"]) / item["Steps"]) + 1
        elif item["Nsteps"] is not None and item["Steps"] is None:
            numbercases = item["Nsteps"]
        else:
            raise Exception

        if TotalCases == 0:
            TotalCases += numbercases
        else:
            TotalCases *= numbercases

    print(f"Total Cases: {TotalCases}")

    return PropsDict, InputDict, ConfigDict, TotalCases

def PrepareOutput(folder, props_grid_dict, input_grid_dict, config_dict, Ncases, PropsIdx=PropsIdx, InputIdx=InputIdxPressure):
    InputSize = config_dict["InputSize"]
    PropsSize = config_dict["PropsSize"]
    ConstantProps = config_dict["ConstantProps"]
    ConstantInputs = config_dict["ConstantInputs"]

    PropsGridKeys = list(props_grid_dict.keys())
    InputGridKeys = list(input_grid_dict.keys())
    print(f"PropsGridKeys: {PropsGridKeys}")
    print(f"InputGridKeys: {InputGridKeys}")

    PropsRanges = []
    InputRanges = []

    for keydict in props_grid_dict.values():
        if keydict["Steps"] is not None:
            proprange = np.arange(keydict["Start"], keydict["End"]+1E-10, keydict["Steps"])
        else:
            proprange = np.linspace(keydict["Start"], keydict["End"], keydict["Nsteps"])

        PropsRanges.append(proprange)

    for keydict in input_grid_dict.values():
        if keydict["Steps"] is not None:
            inputrange = np.arange(keydict["Start"], keydict["End"] + 1E-10, keydict["Steps"])
        else:
            inputrange = np.linspace(keydict["Start"], keydict["End"], keydict["Nsteps"])

        InputRanges.append(inputrange)
    print(f"PropsRanges: {PropsRanges}")
    print(f"InputRanges: {InputRanges}")
    all_prop_combinations = list(itertools.product(*PropsRanges))
    all_input_combinations = list(itertools.product(*InputRanges))

    full_props = []
    full_inputs = []

    for combo in all_prop_combinations:
        vec = np.zeros(PropsSize)
        for key, val in ConstantProps.items():
            vec[PropsIdx[key]] = val

        for i, key in enumerate(PropsGridKeys):
            vec[PropsIdx[key]] = combo[i]

        full_props.append(vec)

    for combo in all_input_combinations:
        vec = np.zeros(InputSize)
        for key, val in ConstantInputs.items():
            vec[InputIdx[key]] = val

        for i, key in enumerate(InputGridKeys):
            vec[InputIdx[key]] = combo[i]

        full_inputs.append(vec)

    CompletePropsSet = np.array(full_props)
    CompleteInputSet = np.array(full_inputs)

    print(f"CompletePropsSet: {len(CompletePropsSet)}")
    print(f"Estimated number of Cases: {Ncases}")

    if config_dict["ArrayExtent"] is not None:
        extent = (config_dict["ArrayExtent"][1] - config_dict["ArrayExtent"][0]) + 1
        Ncasesperarray = len(CompletePropsSet) // extent
        print("Each array job has to treat: ", Ncasesperarray)
        cases = np.array_split(CompletePropsSet, extent)

        for k in range(extent):
            casenumber = config_dict["ArrayExtent"][0] + k
            patharrayjob = os.path.join(folder, f"ArrayJob_{casenumber}")
            os.makedirs(patharrayjob, exist_ok=True)
            propsarrayjob = cases[k]
            CPUcases = np.array_split(propsarrayjob, config_dict["NCPUperNode"])
            for l in range(len(CPUcases)):
                cpupath = os.path.join(patharrayjob, f"CPU_{l}")
                os.makedirs(cpupath, exist_ok=True)
                props = CPUcases[l]
                with h5py.File(os.path.join(cpupath, f"Array{k}_Output_CPU{l}.h5"), 'w') as file:
                    file.create_dataset("PropsGrid", data=props)
                    file.create_dataset("InputGrid", data=CompleteInputSet)

    else:
        with h5py.File(os.path.join(folder, f"GridOutput.h5"), 'w') as file:
            file.create_dataset("PropsGrid", data=CompletePropsSet)
            file.create_dataset("InputGrid", data=CompleteInputSet)

def GenerateINPFiles(folder, template_file):
    folders = [f for f in os.listdir(folder) if f.startswith("ArrayJob")]
    for fold in folders:
        patharray = os.path.join(folder, fold)
        CPUdirectories = [f for f in os.listdir(os.path.join(folder, fold))]

        for CPUdir in CPUdirectories:
            pathCPU = os.path.join(patharray, CPUdir)
            hdf5files = [f for f in os.listdir(pathCPU) if f.endswith(".h5")]
            f = os.path.join(pathCPU, hdf5files[0])
            with h5py.File(f, 'r') as file:
                input_grid = file["InputGrid"][:]
                input_props = file["PropsGrid"][:]
            for k in range(len(input_props)):
                for j in range(len(input_grid)):
                    with open(template_file, "r") as f:
                        lines = f.readlines()

                    updatedlines = []

                    pressure_vector = input_grid[j, :]
                    props_vector = input_props[k, :]

                    insideprops = False
                    insideload = False
                    patternprops = re.compile(r"\*User Material.*")
                    patternload = re.compile(r"\*\* STEP: Charge.*")
                    patternunload = re.compile(r"\*\* STEP: Decharge.*")

                    i = 0
                    while i < len(lines):
                        line = lines[i]
                        if patternprops.match(line):
                            updatedlines.append(line)
                            insideprops = True
                            i += 1
                            continue

                        if patternload.match(line):
                            updatedlines.append(line)
                            insideload = True
                            i += 1
                            continue

                        if patternunload.match(line):
                            updatedlines.append(line)
                            insideload = False
                            i += 1
                            continue

                        if insideprops:
                            if line.startswith("*"):
                                insideprops = False
                                for p in range(0, 20, 8):
                                    chunk = props_vector[p:p+8]
                                    updatedlines.append(', '.join(f"{x:.8g}" for x in chunk) + "\n")
                                updatedlines.append(line)
                                i += 1
                                continue
                            else:
                                i += 1
                                continue

                        if line.strip().lower().startswith(
                                "*initial conditions") and "type=temperature" in line.lower():
                            # Modify the next line with the new temperature
                            updatedlines.append(line)
                            nextline = lines[i + 1]
                            parts = nextline.strip().split(",")
                            parts[-1] = f"{pressure_vector[3]:.8g}"  # Format to 6 decimal places
                            updatedlines.append(", ".join(parts) + "\n")
                            i += 2
                            continue

                        if line.strip() == '*Dsload' and i + 1 < len(lines):
                            nextline = lines[i + 1]
                            surface = nextline.split(',')[0].strip()
                            if surface == "Surf-2":
                                updatedlines.append(line)
                                updatedlines.append(f"{surface}, P, {pressure_vector[0]:.8g}\n") if insideload else updatedlines.append(f"{surface}, P, {0.:.8g}\n")
                                i += 2
                                continue
                            elif surface == "Surf-3":
                                updatedlines.append(line)
                                updatedlines.append(f"{surface}, P, {pressure_vector[1]:.8g}\n") if insideload else updatedlines.append(f"{surface}, P, {0.:.8g}\n")
                                i += 2
                                continue
                            elif surface == "Surf-4":
                                updatedlines.append(line)
                                updatedlines.append(f"{surface}, P, {pressure_vector[2]:.8g}\n") if insideload else updatedlines.append(f"{surface}, P, {0.:.8g}\n")
                                i += 2
                                continue

                        updatedlines.append(line)
                        i += 1

                    with open(os.path.join(pathCPU, f"Case{k}_Input{j}.inp"), "w") as f:
                        f.writelines(updatedlines)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Abaqus Input and Output file structure for Cluster Submission")
    parser.add_argument("-t", "--template_file", required=True, help="Template file")
    parser.add_argument("-i", "--input", required=True, help="Input File Path")
    parser.add_argument("-o", "--output", required=True, help="Output Folder Path")
    args = parser.parse_args()

    props_grid_dict, input_grid_dict, config_dict, Ncases = LoadGridInput(args.input)
    PrepareOutput(args.output, props_grid_dict, input_grid_dict, config_dict, Ncases)
    GenerateINPFiles(args.output, args.template_file)
