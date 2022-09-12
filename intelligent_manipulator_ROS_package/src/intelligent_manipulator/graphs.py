import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def open_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        List = [element.replace("\n","") for element in lines]
    return List

def save_results(file, result):
    # Open the file in append & read mode ("a+")
    with open(file, "a+") as file:
        # Move read cursor to the start of file.
        file.seek(0)
        # If file is not empty then append "\n"
        data = file.read(100)
        if len(data) > 0 :
            file.write("\n")
        # Append text at the end of file
        file.write(f"{result}")

def get_n_steps():
    steps = open_file(f"results/steps_per_episode.txt")
    steps = [int(e) for e in steps]
    print("avg steps", np.mean(steps))
    return steps

def get_success_rate(steps, max_steps):
    success_list = []
    for s in steps:
        if s < max_steps:
            success_list.append(1)
        else:
            success_list.append(0)
    total_success = sum(success_list)
    success_rate = total_success/len(success_list)
    print("success rate", success_rate)

def get_graphs(max_steps):
    steps = get_n_steps()
    print("steps", steps)
    get_success_rate(steps, max_steps)