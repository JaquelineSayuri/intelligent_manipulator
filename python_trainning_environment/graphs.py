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

def get_n_steps(seed):
    steps = open_file(f"results/steps_per_episode_{seed}.txt")
    steps = [int(e) for e in steps]
    steps = steps[13500:14000]
    print("media de steps", np.mean(steps[-1000:]))
    plt.figure(figsize=(25, 12))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax = plt.subplot(111)

    ax.scatter(range(13500,13500+len(steps)), steps, c = "#f21111")
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.title(f"Experimento {seed}", fontsize = 20)
    plt.xlabel("Episódio", fontsize = 25)
    plt.ylabel("Nº de passos", fontsize = 25)
    plt.savefig(f"graphs/steps_per_episode_{seed}.svg", format="svg",pad_inches=0,bbox_inches='tight')
    plt.close()
    return steps

def get_success_rate_list(steps, max_steps):
    T = 30
    success_list = []
    for s in steps:
        if s < max_steps:
            success_list.append(1)
        else:
            success_list.append(0)

    success_rate_list = []
    for i in range(1,len(success_list)):
        try:
            total_success = sum(success_list[round(i-T/2):round(i+T/2)])
            success_rate = total_success/len(success_list[round(i-T/2):round(i+T/2)])
        except:
            total_success = sum(success_list[:i])
            success_rate = total_success/len(success_list[:i])
        success_rate_list.append(success_rate)

    print("taxa de sucesso",success_rate_list[-1])

    return success_rate_list

def get_success_rate(steps, max_steps, seed):
    T = 300
    success = get_success_rate_list(steps, max_steps)
    plt.figure(figsize=(25, 12))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax = plt.subplot(111)
    #plt.scatter(range(len(success)), success, c = "lightsalmon")
    ax.plot(range(len(success)), success, color = 'lightsalmon', linewidth = 5)    
    
    avg_success = []
    for i in range(1,len(success)):
        try:
            total_success = sum(success[round(i-T/2):round(i+T/2)])
            success_rate = total_success/len(success[round(i-T/2):round(i+T/2)])
        except:
            total_success = sum(success[:i])
            success_rate = total_success/len(success[:i])
        avg_success.append(success_rate)

    #plt.scatter(range(len(avg_success)), avg_success, c = "orangered")
    ax.plot(range(len(avg_success)), avg_success, color = 'orangered')
    #plt.title(f'Experimento {seed}', fontsize = 20)
    plt.xlabel("Episódio", fontsize = 25)
    plt.ylabel(f"Taxa de sucesso", fontsize = 25)
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig(f'graphs/episodes_success_rate_{seed}.svg', format="svg",pad_inches=0,bbox_inches='tight')
    plt.close()

def get_total_reward(seed):
    T = 100
    total_reward = open_file(
        f"results/total_score_per_episode_{seed}.txt"
    )
    avg_scores = open_file(f"results/avg_scores_{seed}.txt")
    total_reward = [float(e) for e in total_reward]
    avg_scores = [float(e) for e in avg_scores]
    total_reward = total_reward[500:14000]
    avg_scores = avg_scores[500:14000]
    plt.figure(figsize=(25, 12))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax = plt.subplot(111)
    ax.plot(range(len(total_reward)), total_reward, color = "lime")
    ax.plot(range(len(avg_scores)), avg_scores, color = "green")
    #plt.title(f"Experimento {seed}", fontsize = 20)
    plt.xlabel("Episódio", fontsize = 25)
    plt.ylabel(f"Recompensa total", fontsize = 25)
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig(f"graphs/total_score_per_episode_{seed}.svg", format="svg",pad_inches=0,bbox_inches='tight')
    plt.close()



def get_graphs(max_steps, seed):
    steps = get_n_steps(seed)
    get_success_rate(steps, max_steps, seed)
    get_total_reward(seed)

#get_graphs(350, 2)