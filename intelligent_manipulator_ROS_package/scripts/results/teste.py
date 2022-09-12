import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def open_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        List = [element.replace("\n","") for element in lines]
        List = [int(element) for element in List]
    return List

steps = open_file("steps_per_episode.txt")
steps = steps[10:]
print(len(steps))

avg_steps = [np.mean(steps[:i]) for i in range(1,len(steps)+1)]
#print(avg_steps)
print(np.mean(steps))
#plt.plot(avg_steps)

plt.figure(figsize=(25, 12))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax = plt.subplot(111)

ax.bar(range(1,len(avg_steps)+1), avg_steps, color = "lightskyblue")
ax.plot(range(1,len(avg_steps)+1), [avg_steps[-1]]*len(avg_steps), color = "dodgerblue", linewidth=3)
# Hide the right and top spines
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.title(f"Experimento {seed}", fontsize = 20)
plt.xlabel("Nº de episódios", fontsize = 25)
plt.ylabel("Média de passos", fontsize = 25)
plt.savefig(f"avg_steps.svg",
	format="svg",pad_inches=0,bbox_inches='tight')



success_list = []
for s in steps:
    if s < 100:
        success_list.append(1)
    else:
        success_list.append(0)

success_rate_list = [sum(success_list[:i])/len(success_list[:i]) for i in range(1,len(success_list)+1)]
print(sum(success_list)/len(success_list))

plt.figure(figsize=(25, 12))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax = plt.subplot(111)

ax.bar(range(1,len(success_rate_list)+1), success_rate_list, color = "mediumpurple")
ax.plot(range(1,len(success_rate_list)+1), [success_rate_list[-1]]*len(success_rate_list),color="indigo",linewidth=3)
# Hide the right and top spines
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.title(f"Experimento {seed}", fontsize = 20)
plt.xlabel("Nº de episódios", fontsize = 25)
plt.ylabel("Taxa de sucesso", fontsize = 25)
plt.savefig(f"success.svg",
	format="svg",pad_inches=0,bbox_inches='tight')