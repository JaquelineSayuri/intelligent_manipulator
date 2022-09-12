import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ast import literal_eval

def open_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        List = [element.replace("\n","") for element in lines]
    return List

paths = open_file("paths.txt")
paths = [literal_eval(path) for path in paths]

xt_s_list = []
yt_s_list = []
zt_s_list = []

xt_list = []
yt_list = []
zt_list = []

paths = paths[10:]
for episode, path in enumerate(paths):
	if episode%1 == 0:
	#if len(path) == 350:
	#if episode > 42600:
		x, y, z = [], [], []
		xt, yt, zt = path[0][0], path[0][1], path[0][2]
		xi, yi, zi = path[1][0], path[1][1], path[1][2]
		xf, yf, zf = path[-1][0], path[-1][1], path[-1][2]
		path = path[1:]
		for position in path:
		    x.append(position[0])
		    y.append(position[1])
		    z.append(position[2])

		if len(path) < 350:
			xt_s_list.append(xt)
			yt_s_list.append(yt)
			zt_s_list.append(zt)
		else:
			xt_list.append(xt)
			yt_list.append(yt)
			zt_list.append(zt)

		'''
		plt.figure(figsize=(5, 10))

		plt.subplot(211)
		plt.plot(y, x, 'b', yt, xt, 'ro', yi, xi, 'go', yf, xf, 'ko')
		plt.xlabel("Y")
		plt.ylabel("X")
		plt.xlim([-70, 70])
		plt.ylim([-70, 70])
		plt.title(f'Episode: {episode}')
		#plt.savefig(f'episode_{episode}')

		plt.subplot(212)
		plt.plot(y, z, 'b', yt, zt, 'ro', yi, zi, 'go', yf, zf, 'ko')
		plt.xlabel("Y")
		plt.ylabel("Z")
		plt.xlim([-70, 70])
		plt.ylim([-70, 70])
		#plt.savefig(f'episode_{episode}')
		'''
		#'''
		fig = plt.figure(figsize=(15, 15))
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(xs = x, ys = y, zs = z)
		ax.scatter(xt, yt, zs=zt, c='r')
		ax.scatter(xi, yi, zs=zi, c='g')
		ax.scatter(xf, yf, zs=zf, c='k')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		plt.title(f'Episódio: {episode+1}')
		ax.set_xlim(-40, 40)
		ax.set_ylim(-40, 40)
		ax.set_zlim(0, 40)

		plt.show()
		#'''
'''
fig2 = plt.figure(figsize=(15, 12))
ax = fig2.add_subplot(111, projection='3d')
#ax.plot(xs = xi, ys = yi, zs = zi, c = 'go')
#ax.scatter(xt_s_list, yt_s_list, zs=zt_s_list, c='g')
ax.scatter(xt_list, yt_list, zs=zt_list, c='r')
#ax.scatter(xi, yi, zs=zi, c='g')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(f'Episódio: {episode+1}')
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_zlim(0, 40)

plt.show()
'''
