import numpy as np
import matplotlib
import matplotlib.pyplot as plt



data = np.load('cyx.npy')
label = np.load('cyy.npy')
txts = []
for im in range(90):
    for s in range(4):
        txts.append(im)

X = data[:, 0]
Y = data[:, 1]

color = ['g', 'r', 'b']

fig, ax = plt.subplots(figsize=(24.8, 17.8))
ax.scatter(X, Y, c=label)

for i in range(360):
    ax.annotate(txts[i], (X[i], Y[i]))

fig.savefig('fig.png')