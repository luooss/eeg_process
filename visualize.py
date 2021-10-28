import numpy as np
from dset import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


subjects = ['chenyi', 'huangwenjing', 'huangxingbao', 'huatong', 'wuwenrui', 'yinhao']
txts = []
for im in range(90):
    for s in range(4):
        txts.append(im)

for subj in subjects:
    print(subj)
    data_path = '/mnt/xlancefs/home/gwl20/data_new/{}_data_de.npy'.format(subj)
    label_path = '/mnt/xlancefs/home/gwl20/data_new/{}_label.npy'.format(subj)
    dset = ArtDataset([data_path], [label_path], freq_band='all')

    data = dset.data.numpy().reshape(360, -1)
    labels = dset.label.numpy()

    perp = 20

    tsne = TSNE(n_components=2, perplexity=perp, n_iter=5000, n_jobs=-1)
    data_2d = tsne.fit_transform(data)

    X = data_2d[:, 0]
    Y = data_2d[:, 1]

    fig, ax = plt.subplots(figsize=(24.8, 17.8))
    ax.scatter(X, Y, c=labels)
    for i in range(360):
        ax.annotate(txts[i], (X[i], Y[i]))
    
    ax.set_title(subj)

    fig.savefig('./visualization/fig_{}_{}.png'.format(subj, perp))