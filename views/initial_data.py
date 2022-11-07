from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np


def plot_graph(images_data, threshold=0.5):
    x = np.linspace(-2.5, 2.5, 100)
    y1 = x-threshold
    y2 = x+threshold
    fig, axs = plt.subplots(10, 10, figsize=(40, 40))

    all_labels = {}

    for idx, each_cls in enumerate(list(images_data.keys())):
        predict_class = images_data[each_cls]
        Kmean = KMeans(n_clusters=1)
        labels = Kmean.fit_predict(predict_class)
        all_labels[each_cls] = labels

        axs[idx//10, idx % 10].scatter(predict_class[labels == 0, 0],
                                       predict_class[labels == 0, 1], s=100, c='red', label='Cluster 0')
        axs[idx//10, idx % 10].scatter(Kmean.cluster_centers_[:, 0], Kmean.cluster_centers_[
                                       :, 1], s=300, c='yellow', label='Centroids')
        axs[idx//10, idx % 10].plot(x, y1, '-r', label='y=x-1', c='blue')
        axs[idx//10, idx % 10].plot(x, y2, '-r', label='y=x+1', c='blue')
        axs[idx//10, idx % 10].set_title(f'Class {each_cls}')

    # plt.show()
    plt.savefig('Initial Data Distribution.png')
    return all_labels
