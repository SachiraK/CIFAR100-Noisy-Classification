import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def get_clean_data(all_labels, file_names, images_data, threshold):
    clean_files = {}
    clean_data = {}

    for cls in list(all_labels.keys()):
        clean_data[cls] = []
        clean_files[cls] = []
        labels = all_labels[cls]
        files = file_names[cls]
        predict_class = images_data[cls]
        all_x = predict_class[labels == 0, 0]
        all_y = predict_class[labels == 0, 1]

        for i in range(0, len(all_x)):
            u_limit = all_x[i] + threshold
            l_limit = all_x[i] - threshold

            if l_limit < all_y[i] < u_limit:
                clean_files[cls].append(files[i])
                clean_data[cls].append(predict_class[i])

        clean_data[cls] = np.array(clean_data[cls])

    fig, axs = plt.subplots(10, 10, figsize=(40, 40))
    x = np.linspace(-2.5, 2.5, 100)
    y1 = x-threshold
    y2 = x+threshold

    for idx, cls in enumerate(list(clean_data.keys())):
        # print(f'Class {cls} has {len(clean_data[cls])} clean labels')
        predict_class = clean_data[cls]
        Kmean_clean = KMeans(n_clusters=1)
        labels_clean = Kmean_clean.fit_predict(predict_class)

        axs[idx//10, idx % 10].scatter(predict_class[labels_clean == 0, 0],
                                       predict_class[labels_clean == 0, 1], s=100, c='pink', label='Cluster 1')
        axs[idx//10, idx % 10].scatter(Kmean_clean.cluster_centers_[
                                       :, 0], Kmean_clean.cluster_centers_[:, 1], s=300, c='blue', label='Centroids')
        axs[idx//10, idx % 10].plot(x, y1, '-r', label='y=x-1', c='green')
        axs[idx//10, idx % 10].plot(x, y2, '-r', label='y=x+1', c='green')
        axs[idx//10, idx %
            10].set_title(f'Class {cls} - {len(clean_data[cls])}')

    # plt.show()
    plt.savefig('Clean Data Distribution.png')
    return clean_files, clean_data
