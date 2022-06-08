from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import figure
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import json
from user_generator import new_users_json


def read_users(userJson):
    users = json.loads(userJson)
    X = [d[('x')] for d in users]
    Y = [d[('y')] for d in users]
    weights = [d[('weight')] for d in users]
    weights = np.array(weights)
    X = np.array(list(zip(X, Y)))
    return X, weights


def generate_users():
    sample_number = 200
    X, y_true = make_blobs(n_samples=sample_number,
                           center_box=(-5, 5), n_features=3)
    weights = X[:, 2]
    weights = abs(weights)
    np.random.shuffle(weights)
    X = X[:, :2]
    figure(figsize=(20, 20), dpi=80)
    plt.scatter(X[:, 0], X[:, 1], s=weights*10)
    return X, weights


# Loop over all clusters and find index of closest point to the cluster center and append to closest_pt_idx list.
def find_radius(X, kmeans, weights):
    l = kmeans.n_clusters
    distances = np.empty(l)
    total_traffic = np.empty(l)
    base_users = []
    for iclust in range(kmeans.n_clusters):
        # get all points assigned to each cluster:
        cluster_pts = X[kmeans.labels_ == iclust]

        # for index, c in enumerate(cluster_pts):
        #     user = np.where(c == X)
        #     print(user[0])
        #     base_users[index] = user[0][0]

        # get all indices of points assigned to this cluster:
        cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]
        base_users.append(cluster_pts_indices)
        cluster_cen = kmeans.cluster_centers_[iclust]
        max_idx = np.argmax([euclidean(X[idx], cluster_cen)
                            for idx in cluster_pts_indices])
        distance = euclidean(cluster_pts[max_idx], cluster_cen)

        # Testing:
#         print('farthest point to cluster center: ', cluster_pts[max_idx])
#         print('farthest index of point to cluster center: ', cluster_pts_indices[max_idx])
#         print('longest distance: ', distance)
        distances[iclust] = distance

        cluster_weights = weights[kmeans.labels_ == iclust]
        total_traffic[iclust] = np.sum(cluster_weights)
    # print(base_users)
    return distances, total_traffic, base_users


def cluster(X, weights, cluster_num):

    kmeans = KMeans(n_clusters=int(cluster_num))
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X, sample_weight=weights)

    centers = kmeans.cluster_centers_

    distances, total_traffic, base_users = find_radius(X, kmeans, weights)
    inertia = kmeans.inertia_
    return y_kmeans, centers, distances, inertia, total_traffic, base_users


def base_station_generator(users, max_capacity, max_max_load, max_avg_load):
    base_stations = []
    for j in range(1, 100, 1):
        X, weights = read_users(users)
        y_kmeans, centers, distances, inertia, total_traffic, base_users = cluster(
            X, weights, int(j))

        print("Sum of Squared Distances for k=", j, ": ", inertia)
        print("Mean Distance for k=", j, ": ", np.mean(distances))
        print("Max Distance for k=", j, ": ", distances.max)
        print("Total Max Traffic for k=", j, ": ", total_traffic.max)
        print()
        avg_load = 0
        for t in total_traffic:
            avg_load = avg_load + (t * 100) / max_capacity
        avg_load = avg_load / total_traffic.size
        if((np.mean(distances) < 2) and (np.amax(distances) < 2) and (inertia < 0.0012) and (np.amax(total_traffic) < max_capacity) and ((np.amax(total_traffic) * 100) / max_capacity) < max_max_load and avg_load < max_avg_load):
            print("k: ", j)
            print()
            fig, ax = plt.subplots(figsize=(25, 25))

            ax.scatter(X[:, 0], X[:, 1], c=y_kmeans,
                       s=weights*10, cmap='Pastel1')
            ax.scatter(centers[:, 0], centers[:, 1],
                       c='black', s=200, alpha=0.5)

            for c_number, (center, distance, traffic, users) in enumerate(zip(centers, distances, total_traffic, base_users)):

                users = [str(u) for u in users]
                users = " ".join(users)
                base = {'baseID': c_number,
                        'x': center[0],
                        'y': center[1],
                        'distance': distance,
                        'capacity': max_capacity,
                        'traffic': traffic,
                        'base_users': users
                        }

                base_stations.append(base)

                print("cluster: ", c_number)
                print("center: ", center)
                print("radius: ", distance)
                print()
                ax.annotate(c_number, center)
                circle = mpl.patches.Circle(
                    center, distance, color="black", fill=False)
                ax.add_patch(circle)

            # plt.show()
            plt.savefig(str(len(users)) + "-" + str(j))

            # cx = centers[:, 0]
            # cy = centers[:, 1]
            # keys = np.array2string(np.arange(0, c_number))
            # return json.dumps(dict((z[0], list(z[1:])) for z in zip(keys, cx, cy, distances)))

            # json_str = json.dumps(base_stations)
            # with open('base_data.txt', 'w') as outfile:
            #     outfile.write(json_str)
            # return

            return json.dumps(base_stations)


# print(base_station_generator('[{"_index":0,"x":41.05274800055385,"y":28.951327050358923,"weight":72.0,"userID":"Usr-A00"},{"_index":1,"x":41.05094024657005,"y":28.95393095603621,"weight":90.0,"userID":"Usr-A01"},{"_index":2,"x":41.034172570749895,"y":28.9815821623627,"weight":12.0,"userID":"Usr-A02"},{"_index":3,"x":41.034386532370064,"y":28.983662906741742,"weight":77.0,"userID":"Usr-A03"},{"_index":4,"x":41.048419739031154,"y":28.953409555865615,"weight":63.0,"userID":"Usr-A04"},{"_index":5,"x":41.050533671297025,"y":28.9447935161235,"weight":19.0,"userID":"Usr-A05"},{"_index":6,"x":41.02833899642551,"y":28.972870147239185,"weight":67.0,"userID":"Usr-A06"},{"_index":7,"x":41.03897167855691,"y":28.961485187094834,"weight":79.0,"userID":"Usr-A07"},{"_index":8,"x":41.031482125734804,"y":28.963425268629674,"weight":75.0,"userID":"Usr-A08"},{"_index":9,"x":41.04844967075572,"y":28.943949438446005,"weight":87.0,"userID":"Usr-A09"}]'))
# print(base_station_generator(new_users_json(30, True, "ITU")))
# base_station_generator(new_users_json(50, True, "ITU"))
