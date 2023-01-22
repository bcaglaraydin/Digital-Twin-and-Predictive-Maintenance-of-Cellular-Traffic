"""Module for generating base stations"""
import json
from typing import Tuple, List, Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def _read_users(userJson: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read user data from json string"""
    users = json.loads(userJson)
    X = [d[("x")] for d in users]
    Y = [d[("y")] for d in users]
    weights = [d[("weight")] for d in users]
    weights = np.array(weights)
    X = np.array(list(zip(X, Y)))
    return X, weights


def generate_users(sample_number: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample user coordinates with weights, used for testing"""
    X, _ = make_blobs(n_samples=sample_number, center_box=(-5, 5), n_features=3)
    weights = X[:, 2]
    weights = abs(weights)
    np.random.shuffle(weights)
    X = X[:, :2]
    figure(figsize=(20, 20), dpi=80)
    plt.scatter(X[:, 0], X[:, 1], s=weights * 10)
    return X, weights


def _get_base_detail(
    X: np.ndarray, kmeans: KMeans, weights
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Loop over all clusters and find index of closest point to the cluster center
    and append to closest_pt_idx list"""
    l = kmeans.n_clusters
    distances = np.empty(l)
    total_traffic = np.empty(l)
    base_users = []

    for iclust in range(kmeans.n_clusters):

        # get all points assigned to each cluster
        cluster_pts = X[kmeans.labels_ == iclust]

        # get all indices of points assigned to this cluster
        cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]

        base_users.append(cluster_pts_indices)
        cluster_cen = kmeans.cluster_centers_[iclust]
        max_idx = np.argmax(
            [euclidean(X[idx], cluster_cen) for idx in cluster_pts_indices]
        )
        distance = euclidean(cluster_pts[max_idx], cluster_cen)
        distances[iclust] = distance

        cluster_weights = weights[kmeans.labels_ == iclust]
        total_traffic[iclust] = np.sum(cluster_weights)
    return distances, total_traffic, base_users


def _cluster(
    X: np.ndarray, weights: np.ndarray, cluster_num: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, List[np.ndarray]]:
    """Fit users to a kmeans cluster"""
    kmeans = KMeans(n_clusters=int(cluster_num))
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X, sample_weight=weights)

    centers = kmeans.cluster_centers_

    distances, total_traffic, base_users = _get_base_detail(X, kmeans, weights)
    inertia = kmeans.inertia_
    return y_kmeans, centers, distances, inertia, total_traffic, base_users


def _get_avg_load(total_traffic: List[float], max_capacity: float) -> float:
    """Get average load in a cluster"""
    avg_load = 0
    for t in total_traffic:
        avg_load = avg_load + (t * 100) / max_capacity

    return avg_load / total_traffic.size


def _get_base_station_obj(
    users: List[int],
    c_number: float,
    center: List[float],
    distance: float,
    max_capacity: float,
    traffic: float,
) -> Dict:
    """Return a base staion object"""
    users_str = [str(u) for u in users]
    users_str = " ".join(users_str)
    base = {
        "baseID": c_number,
        "x": center[0],
        "y": center[1],
        "distance": distance,
        "capacity": max_capacity,
        "traffic": traffic,
        "base_users": users_str,
    }
    return base


def base_station_generator(
    users: str, max_capacity: float, max_max_load: float, max_avg_load: float
) -> str:
    """Locate proper base station locations according to user data and settings"""
    base_stations = []
    for j in range(1, 100, 1):
        X, weights = _read_users(users)
        y_kmeans, centers, distances, inertia, total_traffic, base_users = _cluster(
            X, weights, int(j)
        )

        print("Sum of Squared Distances for k=", j, ": ", inertia)
        print("Mean Distance for k=", j, ": ", np.mean(distances))
        print("Max Distance for k=", j, ": ", distances.max)
        print("Total Max Traffic for k=", j, ": ", total_traffic.max)
        print()

        avg_load = _get_avg_load(total_traffic, max_capacity)

        if (
            (np.mean(distances) < 2)
            and (np.amax(distances) < 2)
            and (inertia < 0.0012)
            and (np.amax(total_traffic) < max_capacity)
            and ((np.amax(total_traffic) * 100) / max_capacity) < max_max_load
            and avg_load < max_avg_load
        ):
            print("k: ", j)
            print()
            _, ax = plt.subplots(figsize=(25, 25))

            ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=weights * 10, cmap="Pastel1")
            ax.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)

            for c_number, (center, distance, traffic, users) in enumerate(
                zip(centers, distances, total_traffic, base_users)
            ):

                base = _get_base_station_obj(
                    users, c_number, center, distance, max_capacity, traffic
                )

                base_stations.append(base)

                print("cluster: ", c_number)
                print("center: ", center)
                print("radius: ", distance)
                print()
                ax.annotate(c_number, center)
                circle = mpl.patches.Circle(center, distance, color="black", fill=False)
                ax.add_patch(circle)

            # plt.savefig(str(len(users)) + "-" + str(j))

            return json.dumps(base_stations)
    raise RuntimeError("End of iterations!")
