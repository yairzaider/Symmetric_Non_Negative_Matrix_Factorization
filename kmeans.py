import sys

def check_if_whole(str):
    """
    Checks if the given string represents a whole number.
    :param str: Input string to be checked
    :return: The whole number as an integer if valid, otherwise None
    """
    try:
        number = float(str)
        if number.is_integer():
            return int(number)
        else:
            return None
    except ValueError:
        return None
    

def euc_dis(x, y):
    """
    Calculates the Euclidean distance between two points.
    :param x: First point as a list of coordinates
    :param y: Second point as a list of coordinates
    :return: Euclidean distance as a float
    """
    dist = 0.0
    dim = len(x)
    for i in range(dim):
        dist += ((x[i] - y[i]) ** 2)
    return dist ** 0.5


def get_closest_cent(pnt, k_centroids):
    """
    Finds the index of the closest centroid to a given pants.
    :param pnt: The point for which the closest centroid is to be found
    :param k_centroids: List of centroids
    :return: Index of the closest centroid
    """
    min_cent = 0
    min_dis = euc_dis(pnt, k_centroids[0])
    for k in range(len(k_centroids)):
        curr_dis = euc_dis(pnt, k_centroids[k])
        if curr_dis < min_dis:
            min_cent = k
            min_dis = curr_dis
    return min_cent


def k_means(k, iter, pnt_lst):
    """
    Performs K-means clustering on a list of points.
    :param k: Number of clusters
    :param iter: Maximum number of iterations
    :param pnt_lst: List of points to be clustered
    :return: List of updated centroids after clustering
    """
    k_centroids = []
    pnt_closest_cent = [0] * len(pnt_lst)
    for i in range(k):
        k_centroids.append(pnt_lst[i])
    epsilon_counter = 0
    while iter > 0:
        for pnt_index in range(len(pnt_lst)):
            pnt_closest_cent[pnt_index] = get_closest_cent(pnt_lst[pnt_index], k_centroids) 
        epsilon_counter = 0
        for cent_index in range(len(k_centroids)):
            accu = [0.0] * len(k_centroids[0])
            size_of_cluster = 0
            for i in range(len(pnt_closest_cent)):
                if pnt_closest_cent[i] == cent_index:
                    size_of_cluster += 1
                    for j in range(len(k_centroids[0])):
                        accu[j] += pnt_lst[i][j]
            if size_of_cluster > 0:
                accu = [coo / size_of_cluster for coo in accu]
            if euc_dis(accu, k_centroids[cent_index]) < 0.0001:
                epsilon_counter += 1
            k_centroids[cent_index] = accu
        if epsilon_counter == k:
            return k_centroids
        iter -= 1
    return k_centroids
