import sys
import sklearn.metrics as mt
import numpy as np
import math
import mysymnmf as symn
import kmeans

np.random.seed(1234)
MAX_ITER = 300

def get_pnt_lst_from_file(filename):
    """
    saves the data from the file as a list of lists (2D matrix)
    :param filename: the name of the file containing the data
    :return: 2D matrix of the point values
    """
    result = []
    with open(filename, 'r') as file:
        for line in file:
            coordinates = line.strip().split(',')
            point = [float(coo) for coo in coordinates]
            result.append(point)
    return result


def average_of_mat(mat):
    """
    calculates the average of all matrix enterances
    :param mat: a matrix to calculate its average
    :return: returns a entry-wise average of a 2D matrix
    """
    entries_num = len(mat[0]) * len(mat)
    count = 0
    for row in range(len(mat)):
        for col in range(len(mat[0])):
            count += mat[row][col]
    avg = count / entries_num if entries_num != 0 else 0
    return avg


def create_H_mat(k, n, m):
    """
    builds the H matrix ( n * m matrix )
    :param k: - number of dimensions
    :param n: - number of dimensions
    :param m: - mean of the W matrix
    :return uniformally generated H matrix
    """
    H = []
    C = 2 * (math.sqrt(m / k))
    for i in range(n):
        row = []
        for j in range(k):
            row.append(np.random.uniform(0, C))
        H.append(row)
    return H


def create_X_mat(file):
    """
    turns the data from the file to a list of lists
    :param file: .txt filepath to a file containing N R^m euclidian dots
    :return: the X matrix
    """
    x = []
    with open(file) as fp:
        for line in fp:
            line = [float(x) for x in line.split(",")]
            x.append(line)
    return x


def call_symnmf(X, K):
    """
    a wrapper function from symnmf requests not from __main__
    :param X: initial data
    :param K: amount of clusters
    :return: returns H matrix of correlation betwen each center and each vector
    """
    W = symn.norm(X)
    H = create_H_mat(K,n,average_of_mat(W))
    res = symn.symnmf(H,W,K,n)
    return res


def comparison(X,sym_clusters , kmeans_clusters):
    """
    uses the silhouette_score from sklearn.metrics to compare the scores of kmeans and symnmf 
    :param X: the n * d original data
    :param sym_clusters: the clusters achieved by symnmf algorithm
    :param kmeans_clusters: the clusters achieved by K-means algorith
    :return: tuple of silhouette factor of symnmf and k-means respectively
    """
    sym_res = mt.silhouette_score(X,sym_clusters)
    kmeans_res = mt.silhouette_score(X,kmeans_clusters)
    return sym_res, kmeans_res


def get_classification(X,centers):
    """
    gets the group that each vector in X belongs to according to Kmeans
    :param X: the data given by the user
    :param centers: the centers yielded by the Kmeans algorithms
    :return: vector labeling each vector to its center.
    """
    distances = np.zeros((len(X),len(centers)))
    for vector in range(len(X)):
        for center in range(len(centers)):
            dis = np.linalg.norm(np.asmatrix(X[vector]) - np.asmatrix(centers[center]))
            distances[vector][center] = dis
    return distances.argmin(axis=1)

def err_printer():
    """"
    a general handler function for errors
    """
    print("An Error Has Occurred")
    exit()


if __name__ == "__main__":
    """
    reads the input arguments and checks if they were read valid
    then calls the functions to calculate the kmeans and symnmf values,
    and compares them with silhouette score
    :param X: the data given by the user
    :param centers: the centers returened from the running of the kmeans algorithms
    :return: print the comparison of scores between kmeans and symnmf
    """
    try:
        args = sys.argv[1:]
        file = args[1]
        K = int(args[0])
        X = create_X_mat(file)
        d = len(X[0])
        n = len(X)
    except:
        err_printer()
    pnt_lst = get_pnt_lst_from_file(file)
    symnmf_clusters = call_symnmf(X,K)
    kmeans_clusters = kmeans.k_means(K,MAX_ITER,pnt_lst)
    kmeans_classification = get_classification(X,kmeans_clusters)
    symnmf_classification = np.argmax(symnmf_clusters,axis=1)
    symnmf_res , kmeans_res = comparison(X,symnmf_classification,kmeans_classification)
    print(f"nmf: {symnmf_res:.4f}\nkmeans: {kmeans_res:.4f}")

