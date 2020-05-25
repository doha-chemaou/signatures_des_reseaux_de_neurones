#!/usr/bin/env python3.7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import model_selection
from functions import readXy
from functions import Histogram
from sklearn.metrics import accuracy_score
import scipy as sp
import numpy as np
# X,X_1_0,y = readXy("makemoons_3_10_10_3_/makemoons_l1_3_l2_10_l3_10_l4_3_.csv",False)
# X,X_1_0,y = readXy("iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv",False)
# X,X_1_0,y = readXy("mnist_512_/mnist_l1_512.csv",False)
# train_X, test_X, train_y, test_y = model_selection.train_test_split(X,y,test_size = 0.2, random_state = 0)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(train_X)
#print(len(train_X),len(test_X))
# print(kmeans.predict(test_X),test_y)
# print(accuracy_score(kmeans.predict(test_X),test_y))

def kmModel(layers,nb_clusters):
    models = []
    clusters = []
    for i in range(len(layers)):
        kmeans = KMeans(n_clusters=nb_clusters,random_state=0).fit(layers[i])
        cluster = kmeans.predict(layers[i])
        models.append(kmeans)
        clusters.append(list(cluster))
    return clusters,models

def kmPredict(layers,kmeans):
	clusters = []
	for i in range(len(layers)):
		clusters.append(list(kmeans[i].predict(layers[i])))
	return clusters

def kmAccuracy(predictions,y):
	return accuracy_score(predictions,y)

# cette fonction permet de caculer la discance par rapport à chaque médoid
def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise par défaut (-1 indique noise dans dbscan)
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Boucler sur toutes les entrées pour un label
    for j, x_new in enumerate(X_new):
        # Rechercher l'exemple qui est le plus proche qu'eps
        for i, x_core in enumerate(dbscan_model.components_): 
            if metric(x_new, x_core) < dbscan_model.eps:

                # Affecter le label du plus proche au x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new