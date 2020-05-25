#!/usr/bin/env python3.7
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from sklearn.cluster import DBSCAN
import predict as p
from flask import Flask,render_template
import json
## exemple d’utilisation avec discretisation et histogrammes
# df,bins=discretise_dataset('mnist_64_32_16_/mnist_l1_64_l2_32_l3_16_.csv',3)
# df is a pandas_core_frame_DataFrame 
# df_X,df_y = pandas_core_frame_DataFrame_to_list(df)
# X_1 = [df_X[i] for i in range(len(df_X)) if df_y[i]=='1']
# X_0 = [df_X[i] for i in range(len(df_X)) if df_y[i]=='0']

# enc_sig_val = encrypting_signature_value(df_X)

# encrypting_X_0 = X_to_encrypted_X(X_0,enc_sig_val)
# encrypting_X_1 = X_to_encrypted_X(X_1,enc_sig_val)
# print(encrypting_X_0)
# Histogram(encrypting_X_1,"mnist1.png")
# Histogram(encrypting_X_0,"mnist0.png")

# Load Data:
layers, y = makes_Layers("mnist_64_32_16_/mnist_l1_64_l2_32_l3_16_.csv")
VTlayers, VTy = makes_Layers("VTmnist_64_32_16_/VTmnist_l1_64_l2_32_l3_16_.csv")

# mat_dist1 = matrice_distances(mnlayer) #layer1 -> mnlayer1

######### DBSCAN

# clusters = clustering(layers,False)

# print(clusters[2].labels_)
# print(clusters[1].labels_)
# print(clusters[0].labels_)
# print(p.dbscan_predict(clusters[2],strTolist(layer3)))

########## KMEANS
## mnist

clusters,models = p.kmModel(layers,5)
pourcentages_mnist = pourcentages(clusters,y)
# print(pourcentages_mnist)
clusters_layers = elimination(pourcentages_mnist,10)
tab,nodes = signatures_clusters(clusters,clusters_layers,y)
# plot2D_on_all_layers("mnist",layers,clusters,y)

# ## VTmnist

VTclusters= p.kmPredict(VTlayers,models)
VTpourcentages_mnist = pourcentages(VTclusters,VTy)
VTclusters_layers = elimination(VTpourcentages_mnist,5)

VT_tab,VT_nodes = signatures_clusters(VTclusters,VTclusters_layers,VTy,VT=True)
# plot2D_on_all_layers("VTmnist",VTlayers,VTclusters,VTy)

# # Generating JSON file
with open ("static/mnist.json","w") as f:
	json.dump({"links":tab,"nodes":nodes},f)
with open ("static/VTmnist.json","w") as f:
 	json.dump({"links":VT_tab,"nodes":VT_nodes},f)

# # Lunching Flask server
app = Flask(__name__)
app.static_url_path='/static'

@app.route("/")
def home():
	return render_template("index.html")
if __name__ == "__main__":
	app.run(debug=True)