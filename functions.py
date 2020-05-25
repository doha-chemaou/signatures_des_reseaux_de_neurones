#!/usr/bin/env python3.7
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib
from sklearn.decomposition import PCA
import random as rd

#### permet de sauvegarder également un fichier qui contient tous les layers
def get_directory_layers_from_csv(filename):
    tokens=filename.split("_")
    df = pd.read_csv(filename, sep = ',', header = None) 

    
    # creation d'un répertoire pour sauver tous les fichiers
    repertoire=filename[0:-4]
    os.makedirs(repertoire, exist_ok=True)
    string = repertoire+'/'+tokens[0]+'_'
    f=[]
    filenames=[]
    for nb_tokens in range (1,len(tokens)-1):
        name_file=string+'l'+str(nb_tokens)+'_'+tokens[nb_tokens]+'.csv'
        f.append(open(name_file, "w"))
        filenames.append(name_file)
        
        
    # sauvegarde du dataframe dans une chaîne de caracteres
    ch = df.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
    vals = [','.join(ele.split()) for ele in ch]
    
    # sauvegarde dans des fichiers spécifiques par layer
    token_layer=[]
    token_exemples=[]
    for nb_exemples in range (len(vals)):
        deb=str(df[0][nb_exemples])+','
        # 1 ligne correspond à une chaine
        s=vals[nb_exemples]
        listoftokens=re.findall(r'<b>,(.+?),</b>', s)
        nb_layers=len(listoftokens)
        
        for nb_token in range (nb_layers):
            save_token=''
            save_token=deb+str(listoftokens[nb_token])+'\n'
            
            f[nb_token].write(save_token)

    # sauvegarde d'un fichier qui contient tous les layers en une fois
    # récupération des données pour enlever les <b> et </b>
    df_all=pd.DataFrame()
    myindex=0
    for nb_columns in range(df.shape[1]):
        df[nb_columns]=df[nb_columns].astype(str)
        if (df[nb_columns]!='<b>').all() and (df[nb_columns]!='</b>').all():
            df_all[myindex]=df[nb_columns]
            myindex+=1
    print (df_all.head())
    #cols = [1,2,4,5,12]
    #df_bof=df_all.drop(df_all.columns[cols],axis=1)
    #df_all.drop(df_all.columns[0], axis=1,inplace=True)
    #print (df_all.head())
    # construction du nom du fichier de sauvegarde
    string = repertoire+'/'+tokens[0]+'_'
    for nb_tokens in range (1,len(tokens)-1):
        string+='l'+str(nb_tokens)+'_'+tokens[nb_tokens]+'_'
    string+='.csv'       
    # sauvegarde en .csv
    df_all.to_csv(string, sep=',', encoding='utf-8',index=False)
##### la fonction suivante permet de discretiser en fonction d’une valeur de bin passée en paramètre
def discretise_dataset(filename,bins):
    df = pd.read_csv(filename, sep = ',', header = None) 
    oneColumn = np.array(df[1])
    for i in range(2,df.shape[1]):
        oneColumn=np.append(oneColumn,np.array(df[i]))
    dfoneColumn=pd.DataFrame(oneColumn)
    nb_bins=bins
    dftemp=pd.DataFrame()
    dftemp=dfoneColumn[0]
    # dftemp[0],retbins=pd.cut(dfoneColumn[0], bins=nb_bins, labels=np.arange(nb_bins), right=False,retbins=True)
    # print(retbins)
    df_new=pd.DataFrame(df[0])
    nb_tuples=df.shape[0]
    j=0
    for i in range(1,df.shape[1]):
        df_new[i]=np.copy(dftemp[0][j:j+nb_tuples])
        j+=nb_tuples
    print(df_new)
    return df_new

## Read the signatures from the file
def readXy(filename):
    f = open(filename, "r")
    matrice = f.read().split('\n')
    y = []
    X = []
    matrice = matrice[1:]
    for i in range(len(matrice)-1) :
        tab = matrice[i].split(',')
        y.append((int)(tab[0]))
        # X.append(list(np.array(tab[1:]).astype("float32")))
        X.append(list(np.array(tab[1:]).astype("float32")))
    return X,y

## Drawing and saving the Histogram
def Histogram(X,histname):
    fig = plt.hist(X)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(histname)
    plt.clf()
''' ## NO LONGER USED ##: Intermediate function for encrypting string signature
def encrypting_signature_value(X_) : # param X_ est X_1_0
    listOfX_ = []
    encrypting_X_ = {}
    value = 1
    for x in range(len(X_)) :
        if not (X_[x] in listOfX_) : 
            listOfX_.append(X_[x])
            encrypting_X_[str(X_[x])] = str(value)
            value += 1
    #print(listOfX_)
    return encrypting_X_ # returns a dict of 'signature : value'

## NO LONGER USED ##: Encrypting String signature 
def X_to_encrypted_X(X_,encrypting_signature_value) : # X_ : signatures of a certain class , encrypting_signature_value : the global variable in this code
    listOfX = []
    for x in range(len(X_)) :
        #if not(encrypting_signature_value[str(X_[x])] in listOfX) :
        listOfX.append(encrypting_signature_value[str(X_[x])])
    return listOfX #returns the X_ crypted
'''

## Df to List
def pandas_core_frame_DataFrame_to_list(df) :
    X = []
    y_ = []
    for y in range(len(df)) :
        x = ""
        y_.append(str(df[0][y]))
        for z in range(1,len(df.columns)) :

            x += str(df[z][y])
        X.append(x)
    return X,y_

''' ## NO LONGER USED## : Generating Histograms for Iris Data
def hists_files(file,bins) : # "iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv" should be the file in param
    for x in range(len(bins)) : 
        df,b=discretise_dataset("iris_8_10_8_/iris_l1_8_l2_10_l3_8_.csv",bins[x]) 
        
        name_of_pngHist_class0 = "X_0_disc" + str(bins[x]) + ".png"
        name_of_pngHist_class1 = "X_1_disc" + str(bins[x]) + ".png"

        X_X , y__ = pandas_core_frame_DataFrame_to_list(df) 
        
        X_1_ = [X_X[i] for i in range(len(X_X)) if y__[i]=='1']
        X_0_ = [X_X[i] for i in range(len(X_X)) if y__[i]=='0']

        encr_sig_val_X_X = encrypting_signature_value(X_X)
        enc_X_1_ = X_to_encrypted_X(X_1_,encr_sig_val_X_X)
        enc_X_0_ = X_to_encrypted_X(X_0_,encr_sig_val_X_X)

        Histogram(enc_X_1_,name_of_pngHist_class1)
        Histogram(enc_X_0_,name_of_pngHist_class0)
'''
      
## Read the file and generate a list of signatures for each layer
def makes_Layers(filename) :
    name = filename.split("/")
    name = name[-1].split("_")[1:-1]
    disc_X,y = readXy(filename)
    layers = []
    i=1
    while i<len(name):
        layer = []
        if i == 1:
            ma = int(name[i])
            for X in disc_X:
                layer.append(X[:ma])
        else :
            mi = ma
            ma += int(name[i])
            for X in disc_X:
                layer.append(X[mi:ma])
        layers.append(layer)
        i+=2
    return layers,y

''' ## NO LONGER USED ##: Levenshtein distance
def distance (sig1,sig2) : # special sig1 == sig2
    dist = [[0 for x in range(len(sig1))] for x in range(len(sig2))]
    for x in range(len(sig1)) :
        if x == 0 :
            if sig1[x] == sig2[x] : dist[x][x] = 0
            else : dist[x][x] = 1
        else :
            c = 0 
            while c < x :
                if sig1[x] == sig2[c] : 
                    if c == 0 :
                        dist[c][x] = x
                    else : 
                        dist[c][x] = dist[c-1][x-1] 
                else :
                    if c == 0 :
                        dist[c][x] = min(x , x+1 , dist[c][x-1]) + 1 
                    else :
                        dist[c][x] = min(dist[c-1][x],dist[c][x-1],dist[c-1][x-1]) + 1
                if sig1[c] == sig2[x] : 
                    if c == 0 :
                        dist[x][c] = x
                    else : 
                        dist[x][c] = dist[x-1][c-1]
                else : 
                    if c == 0 :
                        dist[x][c] = min(x , x+1 , dist[x-1][c]) + 1 
                    else :
                        dist[x][c] = min(dist[x-1][c],dist[x][c-1],dist[x-1][c-1]) + 1
                c += 1
            if (sig1[x] == sig2[x]) :
                dist[x][x] = dist[x-1][x-1]
            else : 
                dist[x][x] = min(dist[x][x-1],dist[x-1][x],dist[x-1][x-1]) + 1
    return dist[x][x]

## NO LONGER USED ##: Matrix of distances
def matrice_distances(layer) :
    matrice = []
    for x in range(len(layer)) : 
        mat =[]
        for y in range(len(layer)) :
            mat.append(distance(layer[x],layer[y]))
        matrice.append(mat)
    return matrice

'''
## Percentage of every cluster in one layer
def pourcentages_inter (clusters_of_layer,y) :
    y_prime = list(set(y))
    clust_dic = {}
    clusters = {}
    for i in range(len(y)) :
        if str(y[i]) not in clust_dic:
            clust_dic[str(y[i])] = {}

        if str(clusters_of_layer[i]) in clust_dic[str(y[i])] : 
            clust_dic[str(y[i])][str(clusters_of_layer[i])] += 1
        else : 
            clust_dic[str(y[i])][str(clusters_of_layer[i])] = 1
          
        if str(clusters_of_layer[i]) in clusters : clusters[str(clusters_of_layer[i])] += 1
        else : clusters[str(clusters_of_layer[i])] = 1
    for key in clusters : 
        for i in y_prime:
            if key in clust_dic[str(i)]:
                clust_dic[str(i)][key] = round(clust_dic[str(i)][key]/clusters[key] *100,3)
    return clust_dic

## pourcentages_inter applied on all layers     
def pourcentages(clusters , y) :
    res = []
    for x in range(len(clusters)) :
        res.append(pourcentages_inter(clusters[x],y))
    return res

## removing clusters with a percentage lowen than threshold	
def elimination(pourcentages,threshold) :
    clust_dic = {}
    for i in range(len(pourcentages)):
        for key in pourcentages[i]:
            if key not in clust_dic:
                clust_dic[key] = []
            clusters = []
            for clust in pourcentages[i][key]:
                if pourcentages[i][key][clust]>=threshold:
                    clusters.append(clust)
            clust_dic[key].append(clusters)

    return clust_dic

## identify the clusters that belong to many classes
def shared_clusters(clusters_classes) :
    shared_clust = []
    clust_class = []
    classes = list(clusters_classes.keys())
    for i in range(len(clusters_classes[classes[0]])):
        shared_clust.append(list([]))
    for i in range(0,len(classes)-1) :
        for k in range(len(clusters_classes[classes[i]])) :
            '''shared_clust_layer = []'''
            for h in range(len(clusters_classes[classes[i]][k])) :
                cluster = clusters_classes[classes[i]][k][h]
                for j in range(i+1,len(classes)) :
                    if((cluster in clusters_classes[classes[j]][k]) and (cluster not in shared_clust[k])):
                        shared_clust[k].append(cluster)
    return shared_clust

# def clusters_class(clusters,pourcentages):

#     for i in range(len(clusters)):
#         clusters[i]=list(set(clusters[i]))
#     for i in range(len(clusters)):
#         for j in range(len(clusters[i])):
#             clusters[i][j]=str(clusters[i][j])
#     clust_tab = []
#     for l in range(len(clusters)):
#         clust_dic = {}
#         for clust in clusters[l]:
#             if clust not in clust_dic:
#                 clust_dic[clust]=["0",0]
#             for cla in pourcentages[l]:
#                 if clust in pourcentages[l][cla]:
#                     if pourcentages[l][cla][clust]>=clust_dic[clust][1]:
#                         clust_dic[clust] = [cla,pourcentages[l][cla][clust]]
#             clust_dic[clust]=clust_dic[clust][0]
#         clust_tab.append(clust_dic)
#     return clust_tab


## Generate the JSON file required by html page
def signatures_clusters(clusters,clusters_classes,y,VT=False) :
    classes = list(clusters_classes.keys())
    shared_c = shared_clusters(clusters_classes)
    s_c_p = []
    s_c = []
    # print(pg_clusters)
    tab = []
    nodes = []
    tab_nodes= []
    nb_layers = len(clusters)
    for i in range(len(y)) :
        if VT :
            s = "X11"
        else :
            s = "X"+str(y[i])
        signature = ""+str(y[i])+","
        for j in range(nb_layers) :
            c = "C"+str(j)+str(clusters[j][i])
            k=0
            numclass = ""
            shared = ""
            if(s[0]=="X"):
                if VT :
                    numclass = "C11"
                else :
                    numclass = "C"+s[1]
                shared = "false"
            else:
                shared = str(str(clusters[j-1][i]) in shared_c[j-1]).lower()
                for m in range(len(classes)):
                    if(s[2:] in clusters_classes[classes[m]][j-1]):
                        numclass = "C"+classes[m]
            if s not in tab_nodes:
                tab_nodes.append(s)
                nodes.append({"name":s,"numclass":numclass,"shared":shared})
            if VT : 
                y_vt = "C11"
            else :
                y_vt = "C"+ str(y[i])
            if VT==False or VT:
                if [s,c,str(y[i])] not in s_c:
                    s_c.append([s,c,str(y[i])])
                    s_c_p.append([s,c,str(y[i]),1])
                else:
                    for e in s_c_p:
                        if e[0]==s and e[1]==c and e[2]==str(y[i]):
                            e[3]+=1
            while k < len(tab):
                if tab[k]["source"]==s and tab[k]["target"] == c:

                    tab[k].update({"source":s,"target":c,"value":str(((int)(tab[k]["value"])+1)),"numclass":y_vt})
                    break
                k+=1
            if k == len(tab):
                tab.append({"source":s,"target":c,"value":str(1),"numclass":y_vt})
            s = c
        c =  str(y[i])
        k=0
        numclass = ""   
        shared = str(str(clusters[j][i]) in shared_c[j]).lower()
        for m in range(len(classes)):
            if(s[2:] in clusters_classes[classes[m]][j]):
                numclass = "C"+classes[m]
        if s not in tab_nodes:
            tab_nodes.append(s)
            nodes.append({"name":s,"numclass":numclass,"shared":shared})
        # if VT : 
        #     y_vt = "C11"
        # else :
        y_vt = "C"+str(y[i])
        while k < len(tab):
            if tab[k]["source"]==s and tab[k]["target"] == c:
                tab[k].update({"source":s,"target":c,"value":str(((int)(tab[k]["value"])+1)),"numclass":y_vt})
                break
            k+=1
        if k == len(tab):
            tab.append({"source":s,"target":c,"value":str(1),"numclass":y_vt})
        signature += '\n'
    for t in classes:
        nodes.append({"name":t,"numclass":"C"+str(t),"shared":"false"})
    if VT==False or VT:
        s_c_p = sorted(s_c_p,key=lambda x:x[3])
        for [s,c,y,p] in s_c_p:
            k=0
            while k<len(tab):
                if tab[k]["source"]==s and tab[k]["target"] == c:
                    tab[k]["numclass"]="C"+y
                    break
                k+=1
    return tab,nodes

## Plot Functions
def sans_doublons(liste) :
    l = []
    for x in range(len(liste)) :
        if not(liste[x] in l) : l.append(liste[x])
    return l

def keeps_clear_colors(colors) : 
    c = []
    c = [colors[3]]
    c += [colors[7]]
    c += colors[9:18]
    c += colors[19:]
    return c
    # a enlever : 0,2,1,4,5,6,8,18

def gives_color_to_cluster(clusters,clear_colors) :
    dictio = {}
    c = []
    sans_doublons_clusters = sans_doublons(clusters)
    for i in range(len(sans_doublons_clusters)) : 
        dictio[str(sans_doublons_clusters[i])] = clear_colors[i]
    for i in clusters : 
        c.append(dictio[str(i)])
    return c

def gives_colors_to_classes(classes_y) :
    c = []
    for i in classes_y :
        if i == 0 : 
            c.append('orange')
        if i == 1 : 
            c.append('black')
    return c 

def line_points(pca,classes_y) :
    coord = pandas_core_frame_DataFrame_to_list(pca)
    x = coord[1]
    y = coord[0]
    x_ = []
    y_ = []
    for i in range(len(coord[0])):
        x_.append(float(x[i]))
        y_.append(float(y[i]))

    min_x_0 = min_x_1 = max(x_)
    max_x_0 = max_x_1 = min(x_)
    min_y = min(y_)
    max_y = max(y_)

    for i in range(len(x)) : 
        if classes_y[i] == 0 :
            min_x_0 = min(min_x_0,x_[i])
            max_x_0 = max(max_x_0,x_[i])
        if classes_y[i] == 1 :
            min_x_1 = min(min_x_1,x_[i])
            max_x_1 = max(max_x_1,x_[i])
    bi_x = min(max(min_x_0,min_x_1),min(max_x_0,max_x_1))
    bs_x = max(max(min_x_0,min_x_1),min(max_x_0,max_x_1))
    return bi_x,bs_x,min_y,max_y

def plot2D(name,layer,clusters_per_layer,classes_y,layer_num,pca_done=False) :
    if pca_done==False:
        pca = PCA(n_components=2) 
        pca.fit(layer) 
        pca_data = pd.DataFrame(pca.transform(layer))
    else: pca_data = pd.DataFrame(layer)
    clear_colors = keeps_clear_colors(list(matplotlib.colors.cnames.keys()))
    col = gives_color_to_cluster(clusters_per_layer,clear_colors)    
    classes_colors = gives_colors_to_classes(classes_y)
    plt.title('Plot des données')
    plt.xlabel("$x$", fontsize=10)
    plt.ylabel("$y$", fontsize=10)
    plt.scatter(pca_data[0],pca_data[1],s=100,c=col,marker='o',edgecolors=classes_colors)
    
    class0 = plt.scatter([] , [], c='white',marker='o',edgecolors='orange')
    class1 = plt.scatter([] , [] , c='white',marker='o',edgecolors='black')
    plt.legend((class0,class1), ("class 0", "class 1"),loc='lower right')

    bi_x,bs_x,min_y,max_y = line_points(pca_data,classes_y)
    x1 = (bi_x + bs_x)/2
    x2 = (bi_x + bs_x)/4
    plt.plot([min(x1,x2),max(x1,x2)],[min_y,max_y],'k');
    plt.savefig(name+'_layer_'+str(layer_num)+'.png')
    plt.clf()

def plot2D_on_all_layers(name,layers,clusters,classes_y) :
    for i in range(len(layers)) :
        plot2D(name,layers[i],clusters[i],classes_y,i)
