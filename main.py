import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA

from sklearn.metrics.cluster import rand_score

import matplotlib.pyplot as plt

def createX(fileName: str):
    f = open(fileName, "r")
    xValues = []
    xPred = []
    for line in f:
        arr = line.split("\t")
        x = []
        for i in range(1, len(arr) - 1):
            value = float(arr[i])
            if i == 1:
                xPred.append(value)
            else:
                x.append(value)
        xValues.append(x)
    f.close()
    return np.array(xValues), np.array(xPred)

def createKMeansModel(clusterAmount: int):
    model = KMeans(clusterAmount)
    return model

def createAggWardModel(clusterAmount: int):
    model = AgglomerativeClustering(clusterAmount, linkage="ward")
    return model

def createAggAverageModel(clusterAmount: int):
    model = AgglomerativeClustering(clusterAmount, linkage="average")
    return model

def createDensityModel(eps):
    model = DBSCAN(eps)
    return model

def fitPredictModel(model, X):
    return model.fit_predict(X)

def plot(title, labels, X):
    simplifier = PCA(2)
    simplifiedData = simplifier.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(simplifiedData[:, 0], simplifiedData[:, 1], c=labels, cmap='tab10', s=50)
    plt.title(title)
    plt.grid(True)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()

def computeOnData(fileName: str, clusterAmount: int, eps: int):
    X, predictedLabels = createX(fileName)
    
    kmeansModel = createKMeansModel(clusterAmount)
    trueKMeansLabels = fitPredictModel(kmeansModel, X)
    print(f"Rand Index, KMeans, {fileName}: {rand_score(trueKMeansLabels, predictedLabels)}")
    plot(f"KMeans Clustering {fileName}", trueKMeansLabels, X)
    
    aggWardModel = createAggWardModel(clusterAmount)
    trueAggWardLabels = fitPredictModel(aggWardModel, X)
    print(f"Rand Index, Agglomerate Ward Linkage, {fileName}: {rand_score(trueAggWardLabels, predictedLabels)}")
    plot(f"Agg Ward Clustering {fileName}", trueAggWardLabels, X)

    aggAverageModel = createAggAverageModel(clusterAmount)
    trueAggAverageLabels = fitPredictModel(aggAverageModel, X)
    print(f"Rand Index, Agglomerate Average Linkage, {fileName}: {rand_score(trueAggAverageLabels, predictedLabels)}")
    plot(f"Agg Avg Clustering {fileName}", trueAggAverageLabels, X)
    
    densityModel = createDensityModel(eps)
    trueDensityLabels = fitPredictModel(densityModel, X)
    print(f"Rand Index, Density-based, {fileName}: {rand_score(trueDensityLabels, predictedLabels)}")
    plot(f"Density Clustering {fileName}", trueDensityLabels, X)

computeOnData("cho.txt", 5, 1.02)
computeOnData("iyer.txt", 10, 1)