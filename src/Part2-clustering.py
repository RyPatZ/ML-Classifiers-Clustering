import statistics

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def main():
    # Use a breakpoint in the code line below to debug your script.
    h = .02
    classifiers = [KNeighborsClassifier(1), GaussianNB, DecisionTreeClassifier,
                   LogisticRegression, GradientBoostingClassifier, RandomForestClassifier, MLPClassifier]
    # names = ["Nearest Neighbors", "GaussianNB", "DecisionTreeClassifier",
    #          "LogisticRegression", "GradientBoostingClassifier", "RandomForestClassifier", "MLPClassifier"]


    bankN0 = pd.read_csv("banknotes.csv")
    bankN = bankN0
    bankNY = bankN[bankN.columns[4]].values.tolist()
    del bankN['Class']
    bankNX = bankN.values.tolist()
    bankN = (bankNX, bankNY)

    iris0 = pd.read_csv("iris.csv")
    iris = iris0
    irisY = iris[iris.columns[4]].values.tolist()
    del iris['class']
    irisX = iris.values.tolist()
    iris = (irisX, irisY)

    datasets = [iris,bankN]
    KNdatasets = [iris0,bankN0]
    setsNames = ["iris dataset","bank note dataset:"]

    i = 1
    j=20
    c = 1
    values = 1
    for ds_cnt, ds in enumerate(datasets):
        values = 1
        features, target = ds
        features = StandardScaler().fit_transform(features)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1,i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        bp=ax.scatter(features[:, 0], features[:, 1], cmap=cm_bright,
                   edgecolors='k')
        i+=1
    plt.show()

    for ds_cnt, ds in enumerate(datasets):
        features, target = ds
        # features = StandardScaler().fit_transform(features)
        df = pd.DataFrame(features)
        Kmeans = KMeans(n_clusters=3)
        Kmeans.fit(df)
        Labels = Kmeans.labels_
        df['Label'] = Labels
        features_with_cluster = df.values.tolist()
        ClusterScoreList =[]
        NoClusterScoreList=[]

        classf = GaussianNB()
        classf_cluster = GaussianNB()

        for count in range(100):
            train_feat, test_feat, train_target, test_target = train_test_split(features, target, test_size=0.95)
            classf.fit(train_feat, train_target)
            NoClusterScore = classf.score(test_feat,test_target)
            NoClusterScoreList.append(NoClusterScore)

            train_feat_clu, test_feat_clu, train_target_clu, test_target_clu = train_test_split(features_with_cluster, target, test_size=0.95)
            classf_cluster.fit(train_feat_clu, train_target_clu)
            WithClusterScore = classf_cluster.score(test_feat_clu, test_target_clu)
            ClusterScoreList.append(WithClusterScore)


        print(NoClusterScoreList)
        print(ClusterScoreList)
        fig, ax1 = plt.subplots()
        title = setsNames[ds_cnt], "Part 2"
        plt.title(title)
        ax1.scatter(NoClusterScoreList,ClusterScoreList)
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

