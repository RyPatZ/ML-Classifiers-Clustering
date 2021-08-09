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
    trainFake, testFake = make_classification(n_features=20, n_redundant=0, n_informative=5,
                                              random_state=1, n_clusters_per_class=1)
    trainFake += 4.0 * np.random.uniform(size=trainFake.shape)
    myfakedataset = (trainFake, testFake)

    bankN = pd.read_csv("banknotes.csv")
    bankNY = bankN[bankN.columns[4]].values.tolist()
    del bankN['Class']
    bankNX = bankN.values.tolist()
    bankN = (bankNX, bankNY)

    lonosphere = pd.read_csv("Ionosphere.csv")
    lonosphereY = lonosphere[lonosphere.columns[34]].values.tolist()
    del lonosphere['class']
    lonosphereX = lonosphere.values.tolist()
    lonosphere = (lonosphereX, lonosphereY)

    steel_P = pd.read_csv("steel-plates-fault.csv")
    steel_PY = steel_P[steel_P.columns[33]].values.tolist()
    del steel_P['Class']
    steel_PX = steel_P.values.tolist()
    steel_P = (steel_PX, steel_PY)

    datasets = [bankN, lonosphere, steel_P, myfakedataset]
    setsNames = ["bank note dataset:", "lonosphere dataset:", "steel_P dataset:", "myfakedataset"]

    i = 1
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
        features = StandardScaler().fit_transform(features)
        classfList1 = []
        classfList2 = []
        classfList3 = []
        values=[1e-9,1e-5,1e-1]
        for v in range(len(values)):
            classf = GaussianNB(var_smoothing=values[v])
            for count in range(300):
                train_feat, test_feat, train_target, test_target = train_test_split(features, target, test_size=.5)
                classf.fit(train_feat, train_target)
                score = classf.score(test_feat, test_target)
                if(v==0):
                    classfList1.append(score)
                if (v ==1):
                    classfList2.append(score)
                if (v ==2):
                    classfList3.append(score)
        fig, ax1 = plt.subplots()
        title = setsNames[ds_cnt],"GaussianNB"
        plt.title(title)
        FinalList=[classfList1,classfList2,classfList3]
        bp1=ax1.boxplot(FinalList,labels=values)

        # calculate the highest average value
        HighestAverage = 0
        HighestAValue = 0

        for c, list in enumerate(FinalList):
            average = statistics.mean(list)
            if average>HighestAverage:
                HighestAverage=average
                HighestAValue=values[c]

        print("GaussianNB of",setsNames[ds_cnt])
        print("The best average value of",setsNames[ds_cnt],"is" ,HighestAverage,4)
        print("The highest value for the control parameter is",HighestAValue)

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

