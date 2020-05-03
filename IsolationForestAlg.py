# Timothy Kudryn, cs422, Mr. Peterson
# unsupervised machine learning algorithm that finds outliers in a data set
# This particular project runs a Isolation Forest algorithm
# on two seperate data sets that document credit card transactions

# I also included code for the less effective Local Outlier Factor algorithm that I tried implementing

# importing libraries that will be used in my k-means model3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Reading in the data using the pandas library
data = pd.read_csv("creditcard_copy.csv")
data2 = pd.read_csv("application_data.csv")

###READ ME###
# All the steps for data set 1 and data set 2 were done side to side to help
# Users understnad each step better.
# *************PREPROCESSING**************
# visualize the data columns
# print(data.columns)
# print(data.shape) tells you how many rows and columns exist
# matplotlib and numpy also offer great visualization features. (will use them below)

# spliting the data into 15% of the total data due to time constraints
splitData = data.sample(frac=.25, random_state=1)
splitData2 = data2.sample(frac=.00033, random_state=1)

# to visualize the size of the data
# print("data1: ", splitData.shape)
# print("data2: ", splitData2.shape)

# plotting histogram of all the given parameters to help visualize the data
# histograms are easily created using matplotlib
splitData.hist(figsize=(18, 20))
plt.savefig("credCard_data1_hist.pdf")
plt.clf()

splitData2.hist(figsize=(18, 20))
plt.savefig("credCard_data2_hist.pdf")
plt.clf()

# calculating the fraudulent ratio below

validCases = splitData[splitData['Class'] == 0]
fraudCases = splitData[splitData['Class'] == 1]

# explicit type casting to float to avoid rounding errors
fraudRatio = len(fraudCases) / float(len(validCases) + 1)

validCasesData2 = splitData2[splitData2['TARGET'] == 0]
fraudCasesData2 = splitData2[splitData2['TARGET'] == 1]


# Building a correlation matrix
# correlation matrix tell the the user of the data set whether or not different data variables
# have strong correlation. This matrix will help us find strong linear relations between variables
# and will also let us know if any of the data variables should be removed

correlationMatrix = splitData.corr()

correlationMatrix2 = splitData2.corr()

# plotting via seaborn library
plt.cla()  # clears previous plot. Part of the matplotlib library
correlationMatrixPlot = plt.figure(figsize=(15, 10))
sns.heatmap(correlationMatrix, vmax=.9, square=True)
plt.savefig("data1_correlationHeatMap.pdf")
# plt.show()
plt.clf()

correlationMatrix2Plot = plt.figure(figsize=(15, 10))
sns.heatmap(correlationMatrix2, vmax=.9, square=True)
plt.savefig("data2_correlationHeatMap.pdf")
plt.clf()

# splitting the data to be easier to work with
splitDataColumns = splitData.columns.tolist()

splitData2Columns = splitData2.columns.tolist()

# Filtering out missingness
splitDataColumns = [c for c in splitDataColumns if c not in ["Class"]]
splitData2Columns = [c for c in splitData2Columns if c not in ["TARGET"]]

fraudCol = "Class"

splitDataColumnsXvals = splitData[splitDataColumns]
splitDataColumnsYvals = splitData[fraudCol]

fraudCol2 = "TARGET"

splitData2ColumnsXvals = splitData2[splitData2Columns]
splitData2ColumnsYvals = splitData2[fraudCol2]
a = 0
# print(splitDataColumnsXvals.shape)
# print(splitDataColumnsYvals.shape)


# ********** Algorithm Implementation ********

# Defintion the outlier detection methods
# TESTING first data set
state = 1
####CHANGE NAMES OF DICT
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(splitDataColumnsXvals),
                                        contamination=fraudRatio,
                                        random_state=state)
    # "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
    #                                           contamination=fraudRatio)
}

numOutliers = len(fraudCases)
# Iterating through Classifiers fit the model. Making use of the fit() function
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fitting data and tagging outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(splitDataColumnsXvals)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(splitDataColumnsXvals)
        scores_pred = clf.decision_function(splitDataColumnsXvals)
        y_pred = clf.predict(splitDataColumnsXvals)
    # y_pred will be 0 : inlier, 1 : outlier after reshaping occurs
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    num_errors = (y_pred != splitDataColumnsYvals).sum()

    # classification metrics
    print("test 1", clf_name)
    print('{}: {}'.format(clf_name, num_errors), " errors ")
    print(accuracy_score(splitDataColumnsYvals, y_pred))  # terrible for fp and fn
    print(classification_report(splitDataColumnsYvals, y_pred))

# TESTING Second dataset
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fitting data and tagging outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(splitData2ColumnsXvals)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(splitData2ColumnsXvals)
        scores_pred = clf.decision_function(splitData2ColumnsXvals)
        y_pred = clf.predict(splitData2ColumnsXvals)
    # y_pred will be 0 : inlier, 1 : outlier after reshaping occurs
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    num_errors = (y_pred != splitData2ColumnsYvals).sum()

    # classification metrics
    print("test2: ", clf_name)
    print('{}: {}'.format(clf_name, num_errors))
    print(accuracy_score(splitData2ColumnsYvals, y_pred))  # terrible for fp and fn
    print(classification_report(splitData2ColumnsYvals, y_pred))
