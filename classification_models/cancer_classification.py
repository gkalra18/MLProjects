# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:56:53 2017

@author: gkalra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.cross_validation import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, precision_recall_curve, auc


# read from file
data = pd.read_csv(<path to file>)
data.head(5)

#data.info()

# drop columns that are not required in training the model
data = data.drop(['id', 'Unnamed: 32'], axis = 1)
data.head(5)

# plot frequency against class (M or B)
#plot_classes = pd.value_counts(data['diagnosis'], sort = True).sort_index()
#
#plot_classes.plot(kind = 'bar')
#plot.title("Cancer Diagnosis")
#plot.xlabel("Class")
#plot.ylabel("Frequency")

data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

# assign X and y
X = data.ix[:, data.columns != 'diagnosis']
y = data.ix[:, data.columns == 'diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)

# using logistic regression for prediction of labels
def get_KFold_Scores(X_train, y_train):
    kfold = KFold(len(y_train), 5)
    c_param_values = [0.01, 0.1, 1, 10, 100]
    recall_score_table = pd.DataFrame(index = range(len(c_param_values), 2), columns = ['C_Param', 'Mean_Recall_Score'])
    recall_score_table['C_Param'] = c_param_values
    j = 0
    for c_param in c_param_values:
        print("c param: ", c_param)
        recall_scores = []

        for iteration, index in enumerate(kfold, start = 1):
            
            lr = LogisticRegression(C = c_param)
            lr.fit(X_train.iloc[index[0], :], y_train.iloc[index[0], :].values.ravel())
            y_pred = lr.predict(X_train.iloc[index[1], :].values)
            recall_value = recall_score(y_train.iloc[index[1], :].values, y_pred)
            recall_scores.append(recall_value)
            
        recall_score_table.iloc[j, 1] = np.mean(recall_scores)
        print("Mean Recall Score: ", np.mean(recall_scores))
        print("------------------------")
        j += 1
    best_c_param = recall_score_table.loc[recall_score_table['Mean_Recall_Score'].idxmax()]['C_Param']
    return best_c_param
    
# function to plot the confusion matrix
import itertools

def plot_cnf_matrix(cnf, classes, normalize = False, cmap=plot.cm.Blues):
    plot.imshow(cnf, cmap = cmap)
    plot.title("Confusion Matrix")
    plot.colorbar()
    ticks = np.arange(len(classes))
    plot.xticks(ticks, classes, rotation = 0)
    plot.yticks(ticks, classes)
    threshold = cnf.max() / 2
    plot.xlabel("Predicted Class")
    plot.ylabel("Actual Class")
    
    if normalize:
        cnf = cnf.astype('float') / cnf.sum(axis = 1)[:, np.newaxis]
    
    for i, j in itertools.product(range(cnf.shape[0]), range(cnf.shape[1])):
        plot.text(j, i, cnf[i, j], horizontalalignment = 'center', color = 'white' if cnf[i, j] > threshold else 'black')
    
    plot.tight_layout()

final_c_param = get_KFold_Scores(X, y)
print("Best C param: ", final_c_param)

# train the model with the above value for C param
lr = LogisticRegression(C = final_c_param)
lr.fit(X_train, y_train.values.ravel())
y_pred = lr.predict(X_test.values)
 
cnf_matrix = confusion_matrix(y_test, y_pred)
classes = [0,1]
plot_cnf_matrix(cnf_matrix, classes)
  
recall_score = recall_score(y_test, y_pred)
print ("Recall score", recall_score)

# plot roc curve
y_pred_score = lr.fit(X_train, y_train.values.ravel()).decision_function(X_test.values)
fpr, tpr, thresholds = roc_curve(y_test.values.ravel(), y_pred_score)
roc_auc = auc(fpr, tpr)
print("AUC: ", roc_auc)
 
plot.title("Receiver Operating Characteristic")
plot.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)
plot.legend(loc = 'lower right')
plot.xlabel('False Positive Rate')
plot.ylabel('True Postive Rate')
plot.plot([0,1], [0,1], "r--")
plot.xlim([-1, 2])
plot.ylim([-1, 2])
plot.show()

# confusion matrices for different thresholds
y_predict_prob = lr.predict_proba(X_test.values)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
j = 1
classes = [0, 1]
plot.figure(figsize = (10, 10))
for i in thresholds:
     y_score =  y_predict_prob[:, 1] > i
     plot.subplot(3, 3, j)
     j += 1
     cnf = confusion_matrix(y_test, y_score)
     plot_cnf_matrix(cnf, classes)
     
# precision-recall curve for different thresholds     
plot.figure(figsize = (5, 5))   
colors = itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black'])
for i, color in zip(thresholds, colors):
     y_score =  y_predict_prob[:, 1] > i
     precision, recall, thresh = precision_recall_curve(y_test, y_score)
     plot.plot(recall, precision, color = color, label = 'Threshold: %s'%i)
     plot.title('Precision-Recall Curve')
     plot.xlabel('Recall')
     plot.ylabel('Precision')
     plot.xlim([0, 1.1])
     plot.ylim([0, 1.1])
     plot.legend(loc = 'lower left')

#==============================================================================
     
from sklearn import svm
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score 

# using SVM for prediction
def get_svm_scores(X_train, y_train):
    c_param_values = [0.01, 0.1, 1, 10, 100]
    kernels = ['linear', 'rbf']
    cv_score_table = pd.DataFrame(index = range(len(kernels) * len(c_param_values), 3), columns = ['C_Param', 'Kernel', 'Mean_CV_Score'])
    j = 0
    i = 0
    for c_param in c_param_values:
        for kernel in kernels:
            svm_model = svm.SVC(kernel = kernel, C = c_param)
            scores = cross_val_score(svm_model, X_train, y_train.values.ravel(), cv = 5)
            cv_score_table.ix[j+i, 'C_Param'] = c_param
            cv_score_table.ix[j+i, 'Kernel'] = kernel
            cv_score_table.ix[j+i, 'Mean_CV_Score'] = np.mean(scores)
            i +=1
        j += 1
    print(cv_score_table)
    best_c_param = cv_score_table.loc[cv_score_table['Mean_CV_Score'].idxmax()]['C_Param']
    best_kernel = cv_score_table.loc[cv_score_table['Mean_CV_Score'].idxmax()]['Kernel']                                  
    return best_c_param, best_kernel

final_c_param, final_kernel = get_svm_scores(X_train, y_train)
print("Final C param: ", final_c_param)
print("Final Kernel: ", final_kernel)

svm_model = svm.SVC(kernel=final_kernel, C = final_c_param)
y_predicted = cross_val_predict(svm_model, X_test, y_test.values.ravel(), cv = 10)
acc_score = accuracy_score(y_test, y_predicted)
print("Accuracy score: ", acc_score)
#==============================================================================

from sklearn import neighbors

# using KNeighborsClassifier for prediction of labels
def get_k_nearest_score(X_train, y_train):
    kfold = KFold(len(y_train), 5)
    num_neighbors = range(3, 15)
    weights = ['uniform', 'distance']

    neighbors_score_table = pd.DataFrame(index = range(len(num_neighbors) * len(weights), 3), columns = ['Number of Neighbors', 'Weight', 'Mean Accuracy'])
    j = 0
    i = 0
    for num in num_neighbors:
        for weight in weights:
            accuracy_scores = []
            for iteration, index in enumerate(kfold, start = 1):
                k_nearest_model = neighbors.KNeighborsClassifier(n_neighbors = num, weights = weight)
                k_nearest_model.fit(X_train.iloc[index[0], :], y_train.iloc[index[0], :].values.ravel())
                accuracy = k_nearest_model.score(X_train.iloc[index[1], :], y_train.iloc[index[1], :])
                accuracy_scores.append(accuracy)
                neighbors_score_table.ix[j+i, 'Number of Neighbors'] = num
                neighbors_score_table.ix[j+i, 'Weight'] = weight
                neighbors_score_table.ix[j+i, 'Mean Accuracy'] = np.mean(accuracy_scores)
            i +=1
        j += 1
    print(neighbors_score_table)
    number_neighbors = neighbors_score_table.loc[neighbors_score_table['Mean Accuracy'].idxmax()]['Number of Neighbors']
    weight = neighbors_score_table.loc[neighbors_score_table['Mean Accuracy'].idxmax()]['Weight']                                  
    return number_neighbors, weight

num_neighbors, weight = get_k_nearest_score(X_train, y_train)
print("Number of Neighbors: ", num_neighbors)
print("Weight: ", weight)
  
k_nearest_model = neighbors.KNeighborsClassifier(n_neighbors = num_neighbors, weights = weight)
k_nearest_model.fit(X_train, y_train.values.ravel())
accuracy = k_nearest_model.score(X_test, y_test)
print("Accuracy: ", accuracy)

#==============================================================================
from sklearn import tree

# using decision trees for prediction of labels
decision_tree = tree.DecisionTreeClassifier(random_state = 99)
decision_tree.fit(X_train, y_train.values.ravel())
accuracy = decision_tree.score(X_test, y_test)
print("Mean Accuracy with Decision Trees: ", accuracy)
#==============================================================================
