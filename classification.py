import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Dataset importing
diabetes = pd.read_csv('./diabetes_data_upload.csv')


# Show dataset histogram
diabetes.hist(bins=50,figsize=(20,15))
plt.show()


# Show dataset correlation matrix
corr_matrix = diabetes.corr()
print(corr_matrix)


# Create scatter matrix
from pandas.plotting import scatter_matrix

# Selecting which attributes to use for classification.
attributes = ["Age", "Gender", "Polyuria", "Polydipsia"]
scatter_matrix(diabetes[attributes], figsize=(12,8))


# Drop the estimated result column from the dataset
diabetes_labels = diabetes["class"]
diabetes = diabetes.drop("class",axis=1)


# Min-Max Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(diabetes)


# Train Test Split %33 Begin -------------------------------------------------------------------------------------------
print("\n\nResults For Train Test Split %33 -------------------------------------------------------------------------------")
from sklearn.model_selection import train_test_split
train_set, test_set, train_labels, test_labels = train_test_split(diabetes,diabetes_labels, test_size=0.33, random_state=42)


# StochasticGradientDescentClassifier Begin
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(train_set,train_labels)
prediction = sgd_clf.predict(test_set)

print("SGDClassifier Result: ", prediction)
# StochasticGradientDescentClassifier End


# DecisionTreeClassifier Begin
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(train_set,train_labels)
prediction = tree.predict(test_set)

print("DTClassifier Result:", prediction)
# DecisionTreeClassifier End


# RandomForestClassifier Begin
from sklearn.ensemble import RandomForestClassifier

forest_reg = RandomForestClassifier(random_state=42)
forest_reg.fit(train_set,train_labels)
prediction = forest_reg.predict(test_set)

print("RFClassifier Result:", prediction)
# RandomForestClassifier End


# Train Test Split %33 End ---------------------------------------------------------------------------------------------





# 3 Folds Cross-Validation Begin ---------------------------------------------------------------------------------------
print("\n\nResults For 3 Folds Cross-Validation Begin ---------------------------------------------------------------------")
from sklearn.model_selection import cross_val_score
scores = cross_val_score(sgd_clf, train_set, train_labels, cv=3, scoring="neg_mean_squared_error")


# StochasticGradientDescentClassifier Begin
sgd_clf_rmse_scores = np.sqrt(-scores)

print("SGDClassifier Scores: ", sgd_clf_rmse_scores)
print("SGDClassifier mean: ", sgd_clf_rmse_scores.mean())
print("SGDClassifier std: ", sgd_clf_rmse_scores.std())

# Cross Val Predict
from sklearn.model_selection import cross_val_predict
train_pred = cross_val_predict(sgd_clf, train_set, train_labels, cv=3)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
c_mat = confusion_matrix(train_labels, train_pred)
print("SGDClassifier Confusion Matrix")
print(c_mat)

# Precision, Recall ve F1-score
from sklearn.metrics import precision_score, recall_score, f1_score

print("SGDClassifier Precision Score: ", precision_score(train_labels, train_pred, average="macro"))
print("SGDClassifier Recall Score: ", recall_score(train_labels, train_pred, average="macro"))
print("SGDClassifier F1 Score: ", f1_score(train_labels, train_pred, average="macro"))
print("*******************************************************************************************")
# StochasticGradientDescentClassifier End


# DecisionTreeClassifier Begin
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree,train_set,train_labels,cv=3, scoring="neg_mean_squared_error")
tree_rmse_scores = np.sqrt(-scores)

print("DTClassifier Scores: ", tree_rmse_scores)
print("DTClassifier mean: ", tree_rmse_scores.mean())
print("DTClassifier std: ", tree_rmse_scores.std())

# Cross Val Predict
from sklearn.model_selection import cross_val_predict
train_pred = cross_val_predict(tree, train_set, train_labels, cv=3)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
c_mat = confusion_matrix(train_labels, train_pred)
print("DTClassifier Confusion Matrix")
print(c_mat)

# Precision, Recall ve F1-score
from sklearn.metrics import precision_score, recall_score, f1_score

print("DTClassifier Precision Score: ", precision_score(train_labels, train_pred,average="macro"))
print("DTClassifier Recall Score: ", recall_score(train_labels, train_pred,average="macro"))
print("DTClassifier F1 Score: ", f1_score(train_labels, train_pred,average="macro"))
print("*******************************************************************************************")
# DecisionTreeClassifier End


# RandomForestClassifier Begin
from sklearn.model_selection import cross_val_score

scores = cross_val_score(forest_reg, train_set,train_labels,cv=3, scoring="neg_mean_squared_error")
forest_rmse_scores = np.sqrt(-scores)

print("RFClassifier Scores: ", forest_rmse_scores)
print("RFClassifier mean: ", forest_rmse_scores.mean())
print("RFClassifier std: ", forest_rmse_scores.std())

# Cross Val Predict
from sklearn.model_selection import cross_val_predict
train_pred = cross_val_predict(forest_reg, train_set, train_labels, cv=3)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
c_mat = confusion_matrix(train_labels, train_pred)
print("RFClassifier Confusion Matrix")
print(c_mat)

# Precision, Recall ve F1-score
from sklearn.metrics import precision_score, recall_score, f1_score

print("RFClassifier Precision Score: ", precision_score(train_labels, train_pred.round(),average="macro"))
print("RFClassifier Recall Score: ", recall_score(train_labels, train_pred.round(),average="macro"))
print("RFClassifier F1 Score: ", f1_score(train_labels, train_pred.round(),average="macro"))
print("*******************************************************************************************")
# RandomForestClassifier End

# 3 Folds Cross-Validation End -----------------------------------------------------------------------------------------