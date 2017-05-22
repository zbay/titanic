from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.metrics import confusion_matrix, classification_report

random.seed(1)

testPath = '../ProcessedData/test.csv'
trainPath = '../ProcessedData/train.csv'
survivedPath = '../RawData/survived_test.csv'

train = pd.read_csv(trainPath, header=0, index_col=0)
X_test = pd.read_csv(testPath, header=0, index_col=0)
y_test = pd.read_csv(survivedPath, header=0, index_col=0)

y_train = train['Survived']
X_train = train.drop(['Survived'], axis=1)

estimators = np.linspace(10, 45, 36)
max_features = np.linspace(5, 23, 19)
forest_scores = [[]]

for i in range(len(estimators)):
    for j in range(len(max_features)):
        clf = RandomForestClassifier(n_estimators=int(estimators[i]), max_features=int(max_features[j]))
        clf.fit(X_train, y_train)
        while len(forest_scores) < (i+1):
            forest_scores.append([])
        forest_scores[i].append(clf.score(X_train, y_train))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# reshape data to same dimensions for 3D plotting purposes
estimators_3D = []
max_features_3D = []
scores_3D = []
for i in range(0, len(estimators)):
    for j in range(0, len(max_features)):
        estimators_3D.append(estimators[i])
        max_features_3D.append(max_features[j])
        scores_3D.append(forest_scores[i][j])
ax.scatter(estimators_3D, max_features_3D, scores_3D)
ax.set_xlabel("Trees")
ax.set_ylabel("Features")
ax.set_zlabel("F-Score")
plt.show()

max_score = max(scores_3D)
scores_3D = np.array(scores_3D)
max_index = np.argmax(scores_3D)
estimators_best = estimators_3D[max_index]
max_features_best = max_features_3D[max_index]
print("Top score on training set: " + str(max_score))
print("Number of estimators in best model: " + str(estimators_best))
print("Max features of best model: " + str(max_features_best))

clf = RandomForestClassifier(n_estimators=int(estimators_best), max_features=int(max_features_best))
clf.fit(X_train, y_train)
print(str(clf.score(X_test, y_test)))

y_pred = pd.DataFrame(clf.predict(X_test)) 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

idRange = range(892, 1310)
y_pred['PassengerId'] = idRange
y_pred.columns = ["Survived", "PassengerId"]
y_pred = y_pred[['PassengerId', 'Survived']]
y_pred.to_csv('../Predictions/predictions_forest.csv', index=False)