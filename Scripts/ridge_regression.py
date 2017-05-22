import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report

alpha_ridge = np.linspace(0.01, 3, 200)

testPath = '../ProcessedData/test.csv'
trainPath = '../ProcessedData/train.csv'
survivedPath = '../RawData/survived_test.csv'

train = pd.read_csv(trainPath, header=0, index_col=0)
X_test = pd.read_csv(testPath, header=0, index_col=0)
y_test = pd.read_csv(survivedPath, header=0, index_col=0)

y_train = train['Survived']
X_train = train.drop(['Survived'], axis=1)
ridge_scores = []

for i in range(len(alpha_ridge)):
    alpha = alpha_ridge[i]
    clf = OneVsRestClassifier(Ridge(alpha=alpha,normalize=True))
    clf.fit(X_train, y_train)
    #y_pred = ridgereg.predict(X_test)
    ridge_scores.append(clf.score(X_train, y_train))

plt.plot(alpha_ridge, ridge_scores)
plt.xlabel("Alpha parameter for ridge regression")
plt.ylabel("Model F Score")
plt.show()

maxAlpha = alpha_ridge[np.argmax(ridge_scores)]
clf = OneVsRestClassifier(Ridge(alpha=alpha,normalize=True))
clf.fit(X_train, y_train)
print(str(clf.score(X_test, y_test)))

y_pred = pd.DataFrame(clf.predict(X_test)) 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

idRange = range(892, 1310)
y_pred['PassengerId'] = idRange
y_pred.columns = ["Survived", "PassengerId"]
y_pred = y_pred[['PassengerId', 'Survived']]
y_pred.to_csv('../Predictions/predictions_ridge.csv', index=False)