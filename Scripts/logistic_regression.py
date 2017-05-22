import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report

testPath = '../ProcessedData/test.csv'
trainPath = '../ProcessedData/train.csv'
survivedPath = '../RawData/survived_test.csv'

train = pd.read_csv(trainPath, header=0)
X_test = pd.read_csv(testPath, header=0)
y_test = pd.read_csv(survivedPath, header=0, index_col=0)

y_train = train['Survived']
X_train = train.drop(['Survived'], axis=1)

clf = OneVsRestClassifier(LogisticRegression())

clf.fit(X_train, y_train)

print(str(clf.score(X_test, y_test)))

y_pred = pd.DataFrame(clf.predict(X_test)) 

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_pred_prob = pd.DataFrame(clf.predict_proba(X_test)[:,1])

idRange = range(892, 1310)
y_pred['PassengerId'] = idRange
y_pred.columns = ["Survived", "PassengerId"]
y_pred = y_pred[['PassengerId', 'Survived']]
y_pred.to_csv('../Predictions/predictions_logistic.csv', index=False)