import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

testPath = '../ProcessedData/test.csv'
trainPath = '../ProcessedData/train.csv'
survivedPath = '../RawData/survived_test.csv'

train = pd.read_csv(trainPath, header=0, index_col=0)
X_test = pd.read_csv(testPath, header=0, index_col=0)
y_test = pd.read_csv(survivedPath, header=0, index_col=0)

'''
train.info()
X_test.info()


#train.corr()

first_class_survival = train.loc[train['Pclass'] == 1, 'Survived']
second_class_survival = train.loc[train['Pclass'] == 2, 'Survived']
third_class_survival = train.loc[train['Pclass'] == 3, 'Survived']

print("Survival rate for first class: " + str(first_class_survival.mean()))
print("Survival rate for second class: " + str(second_class_survival.mean()))
print("Survival rate for third class: " + str(third_class_survival.mean()))
# 1st class: 63% survival. 2nd class: 47%. 3rd class: 24%.

# p-value of a variable = 
'''

trainLen = train.shape[0]
pclass = [0.75, 1.75, 2.75]
ticks=[1, 2, 3]
colors = ['Green', 'Blue', 'Red']
survival_rate = [train.loc[train['Pclass'] == 1, 'Survived'].mean() * 100, train.loc[train['Pclass'] == 2, 'Survived'].mean() * 100, train.loc[train['Pclass'] == 3, 'Survived'].mean() * 100]
plt.bar(left=pclass, height=survival_rate, color=colors, width=0.75, tick_label=ticks, align='center')
plt.title("Survival rate by passenger class")
plt.xlabel("Class")
plt.ylabel("Survival rate (%)")
plt.show()


plt.subplot(1, 2, 1)
plt.suptitle("Survival versus fare price")
plt.boxplot(x=list(train.loc[train['Survived'] == True, 'Fare']))
plt.ylim(-10, 150)
plt.xlabel("Survived")
plt.ylabel("Fare ($)")
plt.xticks([])
plt.subplot(1, 2, 2)
plt.boxplot(x=list(train.loc[train['Survived'] == False, 'Fare']))
plt.ylim(-10, 150)
plt.xlabel("Died")
plt.xticks([])
plt.show()

pclass = [0.75, 1.75]
ticks=["Cabin known","Cabin unknown"]
colors = ['Yellow', 'Black']
survival_rate = [train.loc[train['cabinUnknown'] == False, 'Survived'].mean() * 100, train.loc[train['cabinUnknown'] == True, 'Survived'].mean() * 100]
plt.bar(left=pclass, height=survival_rate, color=colors, width=0.75, tick_label=ticks, align='center')
plt.title("Survival rate by cabin status")
plt.xlabel("Cabin status")
plt.ylabel("Survival rate (%)")
plt.show()

embarked = [0.75, 1.75, 2.75]
ticks=["Cherbourg", "Queenstown", "Southampton"]
colors = ['Orange', 'Purple', 'Gray']
survival_rate = [train.loc[train['Embarked_C'] == True, 'Survived'].mean() * 100, train.loc[train['Embarked_Q'] == True, 'Survived'].mean() * 100, train.loc[train['Embarked_S'] == True, 'Survived'].mean() * 100]
plt.bar(left=embarked, height=survival_rate, color=colors, width=0.75, tick_label=ticks, align='center')
plt.title("Survival rate by place of embarkment")
plt.xlabel("Place of boarding")
plt.ylabel("Survival rate (%)")
plt.show()

markers = [0.75, 1.75, 2.75, 3.75]
ticks=["Boy", "Man", "Unmarried Female", "Married Female"]
colors = ['cyan', 'blue', 'pink', 'red']
survival_rate = [train.loc[train['isMaster'] == True, 'Survived'].mean() * 100, train.loc[train['isMr'] == True, 'Survived'].mean() * 100, train.loc[train['isMiss'] == True, 'Survived'].mean() * 100, train.loc[train['isMrs'] == True, 'Survived'].mean() * 100]
plt.bar(left=markers, height=survival_rate, color=colors, width=0.75, tick_label=ticks, align='center')
plt.title("Survival rate by approximate age/sex demographic")
plt.xlabel("Demographic")
plt.ylabel("Survival rate (%)")
plt.ylim(0, 90)
plt.show()

# class and demographic
markers = [0.75, 1.75, 2.75, 3.75]
ticks=["Boy", "Mr.", "Miss", "Mrs."]
plt.suptitle("Survival rate by class and demographic")
colors = ['cyan', 'blue', 'pink', 'red']
survival_rate_1 = [train.loc[(train['isMaster'] == True) & (train['Pclass'] == 1), 'Survived'].mean() * 100, train.loc[(train['isMr'] == True) & (train['Pclass'] == 1), 'Survived'].mean() * 100, train.loc[(train['isMiss'] == True) & (train['Pclass'] == 1), 'Survived'].mean() * 100, train.loc[(train['isMrs'] == True) & (train['Pclass'] == 1), 'Survived'].mean() * 100]
survival_rate_2 = [train.loc[(train['isMaster'] == True) & (train['Pclass'] == 2), 'Survived'].mean() * 100, train.loc[(train['isMr'] == True) & (train['Pclass'] == 2), 'Survived'].mean() * 100, train.loc[(train['isMiss'] == True) & (train['Pclass'] == 2), 'Survived'].mean() * 100, train.loc[(train['isMrs'] == True) & (train['Pclass'] == 2), 'Survived'].mean() * 100]
survival_rate_3 = [train.loc[(train['isMaster'] == True) & (train['Pclass'] == 3), 'Survived'].mean() * 100, train.loc[(train['isMr'] == True) & (train['Pclass'] == 3), 'Survived'].mean() * 100, train.loc[(train['isMiss'] == True) & (train['Pclass'] == 3), 'Survived'].mean() * 100, train.loc[(train['isMrs'] == True) & (train['Pclass'] == 3), 'Survived'].mean() * 100]
plt.subplot(1, 3, 1, axisbg='#e6e6e6')
plt.bar(left=markers, height=survival_rate_1, color=colors, width=0.75, tick_label=ticks, align='center')
plt.ylim(0, 110)
plt.title('1st Class')
plt.ylabel("Survival rate (%")
plt.subplot(1, 3, 2, axisbg='#808080')
plt.bar(left=markers, height=survival_rate_2, color=colors, width=0.75, tick_label=ticks, align='center')
plt.ylim(0, 110)
plt.title('2nd Class')
plt.subplot(1, 3, 3, axisbg='black')
plt.bar(left=markers, height=survival_rate_3, color=colors, width=0.75, tick_label=ticks, align='center')
plt.ylim(0, 110)
plt.title('3rd Class')
plt.show()


markers = [0.75, 1.75, 2.75]
ticks=["Che.", "Que.", "Sou."]
plt.suptitle("Survival rate by class and departure point")
colors = ['orange', 'purple', 'green']
survival_rate_1 = [train.loc[(train['Embarked_C'] == True) & (train['Pclass'] == 1), 'Survived'].mean() * 100, train.loc[(train['Embarked_Q'] == True) & (train['Pclass'] == 1), 'Survived'].mean() * 100, train.loc[(train['Embarked_S'] == True) & (train['Pclass'] == 1), 'Survived'].mean() * 100]
survival_rate_2 = [train.loc[(train['Embarked_C'] == True) & (train['Pclass'] == 2), 'Survived'].mean() * 100, train.loc[(train['Embarked_Q'] == True) & (train['Pclass'] == 2), 'Survived'].mean() * 100, train.loc[(train['Embarked_S'] == True) & (train['Pclass'] == 2), 'Survived'].mean() * 100]
survival_rate_3 = [train.loc[(train['Embarked_C'] == True) & (train['Pclass'] == 3), 'Survived'].mean() * 100, train.loc[(train['Embarked_Q'] == True) & (train['Pclass'] == 3), 'Survived'].mean() * 100, train.loc[(train['Embarked_S'] == True) & (train['Pclass'] == 3), 'Survived'].mean() * 100]
plt.subplot(1, 3, 1, axisbg='#e6e6e6')
plt.bar(left=markers, height=survival_rate_1, color=colors, width=0.75, tick_label=ticks, align='center')
plt.ylim(0, 110)
plt.title('1st Class')
plt.ylabel("Survival rate (%")
plt.subplot(1, 3, 2, axisbg='#808080')
plt.bar(left=markers, height=survival_rate_2, color=colors, width=0.75, tick_label=ticks, align='center')
plt.ylim(0, 110)
plt.title('2nd Class')
plt.subplot(1, 3, 3, axisbg='black')
plt.bar(left=markers, height=survival_rate_3, color=colors, width=0.75, tick_label=ticks, align='center')
plt.ylim(0, 110)
plt.title('3rd Class')
plt.show()
# interaction terms, p-values, then testing. don't linger here