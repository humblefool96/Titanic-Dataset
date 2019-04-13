import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data = pd.concat([train, test], sort=False)

print(train.columns.values)
print(train.head())
print(train.tail())
print(train.info())
print(test.info())

import seaborn as sns
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

#analysing the important features
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#filling missing values
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data['title']=data.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}

data['title']=data.title.map(newtitles)
data.groupby(['title','Sex']).Age.mean()

def newage (cols):
    title=cols[0]
    Sex=cols[1]
    Age=cols[2]
    if pd.isnull(Age):
        if title=='Master' and Sex=="male":
            return 4.57
        elif title=='Miss' and Sex=='female':
            return 21.8
        elif title=='Mr' and Sex=='male': 
            return 32.37
        elif title=='Mrs' and Sex=='female':
            return 35.72
        elif title=='Officer' and Sex=='female':
            return 49
        elif title=='Officer' and Sex=='male':
            return 46.56
        elif title=='Royalty' and Sex=='female':
            return 40.50
        else:
            return 42.33
    else:
        return Age 
    
data.Age=data[['title','Sex','Age']].apply(newage, axis=1)
print(data.isnull().sum())

data['Relatives'] = data.SibSp + data.Parch
data = data.drop(['Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1)
data[['title', 'Survived']].groupby(['title'], as_index=False).mean().sort_values(by='Survived', ascending=False)

x_train = data.iloc[:891, : 9 ]
x_train = x_train.drop(['PassengerId', 'Survived'], axis=1)
x_train = pd.get_dummies(x_train)
y_train = data.iloc[:891, [1]]

x_test = data.iloc[891:, : 9]
x_test = x_test.drop(['PassengerId', 'Survived'], axis=1)
x_test = pd.get_dummies(x_test)

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x  = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

# Fitting model to the Training set
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
RF=RandomForestClassifier(random_state=1)
PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]
GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=2)
scores_rf=cross_val_score(GSRF,x_train,y_train,scoring='accuracy',cv=5)
np.mean(scores_rf)

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
svc=make_pipeline(StandardScaler(),SVC(random_state=1))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)
scores_svm=cross_val_score(GSSVM, x_train.astype(float), y_train,scoring='accuracy', cv=5)
np.mean(scores_svm)

model=GSSVM.fit(x_train, y_train)

# Predicting the Test set results
y_pred = model.predict(x_test).astype(int)

output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})
output.to_csv('submission.csv', index=False)

"""submission=pd.read_csv("gender_submission.csv")
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
pd.DataFrame(submission, columns=['PassengerId','Survived']).to_csv('result.csv')"""