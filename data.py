import pandas as pd
import math 
import re
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

#helper function to get letters from string
def return_letter(df):
    ret = df
    i = 0
    for entry in df:
        try:
            if math.isnan(entry):
                pass
        except:
            ret.iloc[i] = re.findall("^[a-zA-Z]", entry)[0]
        i+=1

    return ret

#helper function to preprocess data
def preprocess(df):
    #filter categotical
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0) #0 female, 1 male

    #onehot on embark
    df = pd.get_dummies(df, columns=['Embarked'])

    #drop passengerId, Name
    df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    #transform age data to account for nan values
    df['Age'].fillna(df['Age'].mean(), inplace=True)

    return_letter(df['Cabin'])
    df = pd.get_dummies(df, columns=['Cabin'])
    df[['Age', 'Fare']] = StandardScaler().fit_transform(df[['Age', 'Fare']])
    return df



#model start
train = pd.read_csv('train.csv')

train = preprocess(train)
X_train = train.drop(['Survived'], axis=1)
y_train = train['Survived']

lgbm = LGBMClassifier(boosting_type= 'gbdt', learning_rate= 0.1, n_estimators= 50, num_leaves= 31)
lgbm.fit(X_train, y_train)

#predictions
test = pd.read_csv('test.csv')
solution = pd.DataFrame(test['PassengerId'], columns=['PassengerId'])

test = preprocess(test)
test['Cabin_T'] = 0 #no one in Cabin T in test set

preds = lgbm.predict(test)

solution['Survived'] = preds
solution.to_csv('solution.csv', index=False)