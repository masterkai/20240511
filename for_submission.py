# import model
import joblib
model_pretrained = joblib.load('titanic-model-20240511.pkl')

import pandas as pd
# Load the data
df_test = pd.read_csv('test.csv')
df_test.drop(['Name', 'Ticket'], axis = 1 ,inplace=True) #刪除無用的列
df_test.drop("Cabin", axis = 1, inplace=True) #刪除Cabin列
df_test.info()

df_test['Age'].fillna(df_test.groupby("Sex")["Age"].transform("median"), inplace=True)
df_test['Fare'].value_counts()
df_test['Fare'].fillna(df_test['Fare'].value_counts().idxmax(), inplace=True)

df_test = pd.get_dummies(data=df_test,dtype=int, columns=["Sex", "Embarked"])

df_test.drop("Sex_female", axis = 1, inplace=True)
df_test.drop("Pclass", axis = 1, inplace=True)

predictions2 = model_pretrained.predict(df_test)

#Prepare the submission file
for_submissionDF = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predictions2})

for_submissionDF.to_csv('submission_20240511.csv', index=False)
