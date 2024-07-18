# import modules
import pandas as pd #導入pandas模組用於數據處理
import numpy as np #導入numpy模組用於數學運算
import matplotlib.pyplot as plt #導入matplotlib.pyplot模組用於繪製圖表
import seaborn as sns #導入seaborn模組用於繪製圖表

# 讀取數據
df = pd.read_csv('https://raw.githubusercontent.com/ryanchung403/dataset/main/train_data_titanic.csv') #讀取數據

# 顯示數據前5行
df.head() #顯示數據前5行
df.info() #顯示數據信息

# 移除無用的列
df = df.drop(['Name', 'Ticket'], axis = 1 ) #刪除無用的列
df.head() #顯示數據前5行

sns.pairplot(df[['Survived','Fare']], dropna=True) #繪製票價與存活情況的數據圖表
sns.pairplot(df[['Survived','Age']], dropna=True) #繪製年齡與存活情況的數據圖表
sns.pairplot(df[['Survived','Pclass']], dropna=True) #繪製艙位等級與存活情況的數據圖表
sns.pairplot(df[['Survived','SibSp']], dropna=True) #繪製兄弟姐妹配偶數量與存活情況的數據圖表
sns.pairplot(df[['Survived','Parch']], dropna=True) #繪製父母子女數量與存活情況的數據圖表

df.groupby('Survived').mean(numeric_only=True) #顯示存活情況的平均值

# #顯示fare>500的數據
# df[df['Fare'] > 500] #顯示票價大於500的數據

# #移除Fare>500的數據
# df = df[df['Fare'] < 500] #刪除票價大於500的數據
# sns.pairplot(df[['Survived','Fare']], dropna=True) #繪製票價與存活情況的數據圖表
# df.groupby('Survived').mean(numeric_only=True) #顯示存活情況的平均值

#data observation
df['SibSp'].value_counts() #顯示兄弟姐妹配偶數量的數據
df['SibSp'].value_counts().sort_index() #顯示兄弟姐妹配偶數量的數據
df['SibSp'].value_counts().sort_index() #顯示兄弟姐妹配偶數量的數據
df['SibSp'].value_counts().sort_values(ascending=False) #顯示兄弟姐妹配偶數量的數據

df['Parch'].value_counts() #顯示父母子女數量的數據
df['Sex'].value_counts() #顯示性別的數據

df.isnull().sum() > len(df)/2 #顯示大於數據一半的空值
len(df/2) #顯示數據一半的長度
df.isnull().sum() #顯示空值的數量

#Cabin 有大量空值，刪除Cabin列
df = df.drop(['Cabin'], axis = 1) #刪除Cabin列

#Obsere the median of different gender's age
df.groupby('Sex')['Age'].median() #顯示不同性別的年齡中位數

df.groupby('Sex')['Age'].transform('median') #用不同性別的年齡中位數創建新列



#用不同性別的年齡中位數填充空值
df['Age'] = df['Age'].fillna(df.groupby("Sex")['Age'].transform('median')) #用不同性別的年齡中位數填充空值
df.isnull().sum() #顯示空值的數量


#Embarked有少量空值，用眾數填充空值
df['Embarked'].value_counts() #顯示Embarked的數據
df['Embarked'].value_counts().idxmax() #顯示Embarked的眾數
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().idxmax()) #用Embarked的眾數填充空值
df.isnull().sum() #顯示空值的數量

#Convert categorical data to numerical data
df = pd.get_dummies(data=df,dtype=int,columns=['Sex', 'Embarked']) #將分類數據轉換為數字數據 
df.head() #顯示數據前5行

#drop the Sex_female column
df = df.drop('Sex_female', axis = 1) 
df.head() #顯示數據前5行

df.corr() #顯示數據的相關性

#split the data into x and y
x = df.drop(['Survived','Pclass'], axis = 1) #刪除Survived列
y = df['Survived'] #設置Survived列為y

#split the data into training and testing data
from sklearn.model_selection import train_test_split #導入train_test_split模組
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) #將數據分為訓練數據和測試數據 80%訓練數據 20%測試數據

#train the model
from sklearn.linear_model import LogisticRegression #導入LogisticRegression模組
model = LogisticRegression(max_iter=1000) #設置模型為LogisticRegression
model.fit(x_train, y_train) #訓練模型

#predict the test data
predictions = model.predict(x_test) #預測測試數據

#Evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score,precision_score,f1_score #導入confusion_matrix和accuracy_score模組
precision_score(y_test, predictions) #顯示精確率
accuracy_score(y_test, predictions) #顯示準確率
recall_score(y_test, predictions) #顯示召回率
f1_score(y_test, predictions) #顯示F1值

#confusion matrix
confusion_matrix(y_test, predictions) #顯示混淆矩陣

#create dataframe and add columns to confusion matrix
pd.DataFrame(confusion_matrix(y_test, predictions), columns=['Predicted Not Survived', 'Predicted Survived'], index=['Actual Not Survived', 'Actual Survived']) #創建數據框並將列添加到混淆矩陣

# save the model
import joblib #導入joblib模組
joblib.dump(model, 'titanic-model-20240511.pkl', compress=3) #保存模型





