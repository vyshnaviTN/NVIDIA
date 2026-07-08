import pandas as pd
import seaborn as sns
df=sns.load_dataset("titanic")
print(df.isnull().sum())
df=df.drop_duplicates()
df['age']=df['age'].fillna(df['age'].median())
df['embarked']=df['embarked'].fillna(df['embarked'].mode()[0])
print(df.info())