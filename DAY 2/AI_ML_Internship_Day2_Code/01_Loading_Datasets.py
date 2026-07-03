import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

# CSV from GitHub
url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df=pd.read_csv(url)
print(df.head())

# Seaborn dataset
print(sns.load_dataset("tips").head())

iris=load_iris(as_frame=True)
print(iris.frame.head())