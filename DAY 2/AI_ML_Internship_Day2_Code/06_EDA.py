import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df=sns.load_dataset("titanic")
print(df.describe(include="all"))
print(df.corr(numeric_only=True))
sns.histplot(df["age"].dropna())
plt.show()
sns.boxplot(x=df["fare"])
plt.show()
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()