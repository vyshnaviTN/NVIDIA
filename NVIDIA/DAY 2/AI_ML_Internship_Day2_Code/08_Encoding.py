import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.DataFrame({"Color":["Red","Blue","Green","Red"]})
le=LabelEncoder()
df["Color_Label"]=le.fit_transform(df["Color"])
print(df)
print(pd.get_dummies(df["Color"],prefix="Color"))