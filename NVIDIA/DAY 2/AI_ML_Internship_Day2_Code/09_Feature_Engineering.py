import pandas as pd

df=pd.DataFrame({"DOB":["2000-01-01","1998-05-10"],"Salary":[50000,70000]})
df["DOB"]=pd.to_datetime(df["DOB"])
df["Age"]=2026-df["DOB"].dt.year
df["Salary_Log"]=df["Salary"].apply(__import__("math").log)
print(df)