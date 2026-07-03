from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import SMOTE
X,y=make_classification(n_samples=500,weights=[0.9,0.1],random_state=42)
print(Counter(y))
Xr,yr=SMOTE(random_state=42).fit_resample(X,y)
print(Counter(yr))