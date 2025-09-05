import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetic retinopathy.csv')

df.head()



sns.set_style('whitegrid')
sns.countplot(x='Class',data=df,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Class',data=df,hue='18',palette='RdBu_r')
plt.title('Based on AM/FM based classification')

sns.set_style('whitegrid')
sns.countplot(x='Class',data=df,hue='1',palette='RdBu_r')
plt.title('Based on Retinal Abnormality')

from sklearn.model_selection import train_test_split

X = df.drop(['id','Class'],axis=1)
y = df['Class']
X.head()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
scaler = StandardScaler()

scaler.fit(X)

scaled_features=scaler.transform(X)

df_feat= pd.DataFrame(scaled_features)
df_feat.head()

X_train, X_test, y_train, y_test = train_test_split(df_feat, y, test_size=0.33, random_state=42)
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
predictions=knn.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
error_rate = []

for i in range(1,100):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn = KNeighborsClassifier(n_neighbors=70)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=80')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))