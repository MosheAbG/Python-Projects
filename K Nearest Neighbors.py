# The following code will use the K Nearest Neighbors approach
# The data from the data set is not easily understandable and therefore scaling will be used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("KNN_Project_Data")
from sklearn.preprocessing import StandardScaler

#scale the data
scaler = StandardScaler()
scaler.fit(df.drop("TARGET CLASS", axis=1))
#transform the features into a scaled version
scaled_version = scaler.transform(df.drop("TARGET CLASS", axis=1))
df_new = pd.DataFrame(scaled_version,columns=df.columns[:-1])
print(df_new)
# Split the data into a training set and a test set
from sklearn.model_selection import train_test_split
X = df_new
y = df["TARGET CLASS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
# Create a KNN Model
KNN_model = KNeighborsClassifier(n_neighbors=1)
KNN_model.fit(X_train,y_train)
#Make predictions about the X_test data using KNN
pred = KNN_model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))
# Find an optimal k value, as 1 isn't the best
error = []

for i in range(1,40):
    knn_i = KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(X_train, y_train)
    prediction = knn_i.predict(X_test)
    error.append(np.mean(y_test != prediction))

print(error)

sns.lineplot(x=range(1,40), y=error, color="blue", linestyle="--", marker="o", markerfacecolor="red")
plt.title("Error Rate vs. K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()
#Retrain the data with the new found K value
KNN = KNeighborsClassifier(n_neighbors=31)
KNN.fit(X_train, y_train)
pred = KNN.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))
sns.histplot()