#Implement k-Nearest Neighbour Algorithm to Classify the Iris Dataset and Print Predictions

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbor = 3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the k-NN model: {accuracy * 100:.2f}%")
