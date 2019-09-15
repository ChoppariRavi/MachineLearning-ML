from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, Y_train) 
  
Y_pred = knn.predict(X_test) 
  
from sklearn import metrics 
print("kNN model accuracy:", metrics.accuracy_score(Y_test, Y_pred)) 
  
sample = [[3, 5, 4, 2], [2, 3, 5, 4]] 
preds = knn.predict(sample) 
pred_species = [iris.target_names[p] for p in preds] 
print("Predictions:", pred_species) 
  
from sklearn.externals import joblib 
joblib.dump(knn, 'iris_knn.pkl')