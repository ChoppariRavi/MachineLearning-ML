from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
Y = iris.target

f_names = iris.feature_names
t_names = iris.target_names

print("Feature names:", f_names) 
print("Target names:", t_names) 

print("\nType of X is:", type(X)) 

print("\nFirst 5 rows of X:\n", X[:5])