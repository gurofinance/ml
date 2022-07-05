from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print(X)
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])