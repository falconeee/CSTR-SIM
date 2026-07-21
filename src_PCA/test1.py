from sklearn.model_selection import train_test_split
import numpy as np

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 1, 2, 3])

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=3, random_state=None)

print("TRAIN:", X_train, y_train)
print("TEST:", X_test, y_test)
