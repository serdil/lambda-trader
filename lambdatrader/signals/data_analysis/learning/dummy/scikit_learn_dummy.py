import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
digits = datasets.load_digits()

print('digits.data')
print(digits.data)
print(len(digits.data))

print('digits.target')
print(digits.target)

print('digits.images[0]')
print(digits.images[0])

clf = svm.SVC(gamma=0.001, C=100.)

print(clf)

clf.fit(digits.data[:-1], digits.target[:-1])

prediction = clf.predict(digits.data[-1:])
print(prediction)

# Reshaping

print(digits.images.shape)
# ax = plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
# plt.show()

data = digits.images.reshape((digits.images.shape[0], -1))


print('KNN')

X_digits = digits.data
y_digits = digits.target

n_samples = len(digits.target)

X_digits_train = X_digits[:int(0.9 * n_samples)]
y_digits_train = y_digits[:int(0.9 * n_samples)]

X_digits_test = X_digits[int(0.9 * n_samples):]
y_digits_test = y_digits[int(0.9 * n_samples):]

knn = KNeighborsClassifier()
knn.fit(X_digits_train, y_digits_train)

predictions = knn.predict(X_digits_test)

print(y_digits_test)
print(predictions)

print('knn score:', knn.score(X_digits_test, y_digits_test))

print('LOGISTIC REGRESSION')

logistic = LogisticRegression(C=1e5)

logistic.fit(X_digits_train, y_digits_train)

predictions = logistic.predict(X_digits_test)

print(y_digits_test)
print(predictions)

print('logistic score:', logistic.score(X_digits_test, y_digits_test))


diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

print('LINEAR REGRESSION')

regr = LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

print('regr.coef_:', regr.coef_)

# The mean square error
mse = np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
score = regr.score(diabetes_X_test, diabetes_y_test)

print('mse:', mse, 'score:', score)

print('LASSO')

alphas = np.logspace(-4, -1, 6)

regr = Lasso()
scores = [regr.set_params(alpha=alpha
            ).fit(diabetes_X_train, diabetes_y_train
            ).score(diabetes_X_test, diabetes_y_test)
       for alpha in alphas]
best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(diabetes_X_train, diabetes_y_train)

mse = np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)
score = regr.score(diabetes_X_test, diabetes_y_test)

print('regr.coef_', regr.coef_)
print('mse:', mse, 'score:', score)


print('RANDOM FOREST')

X_digits = digits.data
y_digits = digits.target

n_samples = len(digits.target)

X_digits_train = X_digits[:int(0.9 * n_samples)]
y_digits_train = y_digits[:int(0.9 * n_samples)]

X_digits_test = X_digits[int(0.9 * n_samples):]
y_digits_test = y_digits[int(0.9 * n_samples):]

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_digits_train, y_digits_train)

predictions = rf.predict(X_digits_test)

print(y_digits_test)
print(predictions)

print('rf score:', rf.score(X_digits_test, y_digits_test))


print('RANDOM FOREST REGRESSION')

regr = RandomForestRegressor(n_estimators=100)
regr.fit(diabetes_X_train, diabetes_y_train)

# The mean square error
mse = np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
score = regr.score(diabetes_X_test, diabetes_y_test)

print('mse:', mse, 'score:', score)

print('shapes:')
print(diabetes_X_train.shape)
print(diabetes_y_train.shape)
