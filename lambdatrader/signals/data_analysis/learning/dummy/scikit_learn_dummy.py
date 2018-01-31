import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.linear_model import LogisticRegression
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
ax = plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
# plt.show()

data = digits.images.reshape((digits.images.shape[0], -1))


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

logistic = LogisticRegression(C=1e5)

logistic.fit(X_digits_train, y_digits_train)

predictions = logistic.predict(X_digits_test)

print(y_digits_test)
print(predictions)

print('logistic score:', logistic.score(X_digits_test, y_digits_test))

