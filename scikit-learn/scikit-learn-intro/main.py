from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib

''' loading the digits data set: '''
digits = datasets.load_digits()
print(digits.data)

''' conversion to input and output: '''
X, y = digits.data, digits.target

''' training a linear SVM model: '''
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X[:-1], y[:-1])

''' prediction: '''
yp = clf.predict(X[-1:])
print("prediction: ", yp, "  actual:", y[-1:])

''' saving the model to a file: '''
joblib.dump(clf, 'digits_model.pkl')

''' loading the model from a file: '''
clf2 = joblib.load('digits_model.pkl')

''' prediction from the reloaded model: '''
yp2 = clf2.predict(X[0:1])
print("prediction: ", yp2, "  actual:", y[0:1])
