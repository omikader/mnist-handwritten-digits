from mnist import MNIST
from sklearn import svm
from sklearn import metrics

# Load MNIST data
mndata = MNIST('../data/raw')
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

# Create one-vs-rest linear SVM classifier and fit to training data
clf = svm.LinearSVC(C=1)
clf.fit(X_train, y_train)

# Run linear classifier on testing data
predictions = clf.predict(X_test)

# Determine training accuracy
train_accuracy = metrics.accuracy_score(y_train.tolist(), clf.predict(X_train))
print('Training accuracy: %0.2f%%' % (train_accuracy*100))

# Determine testing accuracy
test_accuracy = metrics.accuracy_score(y_test.tolist(), predictions)
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

# Determine confusion matrix
c_matrix = metrics.confusion_matrix(y_test.tolist(), predictions)
print('Confusion matrix:\n', c_matrix)
