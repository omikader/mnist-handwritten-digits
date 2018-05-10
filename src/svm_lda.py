from mnist import MNIST
from sklearn import svm
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load MNIST data
mndata = MNIST('../data/raw')
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

# Reduce training and testing samples to c-1 dimensions
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
X_dim_train = lda.transform(X_train)
X_dim_test = lda.transform(X_test)

# Create one-vs-rest linear SVM classifier and fit to training data
clf = svm.LinearSVC(C=1)
clf.fit(X_dim_train, y_train)

# Run linear classifier on testing data
predictions = clf.predict(X_dim_test)

# Determine training accuracy
train_accuracy = metrics.accuracy_score(y_train.tolist(), clf.predict(X_dim_train))
print('Post-LDA Training accuracy: %0.2f%%' % (train_accuracy*100))

# Determine veracity of the predictions
test_accuracy = metrics.accuracy_score(y_test.tolist(), predictions)
print('Post-LDA Testing accuracy: %0.2f%%' % (test_accuracy*100))

# Determine confusion matrix
c_matrix = metrics.confusion_matrix(y_test.tolist(), predictions)
print('Post-LDA Confusion matrix:\n', c_matrix)
