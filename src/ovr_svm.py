from mnist import MNIST
from sklearn import svm

# Load MNIST data
mndata = MNIST('./data')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

# Create one-vs-rest linear SVM classifier and fit to training data
clf = svm.LinearSVC()
clf.fit(training_images, training_labels)

'''
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
'''

# Run linear classifier on testing data
predictions = clf.predict(testing_images)

# Determine veracity of the predictions
res = predictions == testing_labels
accuracy = sum(x == True for x in res) / len(testing_images)
