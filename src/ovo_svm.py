from mnist import MNIST
from sklearn import svm

# Load MNIST data
mndata = MNIST('./data')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

# Create one-vs-one linear SVM classifier and fit to training data
clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
clf.fit(training_images, training_labels)

# Run linear classifier on testing data
predictions = clf.predict(testing_images)

# Determine veracity of the predictions
res = predictions == testing_labels
accuracy = sum(x == True for x in res) / len(testing_images)
