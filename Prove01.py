from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html


# This function computes the accuracy of the algorithms predicted data
def determine_accuracy(test_target, targets_predicted):
    correct = 0
    total = 0
    for x in range(len(test_target)):
        if test_target[x] == targets_predicted[x]:
            correct = correct + 1
        total = total + 1
    percent = (correct * 100) / total

    print("Total correct: (", correct, "/", total, "): ", float("{0:.2f}".format(percent)), "%")

# This function predicts the the outcome of the test data using the
# Gaussian algorithm
def Gaussian(training_data, training_target, test_data, test_target):
    classifier = GaussianNB()
    model = classifier.fit(training_data, training_target)
    targets_predicted = model.predict(test_data)

    determine_accuracy(test_target, targets_predicted)


class HardCodedModel:
    def predict(test_data):
        targets_predicted = {}
        for x in range(len(test_data)):
            targets_predicted[x] = 0;
        return targets_predicted


class HardCodedClassifier:

    def fit(self, training_data, training_target):
        return HardCodedModel

# This function predicts the the outcome of the test data using the
# our Hard Coded algorithm
def HardCoded(training_data, training_target, test_data, test_target):
    classifier = HardCodedClassifier()
    model = classifier.fit(training_data, training_target)
    targets_predicted = model.predict(test_data)

    determine_accuracy(test_target, targets_predicted)

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)



training_data, test_data, training_target, test_target = train_test_split(iris.data, iris.target, test_size=0.30)

print("Gaussian algorithm: ")
Gaussian(training_data, training_target, test_data, test_target)
print("Hard Coded Algorithm: ")
HardCoded(training_data, training_target, test_data, test_target)