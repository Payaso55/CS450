from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
import operator
from collections import Counter
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
def ExampleKNN(training_data, training_target, test_data, test_target):
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(training_data, training_target)
    targets_predicted = model.predict(test_data)

    determine_accuracy(test_target, targets_predicted)


class MyKNNModel:
    def __init__(self, training_data, training_target, n_neighbors):
        self.training_data = training_data
        self.training_target = training_target
        self.n_neighbors = n_neighbors
    def predict(self, test_data):
        test_distance_array = {}
        targets_predicted = {}
        # Compare a test data point to each training data point.
        for x in range(len(test_data)):
            for y in range(len(training_data)):
                # Find the distance between one test point to all training points
                distance0 = test_data[x,0] - training_data[y,0]
                distance1 = test_data[x,1] - training_data[y,1]
                distance2 = test_data[x,2] - training_data[y,2]
                distance3 = test_data[x,3] - training_data[y,3]
                total_distance = sqrt(pow(distance0, 2) + pow(distance1, 2) + pow(distance2, 2) + pow(distance3, 2))
                # We need to keep the original indexes in order to match them to the test targets
                # even after we sort our distance array
                test_distance_pair = (y,total_distance)
                test_distance_array[y] = test_distance_pair


            sorted_distances = sorted(test_distance_array.values(), key=operator.itemgetter(1), reverse=False)
            neighbor_target_index = {}
            # now that the distance array is sorted, we can find the array indexes for the nearest neighbors
            for z in range(self.n_neighbors):
                neighbor_target_index[z] = sorted_distances[z][0]

            # Now we have the index values of the nearest neighbors
            neighbor_target_values = {}
            # Collect values of nearest neighbors
            for n in range(self.n_neighbors):
                neighbor_target_values[n] = training_target[neighbor_target_index[n]]

            # Find the most common target value of the neighbors and set that as the predicted target
            b = Counter(neighbor_target_values)
            targets_predicted[x] = b.most_common(1)[0][1]
        return targets_predicted


class MyKNNClassifier:
    def __init__(self, n_neighbors = 3):
        self.n_neighbors = n_neighbors

    def fit(self, training_data, training_target):
        model = MyKNNModel(training_data, training_target, self.n_neighbors)
        return model

# This function predicts the the outcome of the test data using the
# our Hard Coded algorithm
def MyKNN(training_data, training_target, test_data, test_target, n_neighbors):
    classifier = MyKNNClassifier(n_neighbors)
    model = classifier.fit(training_data, training_target)
    targets_predicted = model.predict(test_data)

    determine_accuracy(test_target, targets_predicted)

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
#print(iris.target)

# Show the actual target names that correspond to each number
#print(iris.target_names)



training_data, test_data, training_target, test_target = train_test_split(iris.data, iris.target, test_size=0.30)


print("Example KNN algorithm: ")
ExampleKNN(training_data, training_target, test_data, test_target)
print("My KNN algorithm: ")
MyKNN(training_data, training_target, test_data, test_target, 3)