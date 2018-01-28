from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from math import sqrt
import operator
from collections import Counter
from pandas import *
from types import *
from sklearn.model_selection import cross_val_score
#http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
#
#
# # This function computes the accuracy of the algorithms predicted data
def determine_accuracy(test_target, targets_predicted, regression):
     correct = 0
     total = 0
     if regression == False:
         for x in range(len(test_target)):
            if test_target[x] == targets_predicted[x]:
                correct = correct + 1
            total = total + 1
     else:
         for x in range(len(test_target)):
            if targets_predicted[x] - 2.5 <= test_target[x] <= targets_predicted[x] + 2.5:
                correct = correct + 1
            total = total + 1
     percent = (correct * 100) / total

     print("Total correct: (", correct, "/", total, "): ", float("{0:.2f}".format(percent)), "%")
#
# # This function predicts the the outcome of the test data using the
# # Gaussian algorithm
def ExampleKNN(data, target, n_neighbors, regression):
    training_data, test_data, training_target, test_target = train_test_split(data, target, test_size=0.30)
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(training_data, training_target)
    targets_predicted = model.predict(test_data)

    determine_accuracy(test_target, targets_predicted, regression)


class MyKNNModel:
     def __init__(self, training_data, training_target, n_neighbors, regression):
         self.training_data = training_data
         self.training_target = training_target
         self.n_neighbors = n_neighbors
         self.regression = regression
     def predict(self, test_data):
         test_distance_array = {}
         targets_predicted = {}
         # Compare a test data point to each training data point.
         for x in range(len(test_data)):
             for y in range(len(self.training_data)):
                 # Find the distance between one test point to all training points
                 distance = 0
                 count = 0

                 for z in range(len(self.training_data[0])):
                    #print(type(test_data[x,y]))
                    #print(type(self.training_data[y,z]))
                    distance = float(test_data[x,z]) - float(self.training_data[y,z])
                    count = count + sqrt(pow(distance,2))
                 # We need to keep the original indexes in orde to match them to the test targets
                 # even after we sort our distance array
                 test_distance_pair = (y,count)
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
                 neighbor_target_values[n] = self.training_target[neighbor_target_index[n]]

             # Find the most common target value of the neighbors and set that as the predicted target
             if self.regression == False:
                b = Counter(neighbor_target_values)
                targets_predicted[x] = b.most_common(1)[0][1]
             else:
                total = 0
                for z in range(len(neighbor_target_values)):
                    total = total + neighbor_target_values[z]
                targets_predicted[x] = total / len(neighbor_target_values)
         return targets_predicted
#
#

class MyKNNClassifier:
    def __init__(self, n_neighbors = 3):
        self.n_neighbors = n_neighbors


    def fit(self, training_data, training_target, regression):
        model = MyKNNModel(training_data, training_target, self.n_neighbors, regression)
        return model


 # This function predicts the the outcome of the test data using the
 # our Hard Coded algorithm
def MyKNN(data, target, n_neighbors, regression):
    training_data, test_data, training_target, test_target = train_test_split(data, target, test_size=0.30)
    classifier = MyKNNClassifier(n_neighbors)
    model = classifier.fit(training_data, training_target, regression)
    targets_predicted = model.predict(test_data)

    determine_accuracy(test_target, targets_predicted, regression)


def Load_Car(file):
    data = read_csv(file, header=0)
    data.columns = ["Buying","Maint","Doors","Persons","Lug_boot","Safety","Class"]
    buying = ["vhigh", "high", "med", "low"]
    maInt = ["vhigh", "high", "med", "low"]
    doors = ["2", "3", "4", "5more"]
    persons = ["2", "4", "more"]
    lug_boot = ["small", "med", "big"]
    safety = ["low", "med", "high"]
    class_value = ["unacc", "acc", "good", "vgood"]

    for x in range(len(buying)):
        data["Buying"] = data["Buying"].replace(buying[x], x)

    for x in range(len(maInt)):
        data["Maint"] = data["Maint"].replace(maInt[x], x)

    for x in range(len(doors)):
        data["Doors"] = data["Doors"].replace(doors[x], x)

    for x in range(len(persons)):
        data["Persons"] = data["Persons"].replace(persons[x], x)

    for x in range(len(lug_boot)):
        data["Lug_boot"] = data["Lug_boot"].replace(lug_boot[x], x)

    for x in range(len(safety)):
        data["Safety"] = data["Safety"].replace(safety[x], x)

    for x in range(len(class_value)):
        data["Class"] = data["Class"].replace(class_value[x], x)

    data_target = data.as_matrix(columns=["Class"])
    data_array = data.as_matrix(columns=["Buying","Maint","Doors","Persons","Lug_boot","Safety","Class"])

    return data_array, data_target


def Load_Indian(file):
    data = read_csv(file, header=0)
    data.columns = ["Pregnant","Glucose","Blood_Pressure","Skin_Thick","Insulin","BMI","Pedigree","Age","Class"]

    data["Glucose"] = data["Glucose"].replace(0, 120)
    data["Blood_Pressure"] = data["Blood_Pressure"].replace(0, 69.1)
    data["Skin_Thick"] = data["Skin_Thick"].replace(0, 20.5)
    data["Insulin"] = data["Insulin"].replace(0, 79.8)
    data["BMI"] = data["BMI"].replace(0, 32.0)
    data["Age"] = data["Age"].replace(0, 33.2)

    data["Pregnant"] = data["Pregnant"] / 3.4
    data["Glucose"] = data["Glucose"] / 32.0
    data["Blood_Pressure"] = data["Blood_Pressure"] / 19.4
    data["Skin_Thick"] = data["Skin_Thick"] / 16.0
    data["Insulin"] = data["Insulin"] / 115.2
    data["BMI"] = data["BMI"] / 7.9
    data["Pedigree"] = data["Pedigree"] / 0.3
    data["Age"] = data["Age"] / 11.8

    data_target = data.as_matrix(columns=["Class"])
    data_array = data.as_matrix(columns=["Pregnant","Glucose","Blood_Pressure","Skin_Thick","Insulin","BMI","Pedigree","Age"])

    return data_array, data_target


def Load_MPG(file):
    data = read_csv(file, header=0)
    data.columns = ["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Year","Origin","Name"]

    del data["Name"]

    data["Horsepower"] = data["Horsepower"].replace("?", 104).astype(float)

    data["Cylinders"] = data["Cylinders"] / 1.7
    data["Displacement"] = data["Displacement"] / 104.3
    data["Horsepower"] = data["Horsepower"] / 38.5
    data["Weight"] = data["Weight"] / 846.8
    data["Acceleration"] = data["Acceleration"] / 2.8
    data["Year"] = (data["Year"] - 70) / 3.7

    data_target = data.as_matrix(columns=["MPG"])
    data_array = data.as_matrix(columns=["Cylinders","Displacement","Horsepower","Weight","Acceleration","Year","Origin"])

    return data_array, data_target

car_data, car_targets = Load_Car('./car data.csv')
indian_data, indian_targets = Load_Indian('./Indian data.csv')
mpg_data, mpg_targets = Load_MPG('./MPG data.csv')

print("Car: Example and My KNN")
ExampleKNN(car_data, car_targets, 3, False)
MyKNN(car_data, car_targets, 3, False)
print("Indian: Example and My KNN")
ExampleKNN(indian_data, indian_targets, 3, False)
MyKNN(indian_data, indian_targets, 3, False)
print("MPG: My KNN")
#ExampleKNN(mpg_data, mpg_targets, 3, True)
MyKNN(mpg_data, mpg_targets, 3, True)



