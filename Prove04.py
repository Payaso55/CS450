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
from sklearn import tree


class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.branches = {}

    def add(self, attribute, attribute_value):
        newNode = Node(attribute)
        self.branches[attribute_value] = newNode

    def display(self):
        print(self.attribute)

    def isLeaf(self):
        if self.branches:
            return False
        else:
            return True

def Calc_entropy(data):
    data_set = unique(data["target"])
    target_count = {}

    for x in range(len(data_set)):
        target_count[data_set[x]] = 0

    for x in data["target"]:
        value = x
        target_count[value] = target_count[value] + 1

    total_entropy = 0

    for x in range(len(data_set)):
        target_count[data_set[x]] = target_count[data_set[x]] / len(data["target"])
        target_count[data_set[x]] = -(target_count[data_set[x]]) * np.log2(target_count[data_set[x]])
        total_entropy = total_entropy + target_count[data_set[x]]

    if total_entropy != 0:
        return total_entropy
    else:
        return 0

class MyDescisionTreeModel:
    def __init__(self, node):
        self.node = node

    def predict(self, test_data):
        columns = ["Buying","Maint","Doors","Persons","Lug_boot","Safety"]
        predicted_targets = []
        for i in range(len(test_data)):
            node = self.node
            while node.isLeaf() != True:
                #print(node.attribute)
                if test_data.iloc[i,columns.index(node.attribute)] in node.branches:
                    node = node.branches[test_data.iloc[i,columns.index(node.attribute)]]
                else:
                    keys = list(node.branches.keys())
                    node = node.branches[keys[0]]
            predicted_targets.append(node.attribute)
        return predicted_targets

class MyDescisionTreeClassifier:
    def build_tree(self, data):
        total_entropy = Calc_entropy(data)
        if total_entropy == 0:
            target = unique(data["target"])
            node = Node(target[0])
            return node
        if len(data.columns) == 1:
            node = Node(data["target"].value_counts().idxmax())
            return node
        information_gain = 0
        previous_info_gain = 0
        key_attribute = ""
        #Calculate information gain from each attribute
        for col in data.columns:
            if col == "target":
                break
            data_unique = unique(data[col])
            attribute_entropy = 0
            for y in data_unique:
                data_subset = data[data[col] == y]
                attribute_entropy = attribute_entropy + Calc_entropy(data_subset)
            information_gain = total_entropy - (attribute_entropy / len(data_unique))
            if information_gain > previous_info_gain:
                previous_info_gain = information_gain
                key_attribute = col

        current_node = Node(key_attribute)
        data_unique = unique(data[key_attribute])

        for attribute_value in data_unique:
            #create a child node for each value of key attribute
            data_subset = data[data[key_attribute] == attribute_value]
            reduced_subset = data_subset.drop(key_attribute, axis= 1)

            node = self.build_tree(reduced_subset)
            current_node.branches[attribute_value] = node

        return current_node

    def fit(self, data):
        Tree_Root = self.build_tree(data)
        model = MyDescisionTreeModel(Tree_Root)
        return model

def determine_accuracy(test_target, targets_predicted):
    correct = 0
    total = 0
    for x in range(len(test_target)):
        if test_target.iloc[x] == targets_predicted[x]:
            correct = correct + 1
        total = total + 1
    percent = (correct * 100) / total

    print("Total correct: (", correct, "/", total, "): ", float("{0:.2f}".format(percent)), "%")

def MyDescisionTree(data):
    training, test = train_test_split(data, test_size=0.30)
    test_data = test.loc[:, test.columns != "target"]
    test_targets = test["target"]
    classifier = MyDescisionTreeClassifier()
    model = classifier.fit(training)
    predicted_targets = model.predict(test_data)
    determine_accuracy(test_targets, predicted_targets)

def ExampleDescisionTree(data):
    training, test = train_test_split(data, test_size=0.30)
    training_data = training.loc[:, training.columns != "target"]
    training_targets = training["target"]
    test_data = test.loc[:, test.columns != "target"]
    test_targets = test["target"]
    classifier = tree.DecisionTreeClassifier()
    model = classifier.fit(training_data,training_targets)
    predicted_targets = model.predict(test_data)
    determine_accuracy(test_targets, predicted_targets)

def Load_Car(file):
    data = read_csv(file, header=0)
    data.columns = ["Buying","Maint","Doors","Persons","Lug_boot","Safety","target"]
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
        data["target"] = data["target"].replace(class_value[x], x)
    return data


car_data = Load_Car('./car data.csv')

classifier = MyDescisionTreeClassifier()
print("My descision tree accuracy:")
MyDescisionTree(car_data)
print("Scikit-learn accuracy")
ExampleDescisionTree(car_data)