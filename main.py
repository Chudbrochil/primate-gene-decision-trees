#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Tristin Glunt / Anthony Galczak - tglunt@unm.edu - agalczak@unm.edu
# CS529 - Project 1 - Decision Trees
# This project is the first programming assignment in graduate level Machine Learning.
# It involves implementing a decision tree and its logic from scratch.
# We implemented the algorithms for entropy, gni index, information gained,
# chi square, and ID3 decision trees.

"""
Created on Fri Aug 24 14:38:29 2018

@author: Tristin G. & Anthony G.
"""

import pandas as pd
import math
import numpy as np
import decision_tree as dt
import dt_math as dt_math
import queue
import copy
import argparse
import random

# Possible things to do:
# Capture base cases in ID3 into a method.
# Try to use partition size of 2, i.e. different sized features
# Implement random forests! Decent amount of work.


# TODO:
# When processing data, figure out which classification is most ubiqituous (most_common_class)
# and then if data is empty, return this classification




# These can be set globally to change what we are using for confidence level
# and whether we are using entropy or gni_index
confidence_level = 0.90
is_entropy = True

def main():
    global confidence_level
    global purity_method

    # Parsing command line arguments such as confidence_level and the file name for the training file.
    parser = argparse.ArgumentParser(description='ID3 Decision Tree Algorithm')
    parser.add_argument('-c', dest='confidence_level', type=float, action='store', nargs='?',
                        default=0.95, help='Confidence level of chi square. Acceptable values are 0.90, 0.95, 0.99, 0')
    parser.add_argument('-p', dest='purity_method', type=str, action='store', nargs='?',
                        default="entropy", help='Which purity method do you want. Acceptable values are entropy, ent, gni.')
    parser.add_argument('-t', dest='training_file', type=str, action='store', nargs='?',
                        default="training.csv", help='Specify the training file you want to use. Default is \"training.csv\"')
    parser.add_argument('-r', dest='testing_file', type=str, action='store', nargs='?',
                        default="testing.csv", help='Specify the testing file you want to use. Default is \"testing.csv\"')
    parser.add_argument('-o', dest="output_file", type=str, action='store', nargs='?',
                        default="output.csv", help='Specify where you want your output of classifications to go. Default is \"output.csv\"')

    args = parser.parse_args()

    confidence_level = args.confidence_level

    if confidence_level not in [0.90, 0.95, 0.99, 0]:
        print("Improper confidence level specified. Using confidence of 0.95.")
        confidence_level = 0.95

    impurity_string = (args.purity_method).lower()

    # Using the argument to select whether we are using gni or entropy
    if impurity_string == "ent" or impurity_string == "entropy":
        is_entropy = True
    elif impurity_string == "gni":
        is_entropy = False

    # Initialized variables. Could be brought in via CLI options
    partition_size = 1
    #decision_tree = train(args.training_file, partition_size)
    #test(decision_tree, args.testing_file, partition_size, args.output_file)

    list_of_decision_trees = train_rf(args.training_file, partition_size)


def train_rf(training_file_name, partition_size):

    num_of_trees = 10
    list_of_data = []
    list_of_data_features_split = []
    list_of_features = []
    list_of_decision_trees = []

    data = load_file(training_file_name)
    data_features_split = split_features(data, partition_size)
    feature_objects = create_features(data_features_split)
    #print(data)

    # Gathering random sets of data and features
    for x in range(num_of_trees):
        num_of_elements = random.randint(0, 400) + 400 # range from 400-800 out of 2000
        np.random.shuffle(data_features_split) #shuffle data so range of 400-800 is always different data

        #random features
        num_of_features = random.randint(0, 60)
        np.random.shuffle(feature_objects)

        #print(data)
        # random_dataset = []
        # random_features = []
        #
        # for y in range(num_of_elements):
        #     #print(data[y])
        #     random_dataset.append(data_features_split[y, :])
        #
        # for y in range(num_of_features):
        #     random_features.append(feature_objects[y])

        # print("Shape of random data: " + str(np.array(random_dataset).shape))
        # print("Shape of random features: " + str(np.array(random_features).shape))

        #the shuffled data for each tree, will have as many shuffled sets of data as there are trees
        list_of_data.append(data_features_split[:num_of_elements, :])
        list_of_features.append(feature_objects[:num_of_features])

    print("Shape of lists of data: " + str(np.array(list_of_data).shape))
    print("Shape of lists of features: " + str(np.array(list_of_features).shape))

    # TODO: Not randomizing the features in the first pass....
    # I will need to partially re-write split_features to take a random list
    # instead of iterating over all values


    """ not sure about this now
    for dataset in list_of_data:
        #print(dataset)
        data_features_split = split_features(dataset, partition_size)
        list_of_data_features_split.append(data_features_split)

    for data_feat_split in list_of_data_features_split:
        feature_objects = create_features(data_feat_split)
        list_of_feature_objects.append(feature_objects)

    """

    print(list_of_data[0])

    #for each tree, get the list of classifictions and pass ID3 a random subset of data and features
    for x in range(num_of_trees):
        list_of_classes = get_classifications(list_of_data[x][:,-1:])
        decision_tree = ID3(list_of_data[x], list_of_classes, list_of_features[x])
        list_of_decision_trees.append(decision_tree)

    return list_of_decision_trees


# train()
# Collection method for building an ID3 tree with training data.
def train(training_file_name, partition_size):

    # Printing some details about what we are training on
    print("Confidence level: %f" % confidence_level)
    print("Impurity method: %s" % ("entropy" if is_entropy else "gni_index"))
    print("Loading file: %s" % training_file_name)
    data = load_file(training_file_name)

    data_features_split = split_features(data, partition_size)
    feature_objects = create_features(data_features_split)
    list_of_classes = get_classifications(data_features_split[:,-1:])
    decision_tree = ID3(data_features_split[:, :], list_of_classes, feature_objects)

    return decision_tree


# test()
# Collection method for classifying testing data against our decision tree.
def test(decision_tree, testing_file_name, partition_size, output_file_name):
    print("Loading file: %s" % testing_file_name)
    testing_data = load_file(testing_file_name, None)
    test_features_split = split_features(testing_data, partition_size, False)
    testing_predictions = predict(decision_tree, test_features_split[:,:], 2001)

    print("Outputting to file: %s" % output_file_name)
    output_predictions(testing_predictions, output_file_name)


# load_file()
# Loads a particular csv file and returns it as a pandas data frame.
def load_file(file_name, header_size = 1):

    file = pd.read_csv(file_name, header = header_size)
    data = file.values
    return data


# split_features()
# Splits the string of features every nth character denoted by partition_size.
# Either returns a list of features or a list of features plus classifications.
def split_features(data, partition_size = 1, is_training = True):
    features = data[:, 1]
    matrix_of_features = []

    for sequence in features:
        split_sequence = [sequence[i:i+partition_size] for i in range(0, len(sequence), partition_size)]
        matrix_of_features.append(split_sequence)

    # Concatenation of features and the classifications
    if is_training == True:
        return np.c_[matrix_of_features, data[:, 2]]
    # If we aren't training, then we won't have classifications to return
    else:
        return np.c_[matrix_of_features]


# create_features()
# Goes through all of our training data and creates feature and branch objects
# corresponding to our features and their child values (A,C,G,T, etc.).
def create_features(data_features_split):
    list_of_features = []

    #go through each feature in data
    for i in range(0, data_features_split.shape[1]-1):
        feature = dt.Feature(i, [])

        # For every entry of data, add a branch to a given feature if it isn't
        # already in the list of branches
        for example in data_features_split:
            example_value = example[i]
            branch_list = feature.get_branches()
            found_example_value = False

            for branch in branch_list:
                if branch.get_branch_name() is example_value:
                    found_example_value = True
                    break

            if found_example_value == False:
                feature.add_branch(example_value)

        #after going through all of the examples, add feature object to list
        list_of_features.append(feature)

    return list_of_features


# get_classifications()
# Obtaining the classifications from our data. For the DNA data, should be ["IE", "EI", "N"]
def get_classifications(class_list):
    classes = set()
    for list in class_list:
        classes.add(list[0])

    list_of_classes = []
    for element in classes:
        list_of_classes.append(element)

    return list_of_classes


# TODO:(Anthony) Read this in-depth and make some comments.
# "examples" is the actual data, "target_attribute" is the classifications, "attributes" are list of features
def ID3(data_features_split, list_of_classes, feature_objects):

    data_features_split_copy = copy.deepcopy(data_features_split)
    feature_objects_copy = copy.deepcopy(feature_objects)

    # If all of the remaining examples have the same classification, return that classification
    # This is the "base case"
    initial_classification = "None"
    found_different = False
    for example in data_features_split_copy:

        classification = example[-1:]

        if initial_classification == "None":
            initial_classification = classification
        elif classification != initial_classification:
            found_different = True
            break

    if found_different is False:
        #print("Leaf value: " + str(classification))
        return classification

    #if there are no more features to look at, return with a leaf of the most common class
    #this is really ugly, but since we're not removing things for the list, not much else to do
    all_features_used = True
    for feature in feature_objects_copy:
        if feature != "None":
            all_features_used = False
            break

    if all_features_used:
        print("All features have been used, returning most common class")
        most_common_class = dt_math.determine_class_totals(data_features_split_copy, list_of_classes, True)
        return most_common_class

    # "The attribute from Attributes that best* classifies Examples"
    highest_ig_feature_index, highest_ig_num = get_highest_ig_feat(data_features_split_copy, feature_objects_copy, list_of_classes)

    current_feature = feature_objects_copy[highest_ig_feature_index]

    #determine if this feature will be of statistical benefit using chi-square
    feature_is_beneficial = dt_math.chi_square_test(data_features_split, current_feature, list_of_classes, confidence_level)

    #if the feature is not beneficial, return a leaf node of the most popular class
    if not feature_is_beneficial:
        most_common_class = dt_math.determine_class_totals(data_features_split, list_of_classes, True)
        return most_common_class

    node = feature_objects_copy[highest_ig_feature_index]
    feature_objects_copy[highest_ig_feature_index] = "None"

    # For every possible branch(value). Should look like {A, C, G, T}
    for branch in node.get_branches():
        # Gathering all examples that match this branch value, returns a numpy matrix
        subset_data_feature_match = np.array(dt_math.get_example_matching_value(data_features_split_copy, branch.get_branch_name(), node))
        #print("Shape of branch " + str(branch.get_branch_name()) + ":" + str(subset_data_feature_match.shape) + ", parent id: " + str(node.feature_index))

        # If the examples list is empty(ie., there are no examples left that have this value after trimming so many subsets)
        if subset_data_feature_match.shape[0] == 0:
            most_common_class = dt_math.determine_class_totals(data_features_split_copy, list_of_classes, True)
            branch.add_child_feature(most_common_class)
        else:
            # Recurse
            #this is being applied to the very first instance of feature objects in the first call when feature 29 is parent
            child_feature = ID3(subset_data_feature_match, list_of_classes, feature_objects_copy)
            branch.add_child_feature(child_feature)

    return node


# TODO:(Tristin) Comment this/clean it up
#i couldn't do this above because of the examples.... frustratingly ugly... FIXXXXX
def recursive_prediction_traversal(single_example, node, data_index):
    current_feature_data_value = single_example[node.feature_index]

    for branch in node.get_branches():
        if current_feature_data_value == branch.branch_value:
            if type(branch.child_feature) is not dt.Feature:
                if type(branch.child_feature) is not str:
                    temp_prediction = branch.child_feature[0]
                else:
                    temp_prediction = branch.child_feature

                tuple_prediction = (temp_prediction, data_index)
                return tuple_prediction
            #move on to next feature if values matched by child is a feature
            else:
                node = branch.child_feature
                return recursive_prediction_traversal(single_example, node, data_index)


# get_highest_ig_feat()
# Obtaining the highest information gain feature index from the remaining list of features
def get_highest_ig_feat(data_features_split, feature_objects, list_of_classes):

    list_of_igs = []

    # Getting how many characters long each example is
    length_of_data = data_features_split.shape[1] - 1 #TODO: Do we want to expand this to n-grams?

    highest_ig_num = 0.0
    highest_ig_index = -1

    #TODO changed range from length_of_data -> len(feature_objects) as we won't always have 60 features with RF

    for feature_index in range(len(feature_objects)):
        if feature_objects[feature_index] != "None":
            info_gained_entropy = dt_math.gain(data_features_split, feature_objects[feature_index], list_of_classes, is_entropy)

            # Getting the highest info gained feature
            if info_gained_entropy > highest_ig_num:
                highest_ig_num = info_gained_entropy
                highest_ig_index = feature_index

    #if highest info gain is 0, return arbitrary index
    if highest_ig_num == 0:
        for feature_index in range(length_of_data):
            if feature_objects[feature_index] != "None":
                print("Highest_ig_num was 0, returning random feature index: " + str(feature_index))
                highest_ig_index = feature_index
                break

    # Outputting the index of the feature that has the highest info gained
    return highest_ig_index, highest_ig_num


# predict()
# Traverses the decision tree that we made from training in order to classify
# new data. Returns a full list of predictions (classifications) for a set of data.
def predict(decision_tree, data, data_index):
    predictions = []
    node = decision_tree

    for example in data:
        #get this current features value at the index of the current node of the tree
        current_feature_data_value = example[node.feature_index]
        #go through the possible values for this feature
        for branch in node.get_branches():
            #if the current value of the feature in this example is equal to a branches value
            if current_feature_data_value == branch.branch_value:
                #if the child feature of this branch is a leaf
                if type(branch.child_feature) is not dt.Feature:
                    if type(branch.child_feature) is not str:
                        temp_prediction = branch.child_feature[0]
                    else:
                        temp_prediction = branch.child_feature

                    tuple_prediction = (temp_prediction, data_index)
                    predictions.append(tuple_prediction)
                    node = decision_tree
                    data_index += 1
                    break
                #if the child feature is not a leaf
                else:
                    node = branch.child_feature
                    recursive_prediction = recursive_prediction_traversal(example, node, data_index)

                    # TODO: Bug is exposed here when we have no confidence level. Must fix.
                    if recursive_prediction == None:
                        print(data_index)
                        traverse_tree(node)
                        #print(decision_tree)

                    predictions.append(recursive_prediction)
                    node = decision_tree
                    data_index += 1
                    break

    return predictions


# output_predictions()
# Takes our predictions and outputs them to a file for eventual submission.
def output_predictions(predictions, file_name):
    file = open(file_name, "w")
    file.write("ID,Class\n")
    for tuple in predictions:
        #print(tuple)
        file.write(str(tuple[1]) + "," + str(tuple[0]) + "\n")

    file.close()


if __name__ == "__main__":
    main()


# Helper/Debug Methods

# traverse_tree()
# BFS-like algorithm to go through decision tree. This is used purely for
# debugging purposes to see how the tree is being built/what it looks like.
def traverse_tree(decision_tree):
    q = queue.Queue()

    #insert root feature into queue
    q.put(decision_tree)

    while not q.empty():
        v = q.get()

        if type(v) is dt.Feature:
            print("\nFeature: " + str(v.feature_index))

            for branch in v.get_branches():
                #print(branch.get_branch_name() + " ", end='')
                q.put(branch)
            #print("")
        elif type(v) is dt.Branch:
            print("\nValue:" + str(v.get_branch_name()))

            if type(v.child_feature) is str or type(v.child_feature) is np.ndarray:
                print("---> Leaf: " + str(v.child_feature))
            elif type(v.child_feature) is dt.Feature:
                print("---> Child Feature: " + str(v.child_feature.feature_index))

            q.put(v.child_feature)
        #elif type(v) is str:
            #print("Leaf: " + str(v))


# run_validation_dataset()
# This method is useful for testing against a known validation dataset instead of
# depending upon verifying our testing dataset against Kaggle.
def run_validation_dataset(data_features_split, list_of_classes, feature_objects, start_index):
    decision_tree = ID3(data_features_split[:1598, :], list_of_classes, feature_objects)

    predictions = predict(decision_tree, data_features_split[1598:, :], 1598)

    correct = 0
    wrong = 0
    for num in range(400):#data_features_split[1598:, :]:
        actual_class = data_features_split[1598 + num, -1]
        predicted_class = predictions[num][0]

        if actual_class == predicted_class:
             correct += 1
        else:
            wrong += 1

    print("Accuracy: %f" % (correct / (correct + wrong)))
