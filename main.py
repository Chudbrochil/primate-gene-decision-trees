#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:38:29 2018

@author: anthony
"""

import pandas as pd
import math
import numpy as np
import decision_tree as dt
import dt_math as dt_math
import queue
import copy


def main():

    data = load_file("training.csv")
    partition_size = 1
    data_features_split = split_features(data, partition_size)
    feature_objects = create_features(data_features_split)
    list_of_classes = get_classifications(data_features_split[:,-1:])


    decision_tree = ID3(data_features_split[:, :], list_of_classes, feature_objects)

    # Doing testing data now
    testing_data = load_file("testing.csv")
    test_features_split = split_features(testing_data, partition_size, False)

    testing_predictions = predict(decision_tree, test_features_split[:,:], 1)

    output_data(testing_predictions, "testing-1.csv")


    # VALIDATION IS HERE
    # TODO: This only goes to 1598 because we are doing "validation data"
    # decision_tree = ID3(data_features_split[:1598, :], list_of_classes, feature_objects)
    #
    # predictions = predict(decision_tree, data_features_split[1598:, :], 1598)
    #
    # output_data(predictions, "training-1.csv")
    #
    # correct = 0
    # wrong = 0
    # for num in range(400):#data_features_split[1598:, :]:
    #     actual_class = data_features_split[1598 + num, -1]
    #     predicted_class = predictions[num][0]
    #
    #     if actual_class == predicted_class:
    #          correct += 1
    #     else:
    #         wrong += 1
    #
    # print("Accuracy: %f" % (correct / (correct + wrong)))







def output_data(predictions, file_name):
    file = open(file_name, "w")
    for tuple in predictions:
        file.write(str(tuple[1]) + "," + str(tuple[0]) + "\n")

    file.close()


def predict(decision_tree, data, data_index):
    print(len(data))
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
                    predictions.append(recursive_prediction_traversal(example, node, data_index))
                    node = decision_tree
                    data_index += 1
                    break

    return predictions

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

    if highest_ig_num == 0:
        for example in data_features_split_copy:
            print(example)

    if highest_ig_num == 0:
        print("No more important features")
        most_common_class = dt_math.determine_class_totals(data_features_split_copy, list_of_classes, True)
        return most_common_class

    """
    current_feature = feature_objects[highest_ig_feature_index]
    #determine if this feature will be of statistical benefit using chi-square
    feature_is_beneficial = chi_square_test(data_features_split, current_feature, list_of_classes)

    #if the feature is not beneficial, return a leaf node of the most popular class
    if not feature_is_beneficial:
        most_common_class = dt_math.determine_class_totals(data_features_split, list_of_classes, True)
        return most_common_class

    #print("current_feature_hierarchy = " + str(current_feature_hierarchy))

    """
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

# Obtaining the highest information gain feature index from the remaining list of features
def get_highest_ig_feat(data_features_split, feature_objects, list_of_classes):

    list_of_igs = []

    # Getting how many characters long each example is
    length_of_data = data_features_split.shape[1] - 1 #TODO: Do we want to expand this to n-grams?

    highest_ig_num = 0.0
    highest_ig_index = -1

    for feature_index in range(length_of_data):
        if feature_objects[feature_index] != "None":
            info_gained_entropy = dt_math.gain(data_features_split, feature_objects[feature_index], list_of_classes, dt_math.entropy)
            #print("Info_gained_num: %f Feature_index: %d" % (info_gained_entropy, feature_index))

            # Getting the highest info gained feature
            if info_gained_entropy > highest_ig_num:
                highest_ig_num = info_gained_entropy
                highest_ig_index = feature_index

    #print("Highest_ig_num: %f Highest_ig_index: %d" % (highest_ig_num, highest_ig_index))

    #if highest info gain is 0, return arbitrary index
    if highest_ig_num == 0:
        for feature_index in range(length_of_data):
            if feature_objects[feature_index] != "None":
                print("Highest_ig_num was 0, returning random feature index: " + str(feature_index))
                highest_ig_index = feature_index
                break

    # Outputting the index of the feature that has the highest info gained
    return highest_ig_index, highest_ig_num


def load_file(file_name):

    file = pd.read_csv(file_name)
    data = file.values

    return data

# Obtaining the classifications from our data. For the DNA data, should be ["IE", "EI", "N"]
def get_classifications(class_list):
    classes = set()
    for list in class_list:
        classes.add(list[0])

    list_of_classes = []
    for element in classes:
        list_of_classes.append(element)

    return list_of_classes

""" splits string of features every nth character denoted by partition_size """
def split_features(data, partition_size = 1, is_training = True):
    features = data[:, 1]
    matrix_of_features = []

    for sequence in features:
        split_sequence = [sequence[i:i+partition_size] for i in range(0, len(sequence), partition_size)]
        matrix_of_features.append(split_sequence)

    # Concatenation of features and the "output". Output also known as labels
    if is_training == True:
        return np.c_[matrix_of_features, data[:, 2]]
    else:
        return matrix_of_features



# TODO: Anallyze complexity of the big gnarly loop in this method
def create_features(data_features_split):
    list_of_features = []

    #go through each feature in data
    for i in range(0, data_features_split.shape[1]-1):
        feature = dt.Feature(i, [])
        #go through each example to determine each value of feature

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

            # for branch in feature.get_branches():
            #     if branch.get_branch_name() is not example[i]:
            #         feature.add_branch(branch.get_branch_name())

        #after going through all of the examples, add feature object to list
        list_of_features.append(feature)

    return list_of_features

#a breadth first like algorithm to go through the decision tree
def traverse_tree(decision_tree):
    q = queue.Queue()

    #insert root feature into queue
    q.put(decision_tree)

    while not q.empty():
        v = q.get()

        if type(v) is dt.Feature:
            #print("\nFeature: " + str(v.feature_index))

            for branch in v.get_branches():
                #print(branch.get_branch_name() + " ", end='')
                q.put(branch)
            #print("")
        elif type(v) is dt.Branch:
            # print("\nValue:" + str(v.get_branch_name()))
            #
            # if type(v.child_feature) is str or type(v.child_feature) is np.ndarray:
            #     print("---> Leaf: " + str(v.child_feature))
            # elif type(v.child_feature) is dt.Feature:
            #     print("---> Child Feature: " + str(v.child_feature.feature_index))

            q.put(v.child_feature)
        #elif type(v) is str:
            #print("Leaf: " + str(v))

def chi_square_test(data, current_feature, list_of_classes):
    class_totals = dt_math.determine_class_totals(data, list_of_classes, False)

    """ build table of real and expected values for current feature """
    total_data_size = len(data) #TODO: might have to take only 1st dimension of data

    #build a matrix the dimensions of, (total_values_for_feature, total_classes)
    variable_matrix_real = np.array([[0 for x in range(len(class_totals))] for y in range(len(current_feature.get_branches()))])
    variable_matrix_expected = np.array([[0 for x in range(len(class_totals))] for y in range(len(current_feature.get_branches()))])

    #determine "real values" for each value and class of this feature
    counter = 0
    for branch in current_feature.get_branches():
        #returns subset of data matching the current value of this feature
        subset_data_feature_match = dt_math.get_example_matching_value(data, branch.get_branch_name(), current_feature)

        #returns a dictionary of totals of each class for this value of this feature
        class_totals_for_subvalue = dt_math.determine_class_totals(subset_data_feature_match, list_of_classes, False)

        #fill variable_matrix_real with class totals for this current branch
        for j in range(0, len(class_totals_for_subvalue)):
            variable_matrix_real[counter][j] = class_totals_for_subvalue["class" + str(j)]
        counter += 1

    #calculate expected values
    counter = 0
    for branch in current_feature.get_branches():
        subset_data_feature_match = dt_math.get_example_matching_value(data, branch.get_branch_name(), current_feature)

        total_subset_size = len(subset_data_feature_match)

        class_totals_for_subvalue = dt_math.determine_class_totals(subset_data_feature_match, list_of_classes, False)

        for j in range(len(class_totals_for_subvalue)):
            variable_matrix_expected[counter][j] = total_subset_size * (class_totals["class" + str(j)] / total_data_size)
        counter += 1

    chi_square_value = 0
    """ run chi-square function over built table """
    #for every class compute the different values chi square
    for i in range(len(current_feature.get_branches())):
        for j in range(len(class_totals)):
            if variable_matrix_expected[i][j] == 0:
                continue
            chi_square_value += ((variable_matrix_real[i][j] - variable_matrix_expected[i][j]) ** 2) / variable_matrix_expected[i][j]

    print("Chi-square value: " + str(chi_square_value))

    """ determine if chi-square value if in or out of distrubution """
    degree_of_freedom = (len(list_of_classes) - 1)  * (len(current_feature.get_branches()) - 1)

    critical_value = compute_critical_value(degree_of_freedom, .95)

    if chi_square_value < critical_value:
        return False
    else:
        return True


#TODO: determine how to computer critical value, mainly how loading in Chi-Square table...
def compute_critical_value(degree_of_freedom, confidence_level):
    return 1

main()
