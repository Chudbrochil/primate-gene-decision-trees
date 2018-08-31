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

def main():
    data = load_file("training.csv")
    partition_size = 1
    data_features_split = split_features(data, partition_size)
    feature_objects = create_features(data_features_split)
    list_of_classes = get_classifications(data_features_split[:,-1:])

    print("Classifications found: " + str(list_of_classes))

    info_gain_feature0 = dt_math.gain(data_features_split, feature_objects[1], list_of_classes, dt_math.entropy)
    info_gain_gni_feature0 = dt_math.gain(data_features_split, feature_objects[1], list_of_classes, dt_math.gni_index)

    print("Information gain on column 1(feature 1), with entropy: " + str(info_gain_feature0))
    print("Information gain on column 1(feature 1), with gni_index: " + str(info_gain_gni_feature0))

    decision_tree = ID3(data_features_split, list_of_classes, feature_objects, [])

    traverse_tree(decision_tree)

    """ printing decision tree for boolean data """
    # for branch in decision_tree.get_branches():
    #     print(branch.get_branch_name())
    #     print("The child feature is " + str(branch.child_feature))
    #
    #     if type(branch.child_feature) is not str:
    #         for sub_branch in branch.child_feature.get_branches():
    #             print("Child branches: " + str(sub_branch.get_branch_name()))
    #             print("Child feature is " + str(sub_branch.child_feature))

# "examples" is the actual data, "target_attribute" is the classifications, "attributes" are list of features
def ID3(data_features_split, list_of_classes, feature_objects, current_feature_hierarchy):

    #no examples left
    if data_features_split.shape[0] == 0:
        return ""

    # If all of the remaining examples have the same classification, return that classification
    # This is the "base case"
    initial_classification = None
    found_different = False
    for example in data_features_split:

        classification = example[-1:]

        if initial_classification is None:
            initial_classification = classification

        #note: is not DOESNT EQUAL != ... bad bad bug
        elif classification != initial_classification:
            found_different = True
            break

    # So instead of putting a feature as a child to a branch.
    # We are putting a "leaf node", which is really just a string that
    # represents a single classification. i.e. "IE" or "EI" or "N"
    if found_different is False:
        #print("Leaf value: " + str(classification))
        return classification

    #if there are no more features to look at, return with a leaf of the most common class
    #this is really ugly, but since we're not removing things for the list, not much else to do
    all_features_used = True
    for feature in feature_objects:
        if feature is not None:
            all_features_used = False
            break

    #TODO determine the most popular classification
    if all_features_used:
        #print("All features have been used, returning most common class")
        most_common_class = dt_math.determine_class_totals(data_features_split, list_of_classes, True)
        return most_common_class

    # "The attribute from Attributes that best* classifies Examples"
    highest_ig_feature_index, highest_ig_num = get_highest_ig_feat(data_features_split, feature_objects, list_of_classes)
    current_feature_hierarchy.append(highest_ig_feature_index)
    #print("current_feature_hierarchy = " + str(current_feature_hierarchy))

    node = feature_objects[highest_ig_feature_index]

    # For every possible branch(value). Should look like {A, C, G, T}
    for branch in node.get_branches():
        # Gathering all examples that match this branch value, returns a numpy matrix
        subset_data_feature_match = dt_math.get_example_matching_value(data_features_split, branch.get_branch_name(), node) # TODO: Change root to make this recursive

        # If the examples list is empty
        if subset_data_feature_match.shape[0] == 0:
            # We found an "A" in column 29, all the other examples aren't "A". We would need to loop over
            # all examples and return the most common classification. (IE, EI, N)
            most_common_class = dt_math.determine_class_totals(data_features_split, list_of_classes, True)
            branch.add_child_feature(most_common_class)
            current_feature_hierarchy.append(most_common_class)
        else:
            # Recurse
            feature_objects[highest_ig_feature_index] = None
            child_feature = ID3(subset_data_feature_match, list_of_classes, feature_objects, current_feature_hierarchy)
            #child feature is going to be a classification, not a feature!
            if type(child_feature) is str:
                #print("Leaf value was returned: " + str(child_feature))
                current_feature_hierarchy.append(child_feature)

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
        if feature_objects[feature_index] is not None:
            info_gained_entropy = dt_math.gain(data_features_split, feature_objects[feature_index], list_of_classes, dt_math.entropy)
            print("Info_gained_num: %f Feature_index: %d" % (info_gained_entropy, feature_index))

            # Getting the highest info gained feature
            if info_gained_entropy > highest_ig_num:
                highest_ig_num = info_gained_entropy
                highest_ig_index = feature_index

    print("Highest_ig_num: %f Highest_ig_index: %d" % (highest_ig_num, highest_ig_index))

    #if highest info gain is 0, return arbitrary index
    if highest_ig_num == 0:
        for feature_index in range(length_of_data):
            if feature_objects[feature_index] is not None:
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
def split_features(data, partition_size = 1):
    features = data[:, 1]
    matrix_of_features = []

    for sequence in features:
        split_sequence = [sequence[i:i+partition_size] for i in range(0, len(sequence), partition_size)]
        matrix_of_features.append(split_sequence)

    # Concatenation of features and the "output". Output also known as labels
    data_features_split = np.c_[matrix_of_features, data[:, 2]]

    return data_features_split



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
            print("\nFeature: " + str(v.feature_index))

            for branch in v.get_branches():
                print(branch.get_branch_name() + " ", end='')
                q.put(branch)
            print("")
        elif type(v) is dt.Branch:
            print("\nValue:" + str(v.get_branch_name()))

            if type(v.child_feature) is str or type(v.child_feature) is np.ndarray:
                print("---> Leaf: " + str(v.child_feature))
            elif type(v.child_feature) is dt.Feature:
                print("---> Child Feature: " + str(v.child_feature.feature_index))

            q.put(v.child_feature)
        elif type(v) is str:
            print("Leaf: " + str(v))

main()
