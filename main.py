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

    simpleID3(data_features_split, list_of_classes, feature_objects)


#TODO:
#Make a function that looks at all features and picks highest information gained one...

# "examples" is the actual data, "target_attribute" is the classifications, "attributes" are list of features
def simpleID3(data_features_split, list_of_classes, feature_objects):

    # "The attribute from Attributes that best* classifies Examples"
    highest_ig_feature_index = get_highest_ig_feat(data_features_split, feature_objects, list_of_classes)

    root = features_objects[highest_ig_feature_index]

    # For every possible branch(value). Should look like {A, C, G, T}
    for branch in root.get_branches():

        # Gathering all examples that match this branch value
        subset_data_feature_match = dt_math.get_example_matching_value(data_features_split, branch, root) # TODO: Change root to make this recursive

        # If the examples list is not empty
        if not subset_data_features_match:
            





# Obtaining the highest information gain feature index from the remaining list of features
def get_highest_ig_feat(data_features_split, feature_objects, list_of_classes):

    list_of_igs = []

    # Getting how many characters long each example is
    length_of_data = data_features_split.shape[1] - 1 #TODO: Do we want to expand this to n-grams?

    highest_ig_num = 0.0
    highest_ig_index = -1

    for feature_index in range(length_of_data):
        info_gained_entropy = dt_math.gain(data_features_split, feature_objects[feature_index], list_of_classes, dt_math.entropy)
        print("Info_gained_num: %f Feature_index: %d" % (info_gained_entropy, feature_index))

        # Getting the highest info gained feature
        if info_gained_entropy > highest_ig_num:
            highest_ig_num = info_gained_entropy
            highest_ig_index = feature_index

    print("Highest_ig_num: %f Highest_ig_index: %d" % (highest_ig_num, highest_ig_index))

    # Outputting the index of the feature that has the highest info gained
    return highest_ig_index


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

def create_features(data_features_split):
    list_of_features = []

    #go through each feature in data
    for i in range(0, data_features_split.shape[1]):
        feature = dt.Feature(i, [])
        #go through each example to determine each value of feature
        for example in data_features_split:
            if example[i] not in feature.get_branches():
                feature.add_branch(example[i])

        #after going through all of the examples, add feature object to list
        list_of_features.append(feature)

    return list_of_features


main()
