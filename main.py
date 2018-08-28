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
    partition_size = 2
    data_features_split = split_features(data, partition_size)
    feature_objects = create_features(data_features_split)
    list_of_classes = get_classifications(data_features_split[:,-1:])

    print("Classifications found: " + str(list_of_classes))

    info_gain_feature0 = dt_math.gain(data_features_split, feature_objects[1], list_of_classes, dt_math.entropy)
    info_gain_gni_feature0 = dt_math.gain(data_features_split, feature_objects[1], list_of_classes, dt_math.gni_index)

    print("Information gain on column 1(feature 1), with entropy: " + str(info_gain_feature0))
    print("Information gain on column 1(feature 1), with gni_index: " + str(info_gain_gni_feature0))


#TODO:
#Make a function that looks at all features and picks highest information gained one...


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
        print(split_sequence)
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
            #if no values currently
            if not feature.values:
                feature.addValue(example[i])
            #else if current value hasn't been seen before, add it to list
            elif example[i] not in feature.values:
                feature.addValue(example[i])
        #after going through all of the examples, add feature object to list
        list_of_features.append(feature)

    return list_of_features


main()
