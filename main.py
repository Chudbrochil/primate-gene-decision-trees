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

# NOTE: Z = 1, Y = 0
def main():
    data = loadFile("boolean.csv")
    print(data)

    data_features_split = splitFeatures(data)
    print(data_features_split)

    feature_objects = detect_features(data_features_split)

    for i in range(0, len(feature_objects)):
        print(feature_objects[i].featureIndex)

    #only 2 classes, 1 (OR be true), 0 (OR be false)
    #totalEntropy = entropy(data_features_split, [1, 0])
    #print(totalEntropy)

    info_gain_feature0 = gain(data_features_split, feature_objects[0])
    print(info_gain_feature0)

    # listOfAttributes = splitAttributes(attributes)
    #
    # makeID3(attributes, targetAttributes)

def loadFile(fileName):

    file = pd.read_csv(fileName)
    data = file.values

    return data

#TODO:
#Make a function that looks at all features and picks highest information gained one...
""" Caclulates entropy on current set of examples.
    Used for the entire dataset and each value of a feature.
    examples - the dataset containing features and labels.
    classes - list of possible classifications of an example.
    based on eq. 3.3 pg. 59 of Machine Learning by Tom Mitchell
 """
def entropy(examples, classes):

    entropy = 0

    #down and dirty dictionaries for now
    labels = {}
    label_totals = {}
    total_examples = len(examples)

    numOfClasses = len(classes)

    for i in range(numOfClasses):
        labels["class" + str(i)] = classes[i]

    for i in range(numOfClasses):
        label_totals["class" + str(i)] = 0

    #go through each example
    for example in examples:
        print("going through example" + str(example))
        #go through each class for current example, once match found, break
        for i in range(numOfClasses):
            #the output will always be the last element of the example
            if(example[-1:] == labels["class" + str(i)]):
                #if the output class for this example matches, add one to total classes
                label_totals["class" + str(i)] = label_totals["class" + str(i)] + 1
                break

    #calculate entropy now that proportions are known (p_i)
    for i in range(numOfClasses):
        print("current label " + str(labels["class" + str(i)]))
        p_i = label_totals["class" + str(i)] / total_examples
        if p_i == 0:
            entropy = entropy - 0
        else:
            entropy = entropy - p_i * math.log(p_i, 2)

    return entropy

""" Gain calculates the information gain of each feature on current passed in examples
   gain(data, A) -> will look through values Y & Z for feature A.
   feature - an object... with this features column index in the
   dataset and needs to have a list of it's values....
"""
#TODO DETERMINE IF FEATURE BEING PASSED INTO GAIN SHOULD BE AN OBJECT
def gain(examples, feature):

    #gain step 1, take entropy of all examples
    gain = entropy(examples, [0, 1])

    #step 1.5, make examles into a dictionary
    # dictionary_examples = convertExamplesToDictionary(examples)

    #gain step2, sum entropies of each value for current feature
    for value in feature.values:
        print(value)
        subset_of_value = valuesInExamples(examples, value, feature)
        total_subset_of_value = len(subset_of_value)
        print("total subset of value " + str(total_subset_of_value))
        proportion_of_subset = total_subset_of_value / len(examples)
        gain = gain + (proportion_of_subset * entropy(subset_of_value, [0,1]))

    return gain

def valuesInExamples(examples, value, feature):
    subset_of_value = []
    #go through each example
    for example in examples:
        #go through only feature passed into gain and check value
        if(example[feature.featureIndex] == value):
            subset_of_value.append(example)

    return subset_of_value

# def convertExamplesToDictionary(examples):
#     dictionary_examples = {}
#
#     for example in examples:
#         for i in range(0, len(example)):
#             dictionary["feature" + str(i)]


def splitFeatures(data):
    features = data[:, 1]
    matrix_of_features = []

    for feature in features:
        split_feature = list(feature) #convert string in column to characters
        matrix_of_features.append(split_feature)

    data_features_split = np.c_[matrix_of_features, data[:, 2]]

    return data_features_split

def detect_features(data_features_split):
    list_of_features = []
    #go through each feature in data
    for i in range(0, data_features_split.shape[1]-1):
        feature = dt.Feature(i, [])
        index_last_value = -1
        #go through each example to determine each type of feature
        for example in data_features_split:
            #if no values currently
            if not feature.values:
                feature.addValue(example[i])
                index_last_value += 1
            #else if current value hasn't been seen before, add it to list
            elif example[i] != feature.values[index_last_value]:
                feature.addValue(example[i])
                index_last_value += 1
        #after going through all of the examples, add feature object to list
        list_of_features.append(feature)

    return list_of_features




#def id3(attributes, targetAttributes):

main()
