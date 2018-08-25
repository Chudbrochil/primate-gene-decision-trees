#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:38:29 2018

@author: anthony
"""

import pandas as pd
import math

# NOTE: Z = 1, Y = 0
def main():
    data = loadFile("boolean.csv")

    attributes = data[1]
    targetAttributes = data[2]

    listOfAttributes = splitAttributes(attributes)

    makeID3(attributes, targetAttributes)



def loadFile(fileName):
    data = []

    file = pd.read_csv(fileName)

    data.append(file.iloc[:, 0])
    data.append(file.iloc[:, 1])
    data.append(file.iloc[:, 2])

    for line in data:
        print(line)

    return data

#TODO:
#Make a function that looks at all features and picks highest information gained one...
""" Caclulates entropy on current set of examples.
    Used for the entire dataset and each value of a feature.
    examples - the dataset containing features and labels
    classes - list of possible classifications of an example
    based on eq. 3.3 pg. 59 of Machine Learning by Tom Mitchell
 """
def entropy(examples, classes):

    entropy = 0

    #down and dirty dictionaries for now
    labels = {}
    label_totals = {}

    for i in range(0, len(classes)):
        labels["class" + str(i)] = classes[i]

    for i in range(0, len(classes)):
        label_totals["class" + str(i)] = 0

    #go through each example (need to determine if column 2 is going to be output)
    for example in examples:
        #check which class it is, once match found, break
        for i in range(0, len(classes)):
            if(example[2] == labels["class" + str(i)]):
                #if the output class for this example matches, add one to total classes
                label_totals["class" + str(i)] = label_totals["class" + str(i)] + 1
                break

    #calculate entropy now that proportions are known (p_i)
    for i in range(0, len(listOfValues)):
        p_i = labels["class" + str(i)] / label_totals["class" + str(i)]
        entropy = entropy - p_i * math.log(p_i, 2)

    return entropy

""" Gain calculates the information gain of each feature on current passed in examples """
def gain(examples, features):
    return 1




def splitAttributes(attributes):


    list0 = []
    list1 = []

    for row in attributes:
        list0.append(row[0])
        list1.append(row[1])

    listOfAttributes = []
    listOfAttributes.append(list0)
    listOfAttributes.append(list1)

    print(listOfAttributes)

    return listOfAttributes


#def id3(attributes, targetAttributes):





main()
