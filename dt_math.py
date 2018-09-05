import math
import numpy as np

""" Caclulates entropy on current set of examples.
    Used for the entire dataset and each value of a feature.
    examples - the dataset containing features and labels.
    classes - list of possible classifications of an example.
    based on eq. 3.3 pg. 59 of Machine Learning by Tom Mitchell
 """
def entropy(examples, classes):

    entropy = 0
    total_examples = len(examples)
    numOfClasses = len(classes)

    if total_examples == 0:
        #print("total examples = 0")
        return 0

    #determine each unique class and count how many times each is in set of examples
    label_totals = determine_class_totals(examples, classes)

    #calculate entropy now that proportions are known (p_i)
    for i in range(numOfClasses):
        p_i = label_totals["class" + str(i)] / total_examples
        if p_i != 0:
            entropy = entropy - (p_i * math.log(p_i, 2))

    return entropy

def gni_index(examples, classes):
    gni = 1

    #determine each unique class and count how many times each is in set of examples
    label_totals = determine_class_totals(examples, classes)

    total_examples = len(examples)
    numOfClasses = len(classes)

    #calculate entropy now that proportions are known (p_i)
    for i in range(numOfClasses):
        p_i = label_totals["class" + str(i)] / total_examples
        if p_i != 0:
            gni = gni - (p_i ** 2)

    return gni

#TODO can easily replace dictionaries with list if preferred
def determine_class_totals(examples, classes, get_most_common_class = False):
    labels = {}
    label_totals = {}

    numOfClasses = len(classes)

    for i in range(numOfClasses):
        labels["class" + str(i)] = classes[i]

    for i in range(numOfClasses):
        label_totals["class" + str(i)] = 0

    #go through each example
    for example in examples:
        #go through each class for current example, once match found, break
        for i in range(numOfClasses):
            #the output will always be the last element of the example
            if(example[-1:] == labels["class" + str(i)]):
                #if the output class for this example matches, add one to total classes
                label_totals["class" + str(i)] = label_totals["class" + str(i)] + 1
                break

    most_common_class_amount = 0
    most_common_class = ""

    if get_most_common_class:
        for i in range(numOfClasses):
            if label_totals["class" + str(i)] > most_common_class_amount:
                most_common_class_amount = label_totals["class" + str(i)]
                most_common_class = labels["class" + str(i)]
        return most_common_class
    else:
        return label_totals

""" Gain calculates the information gain of each feature on current passed in examples
   gain(data, A) -> will look through values Y & Z for feature A.
   feature - an object... with this features column index in the
   dataset and needs to have a list of it's values....
"""
def gain(examples, feature, classes, impurity_func):
    #determine impurity of entire dataset
    gain = impurity_func(examples, classes)

    #print("entropy of entire set: " + str(gain))

    #determine impurity for each value in feature, sum together and return info gain
    for branch in feature.get_branches():

        #return subset of examples that only have this value
        subset_of_example = get_example_matching_value(examples, branch.get_branch_name(), feature)

        # if(examples.shape[0] == 22):
        #     print(subset_of_example.shape)
        #print("shape of subset of a value for this feature" + str(subset_of_example.shape))

        total_subset_of_value = len(subset_of_example)

        #math of information gain
        proportion_of_subset = total_subset_of_value / len(examples) * 1.0
        subset_entropy = proportion_of_subset * impurity_func(subset_of_example, classes)
        gain = gain - subset_entropy

    return gain

def get_example_matching_value(examples, branch_name, feature):
    subset_of_value = []

    #go through each example
    for example in examples:
        #go through only feature passed into gain and check value
        if(example[feature.feature_index] is branch_name):
            subset_of_value.append(example)

    #convert subset to np matrix so we can use .shape
    subset_of_value = np.array(subset_of_value)
    return subset_of_value
