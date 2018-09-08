# Tristin Glunt / Anthony Galczak - tglunt@unm.edu - agalczak@unm.edu
# CS529 - Project 1 - Decision Trees

import math
import numpy as np

# The first index is degrees of freedom, the key afterwards is the confidence level
# degrees of freedom should be max 10, max values = 6, max classes = 3 (3-1)*(6-1) = 10
chi_square = [0, {0.90 : 2.71, 0.95 : 3.84, 0.99 : 6.63}, {0.90 : 4.61, 0.95 : 5.99, 0.99 : 9.21},
{0.90 : 6.25, 0.95 : 7.81, 0.99 : 11.34}, {0.90 : 7.78, 0.95 : 9.49, 0.99 : 13.28}, {0.90 : 9.24, 0.95 : 11.07, 0.99 : 15.09},
{0.90 : 10.64, 0.95 : 12.59, 0.99 : 16.81}, {0.90 : 12.02, 0.95 : 14.07, 0.99 : 18.48}, {0.90 : 13.36, 0.95 : 15.51, 0.99 : 20.09},
{0.90 : 14.68, 0.95 : 16.92, 0.99 : 21.67}, {0.90 : 15.99, 0.95 : 18.31, 0.99 : 23.21}]
# TODO: Possibly expand this table into a method if we want to use partition size of 2, likely need upto 36 degrees of freedom


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

    if total_examples == 0:
        return 0

    #calculate entropy now that proportions are known (p_i)
    for i in range(numOfClasses):
        p_i = label_totals["class" + str(i)] / total_examples
        if p_i != 0:
            gni = gni - (p_i ** 2)

    return gni


""" Gain calculates the information gain of each feature on current passed in examples
   gain(data, A) -> will look through values Y & Z for feature A.
   feature - an object... with this features column index in the
   dataset and needs to have a list of it's values....
"""
def gain(examples, feature, classes, impurity_func):
    #determine impurity of entire dataset
    gain = impurity_func(examples, classes)

    #determine impurity for each value in feature, sum together and return info gain
    for branch in feature.get_branches():

        #return subset of examples that only have this value
        subset_of_example = get_example_matching_value(examples, branch.get_branch_name(), feature)

        total_subset_of_value = len(subset_of_example)

        #math of information gain
        proportion_of_subset = total_subset_of_value / len(examples) * 1.0
        subset_entropy = proportion_of_subset * impurity_func(subset_of_example, classes)
        gain = gain - subset_entropy

    return gain


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


def chi_square_test(data, current_feature, list_of_classes, confidence_level):
    class_totals = determine_class_totals(data, list_of_classes, False)

    """ build table of real and expected values for current feature """
    total_data_size = len(data) #TODO: might have to take only 1st dimension of data

    #build a matrix the dimensions of, (total_values_for_feature, total_classes)
    variable_matrix_real = np.array([[0 for x in range(len(class_totals))] for y in range(len(current_feature.get_branches()))])
    variable_matrix_expected = np.array([[0 for x in range(len(class_totals))] for y in range(len(current_feature.get_branches()))])

    #determine "real values" for each value and class of this feature
    counter = 0
    for branch in current_feature.get_branches():
        #returns subset of data matching the current value of this feature
        subset_data_feature_match = get_example_matching_value(data, branch.get_branch_name(), current_feature)

        #returns a dictionary of totals of each class for this value of this feature
        class_totals_for_subvalue = determine_class_totals(subset_data_feature_match, list_of_classes, False)

        #fill variable_matrix_real with class totals for this current branch
        for j in range(0, len(class_totals_for_subvalue)):
            variable_matrix_real[counter][j] = class_totals_for_subvalue["class" + str(j)]
        counter += 1

    #calculate expected values
    counter = 0
    for branch in current_feature.get_branches():
        subset_data_feature_match = get_example_matching_value(data, branch.get_branch_name(), current_feature)

        total_subset_size = len(subset_data_feature_match)

        class_totals_for_subvalue = determine_class_totals(subset_data_feature_match, list_of_classes, False)

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

    """ determine if chi-square value if in or out of distrubution """
    degree_of_freedom = (len(list_of_classes) - 1)  * (len(current_feature.get_branches()) - 1)

    #critical_value = compute_critical_value(degree_of_freedom, .95)
    critical_value = chi_square[degree_of_freedom][confidence_level]

    # Evaluating whether or not the chi_square_value is within the confidence level or not
    if chi_square_value < critical_value:
        return False
    else:
        return True
