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


# impurity()
# This is the combined functionality of entropy and gni_index.
# It gets the proportionality of each classification and then loops over
# all of the classes to calcuulate the impurity value. If is_entropy is True,
# then the math will be entropy, otherwise the math will be gni_index.
def impurity(examples, classes, is_entropy):

    # Starting value for entropy is 0, 1 for gni_index
    if is_entropy == True:
        impurity_value = 0
    else:
        impurity_value = 1

    # Getting the proportionality for each classification
    label_totals = determine_class_totals(examples, classes)

    total_examples = len(examples)
    total_classes = len(classes)

    if total_examples == 0:
        return 0

    # Loop over all possible classifications and calculate the purity via either
    # entropy or gni_index, depending on which was selected.
    for i in range(total_classes):
        p_i = label_totals["class" + str(i)] / total_examples
        if p_i != 0:
            # Entropy
            if is_entropy == True:
                impurity_value = impurity_value - (p_i * math.log(p_i, 2))
            # GNI Index
            else:
                impurity_value = impurity_value - (p_i ** 2)

    return impurity_value


# gain()
# Gain calculates the information gain of each feature on current passed in examples
# gain(data, A) -> will look through values Y & Z for feature A.
# feature - an object... with this features column index in the
# dataset and needs to have a list of it's values.
def gain(examples, feature, classes, is_entropy):
    #determine impurity of entire dataset
    gain = impurity(examples, classes, is_entropy)

    #determine impurity for each value in feature, sum together and return info gain
    for branch in feature.get_branches():

        #return subset of examples that only have this value
        subset_of_example = get_example_matching_value(examples, branch.get_branch_name(), feature)

        total_subset_of_value = len(subset_of_example)

        #math of information gain
        proportion_of_subset = total_subset_of_value / len(examples) * 1.0
        subset_entropy = proportion_of_subset * impurity(subset_of_example, classes, is_entropy)
        gain = gain - subset_entropy

    return gain


#TODO can easily replace dictionaries with list if preferred
# TODO: (Tristin) Comment this.
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
