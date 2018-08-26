import math

""" Caclulates entropy on current set of examples.
    Used for the entire dataset and each value of a feature.
    examples - the dataset containing features and labels.
    classes - list of possible classifications of an example.
    based on eq. 3.3 pg. 59 of Machine Learning by Tom Mitchell
 """
def entropy(examples, classes):

    entropy = 0

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
        #go through each class for current example, once match found, break
        for i in range(numOfClasses):
            #the output will always be the last element of the example
            if(example[-1:] == labels["class" + str(i)]):
                #if the output class for this example matches, add one to total classes
                label_totals["class" + str(i)] = label_totals["class" + str(i)] + 1
                break

    #calculate entropy now that proportions are known (p_i)
    for i in range(numOfClasses):
        p_i = label_totals["class" + str(i)] / total_examples
        if p_i != 0:
            entropy = entropy - p_i * math.log(p_i, 2)

    return entropy

""" Gain calculates the information gain of each feature on current passed in examples
   gain(data, A) -> will look through values Y & Z for feature A.
   feature - an object... with this features column index in the
   dataset and needs to have a list of it's values....
"""
#TODO DETERMINE IF FEATURE BEING PASSED INTO GAIN SHOULD BE AN OBJECT
def gain(examples, feature, classes):

    #gain step 1, take entropy of all examples
    gain = entropy(examples, classes)

    #step 1.5, make examles into a dictionary
    # dictionary_examples = convertExamplesToDictionary(examples)

    feature.values = set(feature.values)
    print("Features found at root: " + str(feature.values))

    #gain step2, sum entropies of each value for current feature
    for value in feature.values:
        subset_of_example = valuesInExamples(examples, value, feature)
        total_subset_of_value = len(subset_of_example)
        proportion_of_subset = total_subset_of_value / len(examples) * 1.0
        subset_entropy = proportion_of_subset * entropy(subset_of_example, classes)
        gain = gain - subset_entropy

    return gain


def valuesInExamples(examples, value, feature):
    subset_of_value = []
    #go through each example
    for example in examples:
        #go through only feature passed into gain and check value
        if(example[feature.featureIndex] == value):
            subset_of_value.append(example)

    return subset_of_value
