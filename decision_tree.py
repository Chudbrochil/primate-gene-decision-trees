

class Node():
    def __init__(self, feature):
        self.feature = feature
        self.branches = []

    def addBranch(self, branch):
        self.branches.append(branch)

class Branch():
    def __init__(self, value):
        self.value = value

class Feature():
    def __init__(self, featureIndex, values):
        self.featureIndex = featureIndex
        self.values = values

    def addValue(self, value):
        self.values.append(value)
