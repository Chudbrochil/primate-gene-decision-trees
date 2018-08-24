

class Node():
    def __init__(self, feature):
        self.feature = feature
        self.branches = []

    def addBranch(branch):
        self.branches.append(branch)

class Branch():
    def __init__(self, value,):
        self.value = value

class DecisionTree():
    def __init__(self, root):
        self.root = root        #root will be a node containing the feature with highest(?) info gain
