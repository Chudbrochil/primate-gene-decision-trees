
# A given feature is a node, the values are otherwise known as "branches"
class Feature():

    def __init__(self, feature_index, values):
        self.feature_index = feature_index
        self.values = values

    def add_branch(self, branch):
        self.branchs.append(branch)

class Branch():
    def __init__(self, branch_value):
        self.branch_value = branch_value
