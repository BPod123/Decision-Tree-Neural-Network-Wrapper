from keras import Model, layers, activations
from math import log, floor, ceil

from Wrapper import Wrapper


class Tree(Model):
    def __init__(self, root: Wrapper, outputDim: int):
        super(Tree, self).__init__()
        self.root = root
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(outputDim, activation=activations.softmax if outputDim > 1 else activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        out = self.root(inputs)
        out = self.flatten(out)
        out = self.dense(out)
        return out


def nodesInBalancedTree(branchingFactor: int, height: int):
    return int(branchingFactor ** (height + 1) - 1)


def balancedTree(nodes: list, branchingFactor: int, outputDim: int):
    """
    Returns a balanced decision tree made using the passed in nodes and branching factor that makes a decision out of
    outputDim possible choices.
    The height of a balanced tree is log_{branchingFactor} (len(nodes) + 1) - 1
    :param nodes: A list of length branchingFactor ** (height + 1) - 1 where each node is a layer in the tree
    :param branchingFactor: The number of children nodes each non-leaf node should have
    :param outputDim: The output dimension argument for the tree
    :return:
    """
    if len(nodes) < 1:
        raise Exception("Number of nodes must be greater than or equal to one")
    if branchingFactor < 1:
        raise Exception("Branching Factor must be greater than or equal to one")
    height = log(len(nodes) + 1) / log(branchingFactor) - 1
    if not height.is_integer():
        lower = floor(height)
        upper = ceil(height)
        raise Exception(
            f"Invalid number of passed in nodes. There are enough nodes to make a balanced tree of height {height}."
            f"\nUse {int(branchingFactor ** (lower + 1) - 1)} nodes for a balanced tree of height {lower}"
            f"\nor {int(branchingFactor ** (upper + 1) - 1)} nodes for a balanced tree of height {upper}.")
    else:
        height = int(height)
    numLeafNodes = int((branchingFactor ** (height + 1) - 1) - (branchingFactor ** height - 1))
    wrappers = [Wrapper(nodes[i], isLeaf=i >= len(nodes) - numLeafNodes, outputDim=branchingFactor,
                        height=height - i // branchingFactor - 1, index=i) for i in
                range(len(nodes))]
    for i in range(len(nodes)):
        for branch in range(i * branchingFactor + 1, i * branchingFactor + branchingFactor + 1):
            if branch < len(nodes):
                wrappers[i].children.append(wrappers[branch])
            else:
                break
    return Tree(wrappers[0], outputDim)
