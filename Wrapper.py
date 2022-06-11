import tensorflow as tf
from keras import layers, activations


class Wrapper(layers.Layer):
    def __init__(self, node, isLeaf: bool, outputDim=0, sendOutputToChildren=False, **kwargs):
        """
        :param node: A tensorflow layer/model
        :param isLeaf: If this network is a leaf
        :param outputDim: The dimension of the decision vector
        :param sendOutputToChildren: If the output of this network will be used as the input for chil nodes. If false,
        the input given to this network will be passed as the input to any children
        :param kwargs:
        """
        super(Wrapper, self).__init__()
        self.sendOutputToChildren = sendOutputToChildren
        self.node = node
        self.outputDim = outputDim
        self.isLeaf = isLeaf
        self.children = []  # Child nodes get appended outside of initializer
        self.kwargs = kwargs
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(outputDim,
                                  activation=activations.softmax if outputDim > 1 else activations.sigmoid)


    @property
    def hasAllChildren(self):
        return self.isLeaf or len(self.children) == self.outputDim

    def __str__(self):
        if 'index' in self.kwargs:
            try:
                return f"{self.kwargs['index']}: {self.node}\t Children: " + ", ".join(
                    [str(x.kwargs['index']) for x in self.children])
            except:
                pass
        return f"{self.node}\tChildren: " + ", ".join([str(x.node) for x in self.children])

    def __repr__(self):
        return str(self)

    def call(self, inputs, *args, **kwargs):
        if not self.hasAllChildren:
            raise (f"Error in {self}: len(self.children) != numChildren. {len(self.children)} != {self.outputDim}.")
        output = nextInputs = self.node(inputs)
        decisionOut = self.dense(self.flatten(output))
        if self.isLeaf:
            return tf.expand_dims(decisionOut, 1)
        else:
            decision = tf.math.argmax(decisionOut, -1)
            decisionOut = tf.expand_dims(decisionOut, 1)
            indices = tf.cumsum(tf.ones_like(decision)) - 1
            allMappings = []
            allOutputs = []
            for i in range(len(self.children)):
                mappings = indices[decision == i]
                onPath = tf.gather(nextInputs, mappings) if self.sendOutputToChildren else tf.gather(inputs, mappings)
                result = self.children[i](onPath)
                allOutputs.append(result)
                allMappings.append(mappings)

            orderedOutputs = tf.gather(tf.concat(allOutputs, 0), tf.concat(allMappings, -1))
            results = tf.concat([decisionOut, orderedOutputs], -2)
            return results