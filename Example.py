import numpy as np
from keras import layers, Sequential, activations
from Tree import nodesInBalancedTree, balancedTree

def encoder():
    return Sequential([layers.InputLayer(input_shape=(5, 51)),
                       layers.Conv1D(32, 3, 2, padding='causal', activation='relu', data_format='channels_first'),
                       layers.Conv1D(64, 3, 2, padding='causal', activation='relu', data_format='channels_first'),
                       layers.Flatten(), layers.Dense(512)])


def decoder():
    return Sequential([layers.InputLayer(input_shape=(512)), layers.Dense(255, activation=activations.relu),
                       layers.Reshape((5, 51)),
                       layers.Conv1DTranspose(64, 4, padding='same', data_format='channels_first'),
                       layers.Conv1DTranspose(32, 4, padding='same', data_format='channels_first'),
                       layers.Conv1DTranspose(5, 4, padding='same', data_format='channels_first')
                       ])


class Node(layers.Layer):
    def __init__(self, encoder, decoder):
        super(Node, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, *args, **kwargs):
        encoderOut = self.encoder(inputs)
        decoderOut = self.decoder(encoderOut)
        return decoderOut

if __name__ == '__main__':

    branchingFactor, height = 2, 2
    nodes = [Node(encoder(), decoder()) for i in range(nodesInBalancedTree(branchingFactor, height))]
    tree = balancedTree(nodes, branchingFactor, 1)
    batch1 = np.ascontiguousarray(np.random.normal(0, 1, (500, 5, 51)))
    noise1 = np.random.normal(size=batch1.size)
    result = tree(batch1)


