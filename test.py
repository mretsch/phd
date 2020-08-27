import unittest
import numpy as np
import NeuralNet.backtracking as bcktrck


class TestModel():

    def __init__(self, weights):
        if weights:
            self.weights = weights
        else:
            # weights for an MLP with 3 inputs, 2-node hidden layer, 1 output node.
            # Bias is Zero, all weights are positive.
            self.weights = [np.array([[0.125, 4 / 3], [0.25, 1 / 6], [0.125, 1 / 6]]), np.array([0., 0.]),
                            np.array([[0.5], [0.1]]), np.array([0.])]

    def get_weights(self):
        return self.weights


class MlpBacktrackingPercentageTest(unittest.TestCase):

    def test_mlp_backtracking_percentage(self):

        tm = TestModel(weights=None)
        data = np.array([24, 24, 24])

        # did not find a way to compare complex structures, like lists of numpy-arrays
        self.assertListEqual(list(bcktrck.mlp_backtracking_percentage(model=tm, data_in=data)[0]),
                             list(np.array([47., 34., 19.])))


class MlpBacktrackingRelevanceTest(unittest.TestCase):

    def test_mlp_backtracking_relevance(self):

        weights = [np.array([[0.25, -0.5], [0.25, 1], [0.25, -5 / 12]]), np.array([0., 0.]),
                   np.array([[2/3], [-1]]), np.array([0.])]
        tm = TestModel(weights=weights)
        data = np.array([24, 24, 24])

        # did not find a way to compare complex structures, like lists of numpy-arrays
        self.assertListEqual(list(bcktrck.mlp_backtracking_relevance(model=tm, data_in=data, alpha=2, beta=1)[0]),
                             list(np.array([(4/3)+(12/22), -(2/3), (4/3)+(10/22)])*10.))

        # self.assertListEqual(list(bcktrck.mlp_backtracking_relevance(model=tm, data_in=data)[0]),
        #                      list(np.array([1/3, 1/3, 1/3])*10.))


if __name__ == '__main__':
    unittest.main()