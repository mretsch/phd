import unittest
import numpy as np
from NeuralNet.backtracking import mlp_backtracking_percentage


class MlpBacktrackingPercentageTest(unittest.TestCase):

    def test_mlp_backtracking_percentage(self):

        class TestModel():
            def get_weights(self):
                # weights for an MLP with 3 inputs, 2-node hidden layer, 1 output node. Bias is Zero.
                return [np.array([[0.125, 4/3], [0.25, 1/6], [0.125, 1/6]]), np.array([0., 0.]),
                        np.array([[0.5], [0.1]]), np.array([0.])]

        tm = TestModel()
        data = np.array([24, 24, 24])

        # did not find a way to compare complex structures, like lists of numpy-arrays
        self.assertListEqual(list(mlp_backtracking_percentage(model=tm, data_in=data)[0]),
                             list(np.array([47., 34., 19.])))


if __name__ == '__main__':
    unittest.main()