import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def high_correct_predictions(target, predictions, target_percentile, prediction_offset):
    """Return 1d-xarray containing a subset of target and predictions where target has a higher percentile than
    in target-percentile (in [0, 1]) and the prediction deviates less than prediction_offset
    (in [0, 1]) from the target."""

    assert 'percentile' in target.coords, 'Target needs to have <percentile> as a coordinate,' \
                                          'in order to subselect target based upon percentiles.'

    # only times that could be predicted (via large-scale set). Sample size: 26,000 -> 6,000
    target = target.where(predictions.time)
    # only interested in high ROME values. Sample size: O(100)
    target_high = target[target['percentile'] > target_percentile]
    diff = predictions - target_high
    off_percent = (abs(diff) / target_high).values
    # allow x% of deviation from true value
    correct_pred = xr.where(abs(diff) < prediction_offset * target, True, False)
    predictions_sub = predictions.sel(time=target_high[correct_pred].time.values)
    target_sub      = target     .sel(time=target_high[correct_pred].time.values)
    return target_sub, predictions_sub


def mlp_insight(model, data_in, n_highest_node):
    """
    Compute the most contributing node index in each layer of a regression MLP.
    Returns an array with the first element corresponding to the first layer of the MLP,
    the second element to the second layer, etc..

    Parameters
    ----------
    model :
        Trained regression multilayer perceptron from keras, with one output node.
    data_in :
        xarray-dataarray or list with a single instance of prediction values
        for the provided model.
    n_highest_node :
        Which node is backtracked through the model. The most contributing node
        is given for '1', the second-most contributing for '2', etc..
    """

    output = np.array(data_in)
    weight_list = model.get_weights()
    # each layer has weights and biases
    n_layers = int(len(weight_list) / 2)

    # cycle through the layers, a forward pass
    results = []
    for i in range(n_layers):
        # get appropriate trained parameters, first are weights, second are biases
        weights = weight_list[i * 2]
        bias = weight_list[i * 2 + 1]
        # the @ is a matrix multiplication, first output is actually the mlp's input
        output = weights.transpose() @ output + bias
        # ReLU
        output[output < 0] = 0
        # append output, so output can be overwritten in next iteration
        results.append(output)

    # after forward pass, recursively find chain of nodes with maximum value in each layer.
    # Last layer maps to only one output node, thus weigh_list has only one element for last layer.
    last_layer = results[-2] * weight_list[-2][:, 0].transpose()
    idx_ascending = last_layer.argsort()
    max_nodes = [idx_ascending[-n_highest_node]]

    # concatenate the original NN input, i.e. data_in, and the output from the remaining layers,
    # excluding output and last layer. iput, like results, are the values in previous layer which have been calculated
    # in a forward pass, i.e. bias and non-linear function have been applied.
    iput = [np.array(data_in)] + results[:-2]
    for i in range(n_layers - 1)[::-1]:
        # weights are stored in array of shape (# nodes in layer n, # nodes in layer n+1)
        layer_to_maxnode = iput[i] * weight_list[2 * i][:, max_nodes[-1]]
        idx_ascending = layer_to_maxnode.argsort()
        max_nodes.append(idx_ascending[-n_highest_node])

    # first_conn = iput[0] * weight_list[0][:, max_nodes[-2]] # take the max_node in the first layer (not input layer)
    # plt.plot(first_conn, alpha=0.1)
    return np.array(max_nodes[::-1])
