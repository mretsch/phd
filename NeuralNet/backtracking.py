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


def softmax(x):
    return np.exp(x) / (np.exp(x)).sum()


def mlp_forward_pass(input_to_mlp=None, weight_list=None):
    """Computes a forward pass for an MLP given the input, weights list by keras.
    Returns all node values of the MLP as a list containing arrays."""

    n_layers = int(len(weight_list) / 2)
    # output is multiplied with corresponding weights, same goes for input in first iteration
    output = input_to_mlp
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
        # append output to results, so output can be overwritten in next iteration
        results.append(output)
    return results


def mlp_backtracking_maxnode(model, data_in, n_highest_node, return_firstconn=False):
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

    weight_list = model.get_weights()
    # each layer has weights and biases
    n_layers = int(len(weight_list) / 2)

    node_values = mlp_forward_pass(input_to_mlp=np.array(data_in), weight_list=weight_list)

    # after forward pass, recursively find chain of nodes with maximum value in each layer.
    # Last layer maps to only one output node, thus weigh_list has only one element for last layer.
    last_layer = node_values[-2] * weight_list[-2][:, 0].transpose()
    idx_ascending = last_layer.argsort()
    max_nodes = [idx_ascending[-n_highest_node]]

    # concatenate the original NN input, i.e. data_in, and the output from the remaining layers,
    # excluding output and last layer. iput, like results, are the values in previous layer which have been calculated
    # in a forward pass, i.e. bias and non-linear function have been applied.
    iput = [np.array(data_in)] + node_values[:-2]
    for i in range(n_layers - 1)[::-1]:
        # weights are stored in array of shape (# nodes in layer n, # nodes in layer n+1)
        layer_to_maxnode = iput[i] * weight_list[2 * i][:, max_nodes[-1]]
        idx_ascending = layer_to_maxnode.argsort()
        max_nodes.append(idx_ascending[-n_highest_node])

    if return_firstconn:
        first_conn = iput[0] * weight_list[0][:, max_nodes[-2]] # take the max_node in the first layer (not input layer)
        # plt.plot(first_conn, alpha=0.1)
        return np.array(max_nodes[::-1]), first_conn
    else:
        return np.array(max_nodes[::-1])


def mlp_backtracking_percentage(model, data_in):
    """
    Compute the total percentage contribution of each node in an MLP towards the predicted result.
    Returns a list with the first element corresponding to the first layer of the MLP,
    the second element to the second layer, etc..

    Parameters
    ----------
    model :
        Trained regression multilayer perceptron from keras, with one output node.
    data_in :
        xarray-dataarray or list with a single instance of prediction values
        for the provided model.
    """

    weight_list = model.get_weights()
    # each layer has weights and biases
    n_layers = int(len(weight_list) / 2)

    node_values = mlp_forward_pass(input_to_mlp=np.array(data_in), weight_list=weight_list)

    # ===== Backtracking =======
    node_percentages = []
    node_percentages_full = []

    # Last layer maps to only one output node, thus weigh_list has only one element for last layer.
    last_layer = node_values[-2] * weight_list[-2][:, 0].transpose()

    # attribute to each node the percentage with which this node contributed to next layer
    last_layer_perc = last_layer / (last_layer.sum() + weight_list[-1]) * 100
    node_percentages     .append(last_layer_perc)
    node_percentages_full.append(last_layer_perc)

    # concatenate (+ for lists) the original NN input, i.e. data_in, and the output from the remaining layers,
    # excluding output and last layer. iput, like node_values, are the values in previous layer
    # which have been calculated in a forward pass, i.e. bias and non-linear function have been applied.
    iput = [np.array(data_in)] + node_values[:-2]

    # cycle through layers, from back of MLP to front
    for i in range(n_layers - 1)[::-1]:
        weights = weight_list[2 * i    ]
        bias    = weight_list[2 * i + 1]

        # weights are stored in array of shape (# nodes in layer n, # nodes in layer n+1), contributions as well
        contributions_perc = np.zeros_like(weights)
        # for each weight-set calculate how much each node in iput-layers contributes to a node in next layer
        for j in range(weights.shape[1]):
            contribution_to_node = iput[i] * weights[:, j]

            dot_plus_bias = contribution_to_node.sum() + bias[j]

            contributions_perc[:, j] = contribution_to_node / dot_plus_bias * node_percentages[-1][j]

            # Try softmax. Results: not convincing. Cannot differentiate between different input nodes.
            # contributions_perc[:, j] = softmax(contribution_to_node) * node_percentages[-1][j]

        # sum all contributions that went from each node in iput-layer to next layer
        node_percentages     .append(contributions_perc.sum(axis=1))
        node_percentages_full.append(contributions_perc)

    return node_percentages[::-1]


def mlp_backtracking_relevance(model, data_in, alpha=None, beta=None):
    """
    Compute the relevance contribution of each node in an MLP towards the predicted result.
    See 'Methods for interpreting and understanding deep neural networks', Montavon 2018. .
    Returns a list with the first element corresponding to the first layer of the MLP,
    the second element to the second layer, etc..

    Parameters
    ----------
    model :
        Trained regression multilayer perceptron from keras, with one output node.
    data_in :
        xarray-dataarray or list with a single instance of prediction values
        for the provided model.
    """

    weight_list = model.get_weights()
    # each layer has weights and biases
    n_layers = int(len(weight_list) / 2)

    node_values = mlp_forward_pass(input_to_mlp=np.array(data_in), weight_list=weight_list)

    # ===== Backtracking =======
    node_relevance = []

    weights = weight_list[-2][:, 0]
    nodes_times_weights = node_values[-2] * weights

    # values given from last layer maps to (one) output node.
    last_layer_positive = np.where(nodes_times_weights >  0., nodes_times_weights, 0.)
    last_layer_negative = np.where(nodes_times_weights <= 0., nodes_times_weights, 0.)

    # from paper about layer relevance propagation (LRP), Montavon 2018
    # catch dividing by zero
    positive_term = alpha * last_layer_positive / last_layer_positive.sum() if last_layer_positive.sum() != 0. else 0.
    negative_term = beta  * last_layer_negative / last_layer_negative.sum() if last_layer_negative.sum() != 0. else 0.

    # attribute to each node the percentage with which this node contributed to next layer
    # for output node, its relevance is defined as its value.
    last_layer_relevance = (positive_term - negative_term) * node_values[-1]
    node_relevance.append(last_layer_relevance)

    # concatenate (+ for lists) the original NN input, i.e. data_in, and the output from the remaining layers,
    # excluding output and last layer. iput, like node_values, are the values in previous layer
    # which have been calculated in a forward pass, i.e. bias and non-linear function have been applied.
    iput = [np.array(data_in)] + node_values[:-2]

    # cycle through rest of layers, from back of MLP to front
    for i in range(n_layers - 1)[::-1]:
        weights = weight_list[2 * i]
        bias    = weight_list[2 * i + 1]

        # weights are stored in array of shape (# nodes in layer n, # nodes in layer n+1), relevance as well
        relevance = np.zeros_like(weights)

        # for each weight-set calculate how much each node in iput-layers contributes to a node in next layer
        for j in range(weights.shape[1]):

            iput_times_weights = iput[i] * weights[:, j]
            iput_positive = np.where(iput_times_weights >  0., iput_times_weights, 0.)
            iput_negative = np.where(iput_times_weights <= 0., iput_times_weights, 0.)

            positive_term = alpha * iput_positive / iput_positive.sum() if iput_positive.sum() != 0. else 0.
            negative_term = beta  * iput_negative / iput_negative.sum() if iput_negative.sum() != 0. else 0.

            relevance[:, j] = (positive_term - negative_term) * node_relevance[-1][j]

        # sum all contributions that went from each node in iput-layer to next layer
        node_relevance.append(relevance.sum(axis=1))

    return node_relevance[::-1]
