import numpy as np

def mlp_insight(model, data_in):
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
        output[output < 0] = 0
        # append output, so it can be overwritten in next iteration
        results.append(output)

    # after forward pass, recursively find chain of nodes with maximum value in each layer
    last_layer = results[-2] * weight_list[-2].transpose()
    max_nodes = [last_layer.argmax()]
    # concatenate the original NN input, i.e. data_in, and the output from the remaining layers
    iput = [np.array(data_in)] + results[:-2]
    for i in range(n_layers - 1)[::-1]:
        # weights are stored in array of shape (# nodes in layer n, # nodes in layer n+1)
        layer_to_maxnode = iput[i] * weight_list[2 * i][:, max_nodes[-1]]
        max_nodes.append(layer_to_maxnode.argmax())

    return np.array(max_nodes[::-1])
