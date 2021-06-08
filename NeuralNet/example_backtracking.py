from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import keras.layers as klayers
import keras.models as kmodels
import NeuralNet.backtracking as bcktrck
import innvestigate as innv
import xarray as xr
import pandas as pd
import seaborn as sns
from scipy.linalg import cholesky
from scipy.stats import pearsonr

home = expanduser("~")

np.random.seed(5)

def three_inputs_and_target():
    randm = np.random.randint(1, 50, size=(125000, 2))

    # EOF analysis
    # First make the data deviations from the mean and standardise it
    data      = randm - randm.mean(axis=0)
    data_norm = data  / data .std(axis=0)
    dimsize = data_norm.shape

    cov_matrix = np.cov(data_norm, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # sorting eigenvalues with descending values
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues [   idx]
    eigenvectors = eigenvectors[:, idx]

    # how much each pattern accounts for of total variance
    variance_perc =  eigenvalues / eigenvalues.sum()

    # compute the principal component time series for all eigenvectors
    pc_all = eigenvectors.T @ data_norm.T / (dimsize[1] - 1)

    # reconstruct the original data (at first 'time') via the pc time series and the patterns (EOFs)
    pattern_0_back = (pc_all[:, 0] @ eigenvectors.T) * (dimsize[1] - 1)

    # Create specifically correlated data to y (x and y are 0-correlated by construction as PC-series)
    # The equation below is an additive mixture, set by r, of y (for correlation) and x (for non-correlation).
    y = pc_all[0]
    x = pc_all[1]

    r = 0.3
    corr_1 = y * r + x * np.sqrt(1 - r**2)
    r = 0.8
    corr_2 = y * r + x * np.sqrt(1 - r**2)

    # other data has also nearly 0-correlation
    other_data = np.random.randint(1, 50, size =(125000, 1))
    other_data = (other_data - other_data.mean()) / other_data.std()

    return corr_1, corr_2, other_data.squeeze(), y


i_1, i_2, i_3, y = three_inputs_and_target()

# Train MLP, y is the target, [corr_1, corr_2, other_data] is input
target = y
inputs = np.stack((i_1, i_2, i_3), axis=1)

l_train_model = False
if l_train_model:
    model = kmodels.Sequential()
    model.add(klayers.Dense(150, activation='relu', input_shape=(inputs.shape[1],)))
    model.add(klayers.Dense(150, activation='relu'))
    model.add(klayers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(inputs, target, batch_size=10, epochs=15, validation_split=0.3)

    predicted = np.empty_like(target)
    input_percentages = np.zeros_like(inputs)
    l_input_positive = np.full_like(inputs, fill_value=False, dtype='bool')
    for i, iput in enumerate(inputs):
        predicted[i]            = model.predict(np.array([iput]))
        input_percentages[i, :] = bcktrck.mlp_backtracking_percentage(model=model, data_in=iput)[0]
        l_input_positive [i, :] = iput > 0.

else:
    model = kmodels.load_model(home+'/Documents/Data/NN_Models/BasicUnderstanding/No_Functional_Connection/'+
                                    'model.h5')
    predicted = xr.open_dataarray(home+'/Documents/Data/NN_Models/BasicUnderstanding/No_Functional_Connection/'+
                                          'predicted.nc').values
    input_percentages = xr.open_dataarray(home+'/Documents/Data/NN_Models/BasicUnderstanding/No_Functional_Connection/'+
                                          'input_percentages.nc').values

    l_layer_wise_relevance = False
    if l_layer_wise_relevance:
        # the _IB stands for 'ignore bias'. This seems to be the version presented in the Bach/Montavon-papers,
        # because it matches the results of my own implementation of their algorithm.
        lrp10 = innv.create_analyzer(name='lrp.alpha_1_beta_0_IB', model=model)
        lrp21 = innv.create_analyzer(name='lrp.alpha_2_beta_1_IB', model=model)

        bach10_input_percentages = np.zeros_like(inputs)
        bach21_input_percentages = np.zeros_like(inputs)
        lrp10_input_percentages = np.zeros_like(inputs)
        lrp21_input_percentages = np.zeros_like(inputs)
        for i, iput in enumerate(inputs):
            bach10_input_percentages[i, :] = bcktrck.mlp_backtracking_relevance(
                model=model, data_in=iput, alpha=1, beta=0)[0]
            bach21_input_percentages[i, :] = bcktrck.mlp_backtracking_relevance(
                model=model, data_in=iput, alpha=2, beta=1)[0]
            lrp10_input_percentages[i, :] = lrp10.analyze(np.array([iput]))
            lrp21_input_percentages[i, :] = lrp21.analyze(np.array([iput]))


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2*0.7, 5*0.7))
sns.boxplot(data=input_percentages, palette='Set2', fliersize=0., medianprops=dict(lw=1.5, color='black'))
# sns.violinplot(data=input_percentages, palette='Set2', scale='width', inner='quartile')
ax.set_ylim(-150, 250)
ax.axhline(y=0, color='lightgrey', lw=2.5, zorder=-100)
ax.set_xticklabels(['x$_1$', 'x$_2$', 'x$_3$'])
# ax.set_xticklabels(['x$_1$', 'x$_2$', 'x$_3$', 'x$_4$'])
plt.title('Percentage-backtracking')
ax.set_xlabel('Input')
ax.set_ylabel('Contributing percentage [%]')
plt.savefig(home+'/Desktop/backtracking_example.pdf', bbox_inches='tight')
