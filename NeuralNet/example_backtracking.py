from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import keras.layers as klayers
import keras.models as kmodels
import keras.utils as kutils
import keras.callbacks as kcallbacks
import NeuralNet.backtracking as bcktrck
import xarray as xr
import pandas as pd
import seaborn as sns
from scipy.linalg import cholesky
from scipy.stats import pearsonr

home = expanduser("~")

np.random.seed(5)

def three_inputs_and_target():
    randm = np.random.randint(1, 50, size=(125000, 2))

    x_0 = randm[:, 0]
    x_1 = randm[:, 1]

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

    # Create specifically correlated data to x (x and y are 0-correlated by construction as PC-series)
    # The equation below is an additive mixture, set by r, of x (for correlation) and y (for non-correlation).
    y = pc_all[0]
    x = pc_all[1]

    r = 0.3
    corr_1 = y * r + x * np.sqrt(1 - r**2)
    r = 0.8
    corr_2 = y * r + x * np.sqrt(1 - r**2)

    # other data has also nearly 0-correlation
    other_data = np.random.randint(1, 50, size =(125000, 1))
    other_data = (other_data - other_data.mean()) / other_data.std()

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(72, 4))
    # ax.plot(y[:500])
    # ax.plot(corr_1[:500])
    # ax.plot(corr_2[:500])
    # ax.plot(other_data[:500])
    # plt.legend(['Target', 'Corr-0.3', 'Corr-0.8', 'Other'])
    # plt.savefig(home+'/Desktop/a.pdf', bbox_inches='tight')

    return corr_1, corr_2, other_data.squeeze(), y


def four_inputs_intercorrelated():
    # adopted from
    # https://www.gaussianwaves.com/2014/07/generating-multiple-sequences-of-correlated-random-variables/

    C = np.array([[1   , 0.7 , 0.7 , 0.3 , 0.3 ],
                  [0.7 , 1   , 0.0 , 0.0 , 0.0 ],
                  [0.7 , 0.0 , 1   , 0.0 , 0.0 ],
                  [0.3 , 0.0 , 0.0 , 1   , 0.0 ],
                  [0.3 , 0.0 , 0.0 , 0.0 , 1   ]])  # Construct correlation matrix

    l_cholesky = False
    if l_cholesky:
        U = cholesky(C)  # Cholesky decomposition
        R = np.random.randn(125000, 5)  # Three uncorrelated sequences
        generated_data = R @ U  # Array of correlated random sequences

    l_eigen = True
    if l_eigen:
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # sorting eigenvalues with descending values
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # because I don't have data to begin with, I can't construct the PC series
        # pc_all = eigenvectors.T @ data_norm.T / (dimsize[1] - 1)
        # But the PC itself is just orthogonal random stuff, so maybe using random orthogonal stuff allows me to get data
        # with the desired correlations as in cov_matrix. Yes, see below.

        # But I think first I have to scale the vectors appropriately to get correct correlated data back.
        # Think of 2D-example, if eigenvectors are not scaled the correlation for the scatter plot would be Zero,
        # but depending on the scaling the correlation can become close to 1, when multiplying the scaled vectors
        # both with random data.
        # at the moment, all eigenvectors[:, i] are scaled such that each has the l2-norm of 1.
        l_scale_vectors = True
        if l_scale_vectors:
            norm_orig = np.linalg.norm(eigenvectors, axis=0)
            # now scale each vector such that its l2-norm equals sqrt(eigenvalue).
            eigenvalues[eigenvalues < 0.] = 0
            scale_factor = np.sqrt(eigenvalues) / norm_orig
            eigenvectors = scale_factor * eigenvectors

        # fake_pc_all = np.random.random(size=(125000, 5)) # in [0, 1]
        # fake_pc_all = np.random.randn(125000, 5)  # in N(mu=0, sigma=1)
        fake_pc_all = np.random.randint(1, 50, size=(125000, 5))  # in [0, 50]
        generated_data = fake_pc_all @ eigenvectors.T

        # e voila
        print(np.corrcoef(generated_data.T))

    # # compute and display correlation coeff from generated sequences
    # def pearsonCorr(x, y, **kws):
    #     (r, _) = pearsonr(x, y)  # returns Pearson’s correlation coefficient, 2-tailed p-value)
    #     ax = plt.gca()
    #     ax.annotate("r = {:.2f} ".format(r), xy=(.7, .9), xycoords=ax.transAxes)
    #
    # # Visualization
    # df = pd.DataFrame(data=generated_data, columns=['a', 'b', 'X', 'Y', 'Z'])
    # graph = sns.pairplot(df)
    # graph.map(pearsonCorr)
    # plt.show()

    return (generated_data[:, i] for i in range(generated_data.shape[1]))


# i_1, i_2, i_3, y = three_inputs_and_target()
y, i_1, i_2, i_3, i_4 = four_inputs_intercorrelated()



# Train MLP, y is the target, [corr_1, corr_2, other_data] is input
target = y
inputs = np.stack((i_1, i_2, i_3, i_4), axis=1)

l_train_model = True
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

    # positive_positive_ratio = np.zeros_like(input_percentages[:2, :])
    # for i in range(positive_positive_ratio.shape[1]):
    #     positive_positive_ratio[0, i] = (l_input_positive[:, i] & (input_percentages[:, i] > 0.)).sum() \
    #                                   /  l_input_positive[:, i].sum()
    #     positive_positive_ratio[1, i] = ((l_input_positive[:, i] == False) & (input_percentages[:, i] < 0.)).sum() \
    #                                   /  (l_input_positive[:, i] == False).sum()
else:
    model = kmodels.load_model(home+'/Documents/Data/NN_Models/BasicUnderstanding/No_Functional_Connection/'+
                                    'model.h5')
    predicted = xr.open_dataarray(home+'/Documents/Data/NN_Models/BasicUnderstanding/No_Functional_Connection/'+
                                          'predicted.nc').values
    input_percentages = xr.open_dataarray(home+'/Documents/Data/NN_Models/BasicUnderstanding/No_Functional_Connection/'+
                                          'input_percentages.nc').values


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2*0.7, 5*0.7))
sns.boxplot(data=input_percentages, palette='Set2', fliersize=0., medianprops=dict(lw=1.5, color='black'))
# sns.violinplot(data=input_percentages, palette='Set2', scale='width', inner='quartile')
ax.set_ylim(-150, 250)
ax.axhline(y=0, color='lightgrey', lw=2.5, zorder=-100)
ax.set_xticklabels(['x$_1$', 'x$_2$', 'x$_3$', 'x4'])
ax.set_xlabel('Input')
ax.set_ylabel('Contributing percentage [%]')
plt.savefig(home+'/Desktop/backtracking_example.pdf', bbox_inches='tight')
