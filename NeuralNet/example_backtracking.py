from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import keras.layers as klayers
import keras.models as kmodels
import keras.utils as kutils
import keras.callbacks as kcallbacks
import NeuralNet.backtracking as bcktrck
import innvestigate as innv
from NeuralNet.IntegratedGradients import *
import xarray as xr
import pandas as pd
import seaborn as sns
from scipy.linalg import cholesky
from scipy.stats import pearsonr

home = expanduser("~")

np.random.seed(5)

def three_inputs_and_target():
    # randm = np.random.randint(1, 50, size=(125000, 2))

    # EOF analysis
    # First make the data deviations from the mean and standardise it
    # data      = randm - randm.mean(axis=0)
    # data_norm = data  / data .std(axis=0)
    data_norm = np.random.standard_normal(size=(125000, 2))

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
    # other_data = np.random.randint(1, 50, size =(125000, 1))
    # other_data = (other_data - other_data.mean()) / other_data.std()
    other_data = np.random.standard_normal(125000)

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


i_1, i_2, i_3, y = three_inputs_and_target()
# y, i_1, i_2, i_3, i_4 = four_inputs_intercorrelated()

# Train MLP, y is the target, [corr_1, corr_2, other_data] is input
target = y
inputs = np.stack((i_1, i_2, i_3), axis=1)
# inputs = np.stack((i_1, i_2, i_3, i_4), axis=1)

l_train_model = False
if l_train_model:
    model = kmodels.Sequential()
    model.add(klayers.Dense(150, activation='relu', input_shape=(inputs.shape[1],)))
    model.add(klayers.Dense(150, activation='relu'))
    model.add(klayers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    filepath = home + '/Desktop/M/model-{epoch:02d}-{val_loss:.5f}.h5'
    checkpoint = kcallbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=False)
    callbacks_list = [checkpoint]
    model.fit(inputs, target, batch_size=10, epochs=15, validation_split=0.3, callbacks=callbacks_list)

    predicted = np.empty_like(target)
    input_percentages = np.zeros_like(inputs)
    l_input_positive = np.full_like(inputs, fill_value=False, dtype='bool')
    for i, iput in enumerate(inputs):
        predicted[i]            = model.predict(np.array([iput]))
        input_percentages[i, :] = bcktrck.mlp_backtracking_percentage(model=model, data_in=iput)[0]
        l_input_positive [i, :] = iput > 0.

    # bach_input_percentages = np.zeros_like(inputs)
    # for i, iput in enumerate(inputs):
    #     bach_input_percentages[i, :] = bcktrck.mlp_backtracking_relevance(model=model, data_in=iput, alpha=1, beta=0)[0]

    # positive_positive_ratio = np.zeros_like(input_percentages[:2, :])
    # for i in range(positive_positive_ratio.shape[1]):
    #     positive_positive_ratio[0, i] = (l_input_positive[:, i] & (input_percentages[:, i] > 0.)).sum() \
    #                                   /  l_input_positive[:, i].sum()
    #     positive_positive_ratio[1, i] = ((l_input_positive[:, i] == False) & (input_percentages[:, i] < 0.)).sum() \
    #                                   /  (l_input_positive[:, i] == False).sum()
else:
    model = kmodels.load_model(home+'/Documents/Data/NN_Models/BasicUnderstanding/'+
                                    'No_Functional_Connection_2layer150nodes/model.h5')
    predicted = xr.open_dataarray(home+'/Documents/Data/NN_Models/BasicUnderstanding/'+
                                       'No_Functional_Connection_2layer150nodes/predicted.nc').values
    input_percentages = xr.open_dataarray(home+'/Documents/Data/NN_Models/BasicUnderstanding/'+
                                               'No_Functional_Connection_2layer150nodes/input_percentages.nc').values

    # the _IB stands for 'ignore bias'. This seems to be the version presented in the Bach/Montavon-papers,
    # because it matches the results of my own implementation of their algorithm.
    lrp10 = innv.create_analyzer(name='lrp.alpha_1_beta_0_IB', model=model)
    lrp21 = innv.create_analyzer(name='lrp.alpha_2_beta_1_IB', model=model)

    baseline_input = np.array([0., 0., 0.])
    ig = integrated_gradients(model)

    bach10_input = np.zeros_like(inputs)
    bach21_input = np.zeros_like(inputs)
    bach32_input = np.zeros_like(inputs)
    lrp10_input  = np.zeros_like(inputs)
    lrp21_input  = np.zeros_like(inputs)
    ig_input     = np.zeros_like(inputs)
    for i, iput in enumerate(inputs):
        # try:
        #     bach10_input[i, :] = bcktrck.mlp_backtracking_relevance(model=model, data_in=iput, alpha=1, beta=0)[0]
        #     bach21_input[i, :] = bcktrck.mlp_backtracking_relevance(model=model, data_in=iput, alpha=2, beta=1)[0]
        #     bach32_input[i, :] = bcktrck.mlp_backtracking_relevance(model=model, data_in=iput, alpha=3, beta=2)[0]
        # except IndexError:
        #     print(f'This is the index: {i}.')
        # lrp10_input[i, :] = lrp10.analyze(np.array([iput]))
        lrp21_input[i, :] = lrp21.analyze(np.array([iput]))
        # ig_input[i, :] = ig.explain(np.array(iput), reference=np.array(baseline_input))

    lrp_sum = lrp21_input.sum(axis=1)
    sum_enlarge = np.broadcast_to(lrp_sum, (3, 125000))
    lrp_percentages = lrp21_input / sum_enlarge.transpose()

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2*0.7, 5*0.7))
# sns.boxplot(data=input_percentages, palette='Set2', fliersize=0., medianprops=dict(lw=1.5, color='black'))
# # sns.violinplot(data=input_percentages, palette='Set2', scale='width', inner='quartile')
# ax.set_ylim(-150, 250)
# ax.axhline(y=0, color='lightgrey', lw=2.5, zorder=-100)
# ax.set_xticklabels(['x$_1$', 'x$_2$', 'x$_3$'])
# # ax.set_xticklabels(['x$_1$', 'x$_2$', 'x$_3$', 'x$_4$'])
# plt.title('LRP-p')
# ax.set_xlabel('Input')
# ax.set_ylabel('Contributing percentage [%]')
# plt.savefig(home+'/Desktop/backtracking_example.pdf', bbox_inches='tight')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2*0.7, 5*0.7))
sns.boxplot(data=lrp21_input, palette='Set2', fliersize=0., medianprops=dict(lw=1.5, color='black'))
# sns.violinplot(data=input_percentages, palette='Set2', scale='width', inner='quartile')
ax.set_ylim(-4, 5)
ax.axhline(y=0, color='lightgrey', lw=2.5, zorder=-100)
ax.set_xticklabels(['x$_1$', 'x$_2$', 'x$_3$'])
# ax.set_xticklabels(['x$_1$', 'x$_2$', 'x$_3$', 'x$_4$'])
ax.set_xlabel('Input')
plt.title('LRP-${\\alpha_2\\beta_1}$')
ax.set_ylabel('Relevance [1]')
# plt.title('Integrated-Gradients')
# ax.set_ylabel('Contribution [1]')
plt.savefig(home+'/Desktop/backtracking_example.pdf', bbox_inches='tight')

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2*0.7, 5*0.7))
# sns.boxplot(data=lrp_percentages*100, palette='Set2', fliersize=0., medianprops=dict(lw=1.5, color='black'))
# # sns.violinplot(data=input_percentages, palette='Set2', scale='width', inner='quartile')
# ax.set_ylim(-350, 350)
# ax.axhline(y=0, color='lightgrey', lw=2.5, zorder=-100)
# ax.set_xticklabels(['x$_1$', 'x$_2$', 'x$_3$'])
# # ax.set_xticklabels(['x$_1$', 'x$_2$', 'x$_3$', 'x$_4$'])
# plt.title('Normalised LRP$_{\\alpha=2, \\beta=1}$')
# ax.set_xlabel('Input')
# ax.set_ylabel('Contributing percentage [%]')
# plt.savefig(home+'/Desktop/backtracking_example_lrp32.pdf', bbox_inches='tight')
#
# plt.close()
# width =  10
# count, bins, _ = plt.hist(input_percentages[:, 1], bins=np.arange(-350, 360, step=width))
# plt.close()
# plt.bar((bins[:-1] + bins[1:])/2., count/125000., width)
# plt.ylabel('Histogram [1]')
# plt.xlabel('Contributing percent by input x$_1$ [%]')
# # plt.title(f'Normalised LRP$_{{\\alpha=3, \\beta=2}}$, '
# plt.title(f'Percentage-backtracking, '
#           f'showing {round((count/125000.).sum()*100)}% of data')
# plt.savefig(home+'/Desktop/hist_lrp.pdf', bbox_inches='tight')