import xarray as xr
import pandas as pd

dfr_pope = pd.read_csv('/Users/mret0001/Desktop/Pope_regimes.csv', header=None, names=['time', 'regime'], index_col=0)

dse = pd.Series(dfr_pope['regime'])


