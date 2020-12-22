import pickle
import pandas as pd

df = pd.read_pickle(r'K:\Code\Tennis_Betting\2. Model exploration\alphas\mens_alphas_12.0.pickle')
with open('alphas/mens_alphas.pickle', 'wb') as handle:
    pickle.dump(df, handle)