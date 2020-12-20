## Setup
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
import pickle

## Functions from Model Exploration Notebook
def setup_params(time_decay_half_life=240, significance_tol=0.2, high_winning_prob=0.7):
    eps = 1 / (2 * time_decay_half_life)  # time decay epsilon
    max_time_interval = np.log(
        np.log(significance_tol) / np.log(high_winning_prob)) / eps  # in days, which is how far back we'll consider

    return eps, max_time_interval

def log_lilkihood(x,n,df):
    """
    x - a (3n,1) dimentional array with cols [alpha,W]
    n - an integer
    df - an (n,8) df
    """
    # Return -log_lilkihood, so we can max log_lilkihood by minimising -log_lilkihood
    df['aiwi'] = x[df['Winner']] * ((x[n+2*df['Winner']+df['Surface']]) ** df['Surface_mult'])
    df['ajwj'] = x[df['Loser']] *  ((x[n+2*df['Loser']+df['Surface']]) ** df['Surface_mult'])

    return -sum(df['time_decay']*(df['gi']*df['aiwi']+df['gj']*df['ajwj']-(df['gi']+df['gj'])*np.log(np.exp(df['aiwi'])+np.exp(df['ajwj']))))

## Importing Data
mens_df = pd.read_csv('../data/mens.csv',header=0,parse_dates=["Date"])
womens_df = pd.read_csv('../data/womens.csv',header=0,parse_dates=["Date"])

# Remove walkovers
mens_df = mens_df[mens_df['Comment']!='Walkover']
womens_df = womens_df[womens_df['Comment']!='Walkover']

## Model Fitting
eps,max_time_interval = setup_params()
print("We'll consider ",max_time_interval/365," years worth of data")

# Total up games won
winner_cols = [c for c in mens_df.columns if c[0]=="W" and any(char.isdigit() for char in c)]
loser_cols =  [c for c in mens_df.columns if c[0]=="L" and any(char.isdigit() for char in c)]
mens_df[winner_cols].fillna(0);mens_df[loser_cols].fillna(0)
mens_df.loc[:,'gi']= mens_df[winner_cols].sum(axis=1)
mens_df.loc[:,'gj']= mens_df[loser_cols].sum(axis=1)

dates_list = []
for i,df in mens_df.groupby([mens_df.Date.dt.year,'ATP']):
    dates_list.append((min(df['Date']),max(df['Date'])))
dates_list=list(set(dates_list))

alpha_df = pd.DataFrame([],columns=['p_dict','x'],index=pd.MultiIndex.from_tuples(dates_list, names=['start', 'end']))

def save_alpha_df(i):
    print(3000*i/834,"% complete")
    with open('mens_alphas_{c}.pickle'.format(c=i/30), 'wb') as handle:
        pickle.dump(alpha_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

filter_player_universe = False

for i, dates in enumerate(dates_list):
    ## ------ df prep -------- ##
    ## ----------------------- ##

    # 1 - Focus on time periods that we care about
    pred_df = mens_df[(mens_df['Date'] >= dates[0]) & (mens_df['Date'] <= dates[1])]
    prediction_surface = pred_df['Surface'].mode()[0]
    cutoff_date = dates[0] - pd.Timedelta(days=max_time_interval)  # How far back we'll look
    df = mens_df[(mens_df['Date'] >= cutoff_date) & (mens_df['Date'] < dates[0])].copy()  # Strict ineq here important!

    # 2 - Add time weight col
    df.loc[:, 'dt'] = (dates[0] - df.loc[:, 'Date']).dt.days.astype('int16')  # An integer amount of days
    df.loc[:, 'time_decay'] = round(np.exp(-eps * df.loc[:, 'dt']), 2)

    # 3 - Fliter games down even further
    if filter_player_universe:
        players = set(np.concatenate([pred_df['Winner'].values, pred_df['Loser'].values], axis=0))
        df = df[df['Winner'].isin(players) | df['Loser'].isin(players)]

    # 4 - Filter cols
    df = df[['Surface', 'Winner', 'Loser', 'time_decay', 'gi', 'gj']]

    # 5 - Player dict
    unique_players = set(np.concatenate([df['Winner'].values, df['Loser'].values], axis=0))
    n = len(unique_players)
    if n == 0: continue
    player_dict = {}
    for i, player in enumerate(unique_players):
        player_dict[player] = i

    # 6 - surface dict
    if prediction_surface == 'Grass':
        surface_dict = defaultdict(lambda: 1, {'Clay': 0, 'Grass': -1})
    elif prediction_surface != 'Clay':
        surface_dict = defaultdict(lambda: -1, {'Clay': 0, 'Grass': 1})
    else:
        surface_dict = defaultdict(lambda: 1, {'Clay': -1, 'Grass': 0})
    surface_dict  # The prediction surface should have key -1

    # 7 - Mapping cols
    df.loc[:, 'Winner'] = df.loc[:, 'Winner'].map(player_dict)
    df.loc[:, 'Loser'] = df.loc[:, 'Loser'].map(player_dict)
    df.loc[:, 'Surface'] = df['Surface'].map(surface_dict)
    df.loc[:, 'Surface_mult'] = np.where(df['Surface'] == -1, 0, 1)

    # 8 - Data typing
    df.loc[:, df.columns != 'time_decay'] = df.loc[:, df.columns != 'time_decay'].astype(
        int)  # So we can use the entries as list indecies later on

    ## ------ Fitting -------- ##
    ## ----------------------- ##
    x0 = np.ones((3 * n))
    bds = [(0.1, 2.5)] * n + [(0.1, 1.5)] * (2 * n)
    res = minimize(log_lilkihood, x0=x0, args=(n, df), bounds=bds, options={'disp': False, 'maxiter': 50})

    alpha_df.loc[dates]['p_dict'] = player_dict
    alpha_df.loc[dates]['x'] = res.x

    if i % 30 == 0:
        save_alpha_df(i)