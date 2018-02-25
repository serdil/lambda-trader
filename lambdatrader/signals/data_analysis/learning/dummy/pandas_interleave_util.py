import pandas as pd


def interleave_dfs(dfs):
    return pd.concat(dfs).sort_index()
