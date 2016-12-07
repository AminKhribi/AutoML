import numpy as np
import pandas as pd

import operator

from scipy import sparse
from sklearn import preprocessing


def compute_X_10(df=None, feature_index=None, feature=None):
    """
    Same as compute_X_tfidf but keeps only occurennces of (feature_index, feature)
    """

    all_users = df[feature_index].values

    n_devices = df[feature_index].nunique()
    n_feature = df[feature].nunique()

    df.loc[df[feature].isnull(), feature] = -1  # we will drop this index later

    data = np.ones(df.shape[0])

    #
    dict_cor = {k: v for k, v in zip(all_users, range(len(all_users)))}
    row = [dict_cor[k] for k in df[feature_index].values]

    #
    # col = LabelEncoder().fit_transform(df[feature])
    col = pd.factorize(df[feature])[0]

    corr_col = {k: v for k, v in zip(df[feature].values, col)}
    list_cols = [feature + '_' + str(i[0]) for i in sorted(corr_col.items(), key=operator.itemgetter(1))]

    if -1 in df[feature].values:
        
        #
        X_10 = sparse.csr_matrix(
            (data, (row, col)), shape=(n_devices, n_feature + 1))

        to_drop = col[list(df[feature].values).index(-1)]  # this is the index of missing label

        X_10 = X_10[:, list(set([i for i in range(X_10.shape[1])]).difference([to_drop]))]

    else:

        X_10 = sparse.csr_matrix(
            (data, (row, col)), shape=(n_devices, n_feature))

    return X_10, list_cols


def one_hot_encoder_wpop_cols(cat_columns, df_train):

    df_train['id_amin'] = range(df_train.shape[0])

    X_cat, cat_cols = compute_X_10(df=df_train, feature_index='id_amin', feature=cat_columns[0])
    X_pop = df_train[cat_columns[0]].map(df_train[cat_columns[0]].value_counts().to_dict()) / df_train.shape[0]

    for cat in cat_columns[1:]:
        
        X, ids = compute_X_10(df=df_train, feature_index='id_amin', feature=cat)

        Xp = df_train[cat].map(df_train[cat].value_counts().to_dict()) / df_train.shape[0]

        X_pop = pd.concat([X_pop, Xp], axis=1)
        X_cat = sparse.hstack((X_cat, X)).tocsr()
        cat_cols += ids

    return X_cat, X_pop, cat_cols



def normalize(do_all, cont_columns, df_train):

    if do_all:
        X_cont = df_train[cont_columns]
        cont_cols = cont_columns

        sc = preprocessing.StandardScaler()
        X_cont = sc.fit_transform(X_cont)
        X_cont = sparse.csr_matrix(X_cont)

    else:

        X_cont = df_train[cont_columns].astype(float)
        X_cont = X_cont.dropna(axis=1, how='all')

        cont_cols = list(X_cont.columns)

        for f in cont_cols:

            if not(np.isnan(X_cont[f].unique()[0]) and (X_cont[f].nunique() == 0)):
                sc = preprocessing.StandardScaler()
                X_cont.loc[~X_cont[f].isnull(), f] = sc.fit_transform(X_cont[~X_cont[f].isnull()][f])

        X_cont = sparse.csr_matrix(X_cont)

    return X_cont, cont_cols