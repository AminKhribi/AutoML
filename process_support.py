import pandas as pd
import numpy as np

import datetime, multiprocessing, operator, random

from scipy import sparse

from sklearn import preprocessing, model_selection, linear_model, neighbors, metrics, tree, ensemble, cross_validation



import matplotlib.pyplot as plt

def plot_imp(imp, name):

	df = pd.DataFrame(imp, columns=['feature', 'fscore'])

	fig = plt.figure()
	df.plot()
	df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 16))
	plt.xlabel('relative importance')
	plt.gcf().savefig("feature_importance_{}.png".format(name))









def run_KNN(X_learn=None, Y_learn=None, X_test=None, Y_test=None, score_func=None, columns=None, dict_cv=None, verbose=None):

    clf = neighbors.KNeighborsRegressor()

    clf.fit(X_learn, Y_learn)

    # bp = gs.best_params_
    # print(bp)
    # clf = gs.best_estimator_

    y_hat = clf.predict(X_test)

    score = 0
    if not (score_func is None):
        score = score_func(Y_test, y_hat)

    return y_hat, score





def run_LR(X_learn=None, Y_learn=None, X_test=None, Y_test=None, alpha=0.01,
                                                                 score_func=None,
                                                                 columns=None,
                                                                 verbose=None):

    clf = linear_model.Ridge(alpha=alpha)

    clf.fit(X_learn, Y_learn)

    y_hat = clf.predict(X_test)

    score = None
    if not (score_func is None):
        score = score_func(Y_test, y_hat)

    return y_hat, score





def run_LR_cv(X_learn=None, Y_learn=None, X_test=None, Y_test=None, dict_cv={"alpha": np.linspace(0.01, 15, 7)},
                    											 score_func=None,
                    											 scoring='mean_squared_error',
                    											 columns=None,
                    											 verbose=None):

    gs = model_selection.GridSearchCV(linear_model.Ridge(),
                                  dict_cv,
                                  scoring=scoring,
                                  n_jobs=-1,
                                  iid=False,
                                  refit=True,
                                  verbose=verbose)

    gs.fit(X_learn, Y_learn)

    bp = gs.best_params_
    print(bp)
    clf = gs.best_estimator_

    y_hat = clf.predict(X_test)

    score = None
    if not (score_func is None):
    	score = score_func(Y_test, y_hat)

    return y_hat, score









from sklearn import tree

def run_DT(X_learn=None, Y_learn=None, X_test=None, Y_test=None, max_depth=3,
                                                     score_func=None,
                                                     columns=None,
                                                     verbose=None):


    clf = tree.DecisionTreeRegressor(max_depth=max_depth)

    clf.fit(X_learn, Y_learn)

    importance = [(i, j) for i, j in zip(columns, clf.feature_importances_)]
    importance = sorted(importance, key=operator.itemgetter(1))

    plot_imp(importance[-10:], "DT")

    y_hat = clf.predict(X_test)

    score = None
    if not (score_func is None):
        score = score_func(Y_test, y_hat)

    return y_hat, score



def run_DT_cv(X_learn=None, Y_learn=None, X_test=None, Y_test=None, dict_cv={"max_depth": np.arange(3, 16, 4)},
        											 score_func=None,
        											 scoring='mean_squared_error',
        											 columns=None,
        											 verbose=None):

    gs = model_selection.GridSearchCV(tree.DecisionTreeRegressor(),
                                  dict_cv,
                                  scoring=scoring,
                                  n_jobs=-1,
                                  iid=False,
                                  refit=True,
                                  verbose=verbose)

    gs.fit(X_learn, Y_learn)

    bp = gs.best_params_
    print(bp)
    clf = gs.best_estimator_

    importance = [(i, j) for i, j in zip(columns, clf.feature_importances_)]
    importance = sorted(importance, key=operator.itemgetter(1))

    plot_imp(importance[-10:], "DT")

    y_hat = clf.predict(X_test)

    score = None
    if not (score_func is None):
    	score = score_func(Y_test, y_hat)

    return y_hat, score








def run_RF(X_learn=None, Y_learn=None, X_test=None, Y_test=None, max_depth=3,
                                                                 n_estimators=100,
                                                                 score_func=None,
                                                                 columns=None,
                                                                 verbose=None):

    clf = ensemble.RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, n_jobs=-1)

    clf.fit(X_learn, Y_learn)

    importance = [(i, j) for i, j in zip(columns, clf.feature_importances_)]
    importance = sorted(importance, key=operator.itemgetter(1))
    
    plot_imp(importance[-10:], "RF")

    y_hat = clf.predict(X_test)

    score = None
    if not (score_func is None):
        score = score_func(Y_test, y_hat)

    return y_hat, score


def run_RF_cv(X_learn=None, Y_learn=None, X_test=None, Y_test=None, dict_cv={"max_depth": np.arange(3, 16, 5),
                                                                        "n_estimators": np.arange(20, 200, 100)},
                                                                 score_func=None, scoring='mean_squared_error',
                                                                 columns=None,
                                                                 verbose=None):

    gs = model_selection.GridSearchCV(ensemble.RandomForestRegressor(n_jobs=-1),
                                  dict_cv,
                                  scoring=scoring,
                                  iid=False,
                                  refit=True,
                                  verbose=verbose)

    gs.fit(X_learn, Y_learn)

    bp = gs.best_params_
    print(bp)
    clf = gs.best_estimator_

    importance = [(i, j) for i, j in zip(columns, clf.feature_importances_)]
    importance = sorted(importance, key=operator.itemgetter(1))
    
    plot_imp(importance[-10:], "RF")

    y_hat = clf.predict(X_test)

    score = None
    if not (score_func is None):
        score = score_func(Y_test, y_hat)

    return y_hat, score











from fastFM import als



def run_FM(X_learn=None, Y_learn=None, X_test=None, Y_test=None, l2_reg_w=0.01,
                                                                l2_reg_V=0.01,
                                                                rank=2,
                                                                score_func=None,
                                                                columns=None,
                                                                verbose=None):




    fm = als.FMRegression(l2_reg_V=l2_reg_V,
                           l2_reg_w=l2_reg_w,
                           rank=rank)

    fm.fit(X_learn, Y_learn)

    y_pred = fm.predict(X_test)

    score_fm = 0
    if not (score_func is None):
        score_fm = score_func(Y_test, y_pred)

    return y_pred, score_fm






def run_FM_cv(X_learn=None, Y_learn=None, X_test=None, Y_test=None, dict_cv={"l2_reg_w": np.linspace(0.01, 15, 5),
                                                                      "l2_reg_V": np.linspace(0.01, 15, 5),
                                                                      "rank": [2, 4, 6, 8]},
                                                            score_func=None,
                                                            scoring='mean_squared_error',
                                                            columns=None,
                                                            nb_trails=5,
                                                            verbose=None):


    ss = []

    skf = list(cross_validation.KFold(X_learn.shape[0], nb_trials))

    for k, (train, test) in enumerate(skf):

        X_learn_cv, X_test_cv = X_learn[train], X_learn[test]
        Y_learn_cv, Y_test_cv = Y_learn[train], Y_learn[test]

        print("cv {}".format(k), end='\r')
        l2_reg_V = random.choice(dict_cv['l2_reg_V'])
        l2_reg_w = random.choice(dict_cv['l2_reg_w'])
        rank = random.choice(dict_cv['rank'])

        reg = als.FMRegression(l2_reg_V=random.choice(dict_cv['l2_reg_V']),
                               l2_reg_w=random.choice(dict_cv['l2_reg_w']),
                               rank=random.choice(dict_cv['rank']))

        reg.fit(X_learn_cv, Y_learn_cv)

        y_pred = reg.predict(X_test_cv)

        s = metrics.mean_squared_error(y_pred, Y_test_cv)
        ss.append((l2_reg_V, l2_reg_w, rank, s))

    best = min(ss, key=operator.itemgetter(3))
    print(best)

    fm = als.FMRegression(l2_reg_V=best[0],
                           l2_reg_w=best[1],
                           rank=best[2])

    fm.fit(X_learn, Y_learn)

    y_pred = fm.predict(X_test)

    score_fm = 0
    if not (score_func is None):
        score_fm = score_func(Y_test, y_pred)

    return y_pred, score_fm








import xgboost as xgb



def run_xgb(X_learn=None, Y_learn=None, X_test=None, Y_test=None, random_state=0, score_func=None, columns=None, 
                        md=3, nbr=100, ss=0.7, eta=0.2, csbt=0.7, gamma=0, patience=1000, weights=None, verbose=False):
    
    num_boost_round = nbr
    early_stopping_rounds = patience
    test_size = 0.3

    eta = eta
    gamma = gamma
    max_depth = md
    subsample = ss
    colsample_bytree = csbt if X_learn.shape[1] > 1 else 1

    params_dict = {
                    "objective": "reg:linear",  # "binary:logistic" or "reg:linear" for regression
                    "booster" : "gbtree",  # "gblinear"
                    "eval_metric": "rmse",  # binary clf: "logloss", regression: "rmse"
                    "eta": eta,
                    "max_depth": max_depth,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "silent": 1,
                    "seed": random_state,
                    "gamma": gamma,
                    }

    if not (weights is None):
        X_train, X_valid, y_train, y_valid, w_train, w_valid = cross_validation.train_test_split(X_learn, Y_learn, weights, train_size=.80, random_state=random_state)

    else:
        X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X_learn, Y_learn, train_size=.80, random_state=random_state)


    if not (weights is None):
        dtrain = xgb.DMatrix(X_train, y_train, missing=np.nan, weight=w_train)
        dvalid = xgb.DMatrix(X_valid, y_valid, missing=np.nan, weight=w_valid)
    
    else:
        dtrain = xgb.DMatrix(X_train, y_train, missing=np.nan)
        dvalid = xgb.DMatrix(X_valid, y_valid, missing=np.nan)
    
    dtest = xgb.DMatrix(X_test, missing=np.nan)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params_dict, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose)

    # print("Validating...")
    check = gbm.predict(dtest, ntree_limit=gbm.best_iteration)

    score = None
    if not (score_func is None):
        score = score_func(Y_test, check)


    def create_feature_map(columns):
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in columns:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1

        outfile.close()

    create_feature_map(columns)

    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    sum_all = sum([i[1] for i in importance])
    importance = [(i[0], 100 * i[1] / sum_all) for i in importance]

    plot_imp(importance[-10:], "xgb")

    # for i in importance[-10:]:
    #   print("{} \n".format(i))

    # df = pd.DataFrame(importance[-10:], columns=['feature', 'fscore'])

    # fig = plt.figure()
    # df.plot()
    # df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 16))
    # plt.title('XGBoost Feature Importance')
    # plt.xlabel('relative importance')
    # plt.gcf().savefig('feature_importance_xgb.png')

    return check, score






def run_xgb_cv(X_learn=None, Y_learn=None, X_test=None, Y_test=None, dict_cv={'md': range(2, 16, 2),
                                                                           'ss': np.linspace(0.1, 0.9, 5),
                                                                           'eta': np.linspace(0.1, 1, 5),
                                                                           'csbt': np.linspace(0.1, 1, 5),
                                                                           'gamma': np.linspace(0, 1, 5)},
                                                                 score_func=None, columns=None, nb_trials=5, nbr=1000, patience=500,
                                                                 verbose=None, random_state=0):


	# X_learn_local, Y_learn_local = X_learn[[range(int(0.75 * X_learn.shape[0]))], :], Y_learn[[range(int(0.75 * X_learn.shape[0]))]]
	# X_test_local, Y_test_local = X_learn[[range(int(0.75 * X_learn.shape[0]), X_learn.shape[0])], :], Y_test[[range(int(0.75 * X_learn.shape[0]); , X_learn.shape[0])]]

    X_learn_local, X_test_local, Y_learn_local, Y_test_local = cross_validation.train_test_split(X_learn, Y_learn, train_size=.80, random_state=random_state)


    res = {}
    for i in range(nb_trials):

        kw = {k: random.choice(dict_cv[k]) for k in dict_cv.keys()}

        preds, score_all = run_xgb_fixed(X_learn_local, Y_learn_local, X_test_local, Y_test_local, score_func=score_func, columns=columns, verbose=False,
                                   nbr=nbr,
                                   # md=md, ss=ss, eta=eta, csbt=csbt
                                   **kw
                                   )

        res[str(kw['md']) + '-' + str(kw['ss']) + '-' + str(kw['eta']) + '-' + str(kw['csbt']) + '-' + str(kw['gamma'])] = score_all
        print("trial {}, score {}".format(i, score_all))

    best_ii = min(res.items(), key=operator.itemgetter(1))[0]
    best_score = min(res.items(), key=operator.itemgetter(1))[1]

    best_dic = dict(md=float(best_ii.split('-')[0]),
                    ss=float(best_ii.split('-')[1]),
                    eta=float(best_ii.split('-')[2]),
                    csbt=float(best_ii.split('-')[3]),
                    gamma=float(best_ii.split('-')[4]))

    y_preds, score = run_xgb_fixed(X_learn, Y_learn, X_test, Y_test,
				    			 score_func=score_func,
				    			 columns=columns,
                                 nbr=nbr,
                                 patience=patience,
                                 verbose=verbose,
				    			 **best_dic
				    			 )

    return y_preds, score




