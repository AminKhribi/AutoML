import luigi

import pandas as pd

import functools
import random

from load_to_pkl_support import *


class input_files(luigi.ExternalTask):

    def output(self):
        return {"train_file": luigi.LocalTarget("train.csv"),
                "test_file": luigi.LocalTarget("test.csv"),
                "cat_file": luigi.LocalTarget("cat_columns"),
                "cont_file": luigi.LocalTarget("cont_columns"),
                "target": luigi.LocalTarget("target")}


class load_data(luigi.Task):

    is_local = luigi.Parameter()
    n_sample = luigi.IntParameter(significant=False, default=100000)

    def requires(self):
        return input_files()


    def output(self):
        return {"df": luigi.LocalTarget("df_all_{}.pkl".format(self.is_local)),
                "idx_learn": luigi.LocalTarget("idx_learn"),
                "idx_test": luigi.LocalTarget("idx_test"),
                "ids_test": luigi.LocalTarget("ids_test"),
                "new_cont_file": luigi.LocalTarget("new_cont_columns"),
                "new_cat_file": luigi.LocalTarget("new_cat_columns")} 


    def run(self):

        cat_columns = read_file('cat_columns')
        print("CAT COLUMNS: ", cat_columns)

        cont_columns = read_file('cont_columns')
        print("CONT COLUMNS: ", cont_columns)


        target = read_file('target')
        print("TARGET: ", target)

        cols_learn = list(set(cat_columns).union(cont_columns).union(target))
        cols_test = list(set(cat_columns).union(cont_columns))




        #
        ff = lambda x: str if x in cat_columns else float

        df_train = pd.read_csv('train.csv', sep=';', dtype={k: ff(k) for k in cols_learn})
        df_test = pd.read_csv('test.csv', sep=';', dtype={k: ff(k) for k in cols_test})

        n_train = df_train.shape[0]
        n_test = df_test.shape[0]
        print("TRAIN: #{} --- TEST: #{}".format(n_train, n_test))





        #######
        # Split

        if self.is_local:

            df_all = df_train
            df_all = df_all.reset_index(drop=True)

            indexes = list(df_all.index)

            random.shuffle(indexes)

            indexes = indexes[:self.n_sample]

            idx_learn = indexes[: int(0.75 *  len(indexes))]
            idx_test = indexes[int(0.75 *  len(indexes)):]


        else:

            df_all = pd.concat([df_train, df_test])
            df_all = df_all.reset_index(drop=True)

            idx_learn = range(n_train)
            idx_test = range(n_train, n_test + n_train)


        # we ll be writing output with this ids order
        if 'id' in df_all.columns:
            ids_test = df_all.ix[idx_test]['id']

        else:
            ids_test = range(df_all.shape[0])


        print("FINAL:  TRAIN: #{} --- TEST: #{}".format(len(idx_learn), len(idx_test)))




        ##########
        # ANALYSIS

        print('missing')
        for f in set(cat_columns).union(cont_columns):
            print("{} missing {}%".format(f, round(100 * sum(df_all[f].isnull()) / df_all.shape[0], 2)))




        #####
        # CAT

        mod_dict = {k: df_all[k].nunique() for k in cat_columns}
        print("MODALITIES")
        for i, j in mod_dict.items():
            print("{}: {} \n".format(i, j))


        inter_dict = {k: v for k, v in zip(cat_columns, list(map(lambda col: len(set(df_all[col].unique()).intersection(df_test[col].unique())) / df_all[col].nunique(), cat_columns)))}


        print("INTERSECTION")
        for i, j in inter_dict.items():
            print("{}: {} \n".format(i, j))


        ######
        # CONT

        print("CONT")
        print(df_all[cont_columns].describe())








        #################
        # FILTER FEATURES

        print("clean_missing")

        threshold = 0.5

        learn_cat_dict, test_cat_dict, new_cat_columns = clean_missing(df_all, cat_columns, idx_learn, idx_test, 'cat', threshold)
        print("new cat columns: ", new_cat_columns)

        learn_cont_dict, test_cont_dict, new_cont_columns = clean_missing(df_all, cont_columns, idx_learn, idx_test, 'cont', threshold)
        print("new cont columns: ", new_cont_columns)








        ########
        # OUTPUT

        print("writing output")

        df_all.to_pickle("df_all_{}.pkl".format(self.is_local))

        write_file(idx_learn, 'idx_learn')
        write_file(idx_test, 'idx_test')

        write_file(ids_test, 'ids_test')

        write_file(new_cat_columns, 'new_cat_columns')
        write_file(new_cont_columns, 'new_cont_columns')















