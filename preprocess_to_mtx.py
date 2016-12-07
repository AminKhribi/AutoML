import pandas as pd
import numpy as np

import functools

from scipy import io

from preprocess_to_mtx_support import *

from load_to_pkl_support import read_file

from load_to_pkl import *


class preprocess(luigi.Task):

	is_local = luigi.Parameter()

	def requires(self):
		# return {"df": luigi.LocalTarget("df_all_{}.pkl".format(is_local)),
		# 		"idx_learn": luigi.LocalTarget("idx_learn"),
		# 		"idx_test": luigi.LocalTarget("idx_test"),
		# 		"ids_test": luigi.LocalTarget("ids_test"),
		# 		"new_cont_file": luigi.LocalTarget("new_cont_file"),
		# 		"new_cat_file": luigi.LocalTarget("new_cat_file")} 
		return load_data(is_local=self.is_local)


	def output(self):
		return {"X_all": luigi.LocalTarget("X_all_{}.mtx".format(self.is_local)),
				"Y_all": luigi.LocalTarget("Y_all_{}.npy".format(self.is_local)),
				"idx_learn": luigi.LocalTarget("idx_learn"),
                "idx_test": luigi.LocalTarget("idx_test"),
                "ids_test": luigi.LocalTarget("ids_test"),
                "all_cols": luigi.LocalTarget("all_cols")} 


	def run(self):


		#########
		# READING

		print('reading pkl & columns')
		df_all = pd.read_pickle("df_all_{}.pkl".format(self.is_local))

		cont_columns = read_file('new_cont_columns')
		cat_columns = read_file('new_cat_columns')

		target = read_file('target')




		###############
		# OneHotEncoder

		print("OneHotEncoder")

		#
		X_cat, X_pop, cat_cols = one_hot_encoder_wpop_cols(cat_columns, df_all)

		#
		pop_cols = [str(i) + '_pop' for i in cat_columns]

		scp = preprocessing.StandardScaler()
		X_pop = sparse.csr_matrix(scp.fit_transform(X_pop))





		######
		# NORM

		print("cont norm")

		X_cont, cont_cols = normalize(False, cont_columns, df_all)




		#########
		# COMBINE

		all_cols = cat_cols + pop_cols + cont_cols

		X_all = sparse.hstack((X_cat, X_pop, X_cont)).tocsr()
		Y_all = df_all[target].values





		######
		# SAVE

		io.mmwrite("X_all_{}.mtx".format(self.is_local), X_all)
		np.save("Y_all_{}.npy".format(self.is_local), Y_all)

		write_file(all_cols, 'all_cols')


