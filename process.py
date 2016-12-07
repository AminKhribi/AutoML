import pandas as pd
import scipy as sparse
from sklearn import metrics

from functools import reduce

import process_support

from load_to_pkl_support import read_file

from preprocess_to_mtx import *


regs = ['KNN', 'DT', 'RF', 'XGB']



class process(luigi.Task):

	is_local = luigi.Parameter()

	reg = luigi.Parameter()
	dict_param = luigi.DictParameter(default=None)

	def requires(self):
		# return {"X_all": luigi.LocalTarget("X_all.mtx"),
		# 		"Y_all": luigi.LocalTarget("Y_all.npy"),
		# 		"idx_learn": luigi.LocalTarget("idx_learn"),
		# 		"idx_test": luigi.LocalTarget("idx_test"),
		# 		"ids_test": luigi.LocalTarget("ids_test")}
		return preprocess(is_local=self.is_local) 


	def output(self):
		return [luigi.LocalTarget("preds_{}.npy".format(str(self.reg) + str(reduce(lambda x, y: str(x) + '_' + str(y), self.dict_param.values())))),
   				luigi.LocalTarget("ids_test")] 


	def run(self):

		######
		# LOAD

		X_all = io.mmread("X_all_{}.mtx".format(self.is_local)).tocsr()
		Y_all = np.load("Y_all_{}.npy".format(self.is_local))

		all_cols = read_file('all_cols')

		ids_test = read_file('ids_test')

		idx_learn = read_file('idx_learn')
		idx_learn = [int(i) for i in idx_learn]

		idx_test = read_file('idx_test')
		idx_test = [int(i) for i in idx_test]





		############
		# REGRESSION

		if self.reg == 'KNN':
			func_reg = process_support.run_KNN

		elif self.reg == 'DT':
			func_reg = process_support.run_DT

		elif self.reg == 'RF':
			func_reg = process_support.run_RF

		else:
			func_reg = process_support.run_xgb





		X_learn, X_test = X_all[idx_learn], X_all[idx_test]
		# X_learn, X_test = sparse.csr_matrix(X_all.todense()[idx_learn, :]), sparse.csr_matrix(X_all.todense()[idx_test, :])

		Y_learn, Y_test = Y_all[idx_learn], Y_all[idx_test]

		params = {'score_func': metrics.mean_squared_error,
				   'columns': all_cols,
					'verbose': False}

		if self.dict_param:
			params.update(self.dict_param)


		preds, score = func_reg(X_learn=X_learn, Y_learn=Y_learn, X_test=X_test, Y_test=Y_test,
							   **params)






		########
		# OUTPUT

		np.save("preds_{}.npy".format(str(self.reg) + str(reduce(lambda x, y: str(x) + '_' + str(y), self.dict_param.values()))), preds)

