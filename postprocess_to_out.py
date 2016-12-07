from load_to_pkl_support import read_file

import numpy as np

from process import *

class postprocess(luigi.Task):

	is_local = luigi.Parameter()

	reg = luigi.Parameter()
	dict_param = luigi.DictParameter(significant=False, default=None)

	csv_name = luigi.Parameter()

	def requires(self):
		return process(is_local=self.is_local, reg=self.reg, dict_param=self.dict_param) 


	def output(self):
		return luigi.LocalTarget("{}.csv".format(self.csv_name))


	def run(self):

		preds = np.load('preds.npy')

		ids_test = read_file('ids_test')


		########
		# OUTPUT

		print('writing output')


		text_file = open("{}.csv".format(self.csv_name), "w")
		print('id;cible', file=text_file)

		for k, v in zip(ids_test, preds):
			print(int(float(k)), ';', v, file=text_file)

		text_file.close()
