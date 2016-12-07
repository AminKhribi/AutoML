from load_to_pkl_support import read_file

import numpy as np

from process import *

class postprocess_cv(luigi.Task):

	is_local = luigi.Parameter()

	reg = luigi.Parameter()
	list_cv = luigi.ListParameter()

	csv_name = luigi.Parameter()

	def requires(self):
		return [process(is_local=self.is_local, reg=self.reg, dict_param=i) for i in self.list_cv] 


	def output(self):
		return [luigi.LocalTarget("{}_{}.csv".format(self.csv_name, str(self.reg) + str(reduce(lambda x, y: str(x) + '_' + str(y), i.values()))), "w") \
			    for i in self.list_cv]


	def run(self):

		ids_test = read_file('ids_test')

		for i in self.list_cv:

			name_i = str(self.reg) + str(reduce(lambda x, y: str(x) + '_' + str(y), i.values()))
			preds = np.load("preds_{}.npy".format(name_i))


			########
			# OUTPUT

			print('writing output')


			text_file = open("{}_{}.csv".format(self.csv_name, name_i), "w")
			print('id;cible', file=text_file)

			for k, v in zip(ids_test, preds):
				print(int(float(k)), ';', v, file=text_file)

			text_file.close()

