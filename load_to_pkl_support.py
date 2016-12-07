



def clean_missing(df_train, new_cat_columns, idx_learn, idx_test, name, threshold):

    learn_cat_dict = {}
    for f in new_cat_columns:
        learn_cat_dict[f] = sum(df_train.ix[idx_learn][f].isnull()) / len(idx_learn)

    test_cat_dict = {}
    for f in new_cat_columns:
        test_cat_dict[f] = sum(df_train.ix[idx_test][f].isnull()) / len(df_train.ix[idx_test])

    to_keep_cat_learn = [f for f, v in learn_cat_dict.items() if v < threshold]
    to_keep_cat_test = [f for f, v in test_cat_dict.items() if v < threshold]
    new_cat_columns_out = list(set(to_keep_cat_learn).intersection(to_keep_cat_test))
    print("kept {}% of {} columns".format(round(100 * len(new_cat_columns_out) / len(learn_cat_dict), 2), name))

    return learn_cat_dict, test_cat_dict, new_cat_columns_out




def read_file(name):

	filename = open(name, 'r')
	ufile = []
	for i in filename.readlines():
		i = i.replace('\n', '')
		ufile.append(i)
	filename.close()
	return ufile


def write_file(ufile, name):
	filename = open(name, 'w')
	for u in ufile:
	  filename.write("%s\n" % u)
	filename.close()
