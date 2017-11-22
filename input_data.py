#coding = utf-8

"""Functions for reading local data for survival analyze."""
from __future__ import print_function
import gzip
import os
import urllib
import numpy
import pandas

# maybe need to transform Data
def do_some_trans(X_data):
	pass
	return X_data

class DataSet(object):
	"""docstring for DataSet"""

	"""class init
	Params:
	    X_data: a DataFrame
	    labels: DataFrame, two columns as ['T', 'E']

	Returns:
	    DataSet object
	"""
	def __init__(self, X_data, labels):
		assert len(X_data) == len(labels), ("X_data: (%d, %d); labels: (%d, %d)" 
			% (len(X_data), len(X_data.columns), len(labels), len(labels.columns)))
		self._num_examples = len(X_data)
		X_data = do_some_trans(X_data)
		self._X_data = X_data
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def X_data(self):
		return self._X_data

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	"""Return [start, end] examples from this data set.
	Params:
	    start: start index
	    end: end index

	Returns:
	    [start, end] examples should be sorted by labels.'T' DESC!
	    X_data_batch: np.ndarray
	    labels_batch: np.ndarray
	"""
	def prepare_data(self, start, end):
		assert 'T' in self._labels.columns
		labels_batch = self._labels.iloc[start:end]
		X_data_batch = self._X_data.iloc[start:end]
		labels_batch = labels_batch.sort_values(by = 'T', ascending = False)
		X_data_batch = X_data_batch.loc[labels_batch.index]
		return X_data_batch.as_matrix(), labels_batch.as_matrix()

	"""Return the next `batch_size` examples from this data set.
	Params:
	    batch_size: next `batch_size` examples

	Returns:
	    In survival analyze, the data should be sorted by labels.'T' DESC!
	    X_data_batch: np.ndarray, batch data of X_data
	    labels_batch: np.ndarray, batch data of labels
	"""
	def next_batch(self, batch_size = 500):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm)
			self._X_data = self._X_data.iloc[perm]
			self._labels = self._labels.iloc[perm]
			# start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		# prepare sorted data
		X_data_batch, labels_batch = prepare_data(start, end)
		return X_data_batch, labels_batch

def read_data_sets(train_dir, targets = ['T', 'E'], validation_ratio = 0.0):
	class DataSets(object):
		pass
	data_sets = DataSets()
	# Data should already be shuffled 
	TRAIN_DATA = "train_data.csv"
	TEST_DATA = "test_data.csv"
	local_train = pandas.read_csv(train_dir + TRAIN_DATA)
	local_test = pandas.read_csv(train_dir + TEST_DATA)
	x_cols = [col for col in local_train.columns if col not in targets]
	# if need validation
	if validation_ratio > 0:
		validation_size = int(len(local_train) * validation_ratio)
		local_validation = local_train[:validation_size]
		local_train = local_train[validation_size:]
		data_sets.validation = DataSet(local_validation[x_cols], local_validation[targets])
	data_sets.train = DataSet(local_train[x_cols], local_train[targets])
	data_sets.test = DataSet(local_test[x_cols], local_test[targets])