# coding: utf-8
###
 # @file   datasets.py
 # @author Arsany Guirguis  <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright (c) 2019-2020 Arsany Guirguis.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in all
 # copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 # @section DESCRIPTION
 #
 # Datasets management and partitioning.
###

#!/usr/bin/env python

import os
import pathlib
import torch
from random import Random
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


datasets_list = ['mnist', 'cifar10', 'pima']
MNIST = datasets_list.index('mnist')
CIFAR10 = datasets_list.index('cifar10')
PIMA = datasets_list.index('pima')

class PimaDiabetesDataset(torch.utils.data.Dataset):
    """ Pima Indians Diabetes Database
    Predict the onset of diabetes based on diagnostic measures

    See https://www.kaggle.com/uciml/pima-indians-diabetes-database
    """
    TRAIN_SPLIT = 500
    TEST_SPLIT = 200

    def __init__(self, train, train_size=None, transform=None, target_transform=None):
        libdir = os.path.dirname(__file__)
        raw = pd.read_csv(os.path.join(libdir, 'pima_diabetes.csv'))

        train_split = self.TRAIN_SPLIT
        if train_size is not None and train_size < train_split:
            train_split = train_size

        logger.debug(f"Using {train_split} samples for training")

        raw = raw[:train_split] if train else raw[-self.TEST_SPLIT:]

        data, targets = raw.iloc[:, :-1], raw.iloc[:, -1]
        data -= data.mean(axis=0)
        data /= data.std(axis=0)

        self.data= np.array(data).astype('float32')
        self.targets = np.array(targets).astype('float32').reshape(-1, 1)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        val, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            val = self.transform(val)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return val, target


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
       """ Constructor of Partition Object
           Args
           data		dataset needs to be partitioned
           index	indices of datapoints that are returned
        """
       self.data = data
       self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """ Fetching a datapoint given some index
	    Args
            index	index of the datapoint to be fetched
        """
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        """ Constructor of dataPartitioner object
	    Args
	    data	dataset to be partitioned
	    sizes	Array of fractions of each partition. Its contents should sum to 1
	    seed	seed of random generator for shuffling the data
	"""
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]

        for frac in sizes:
            part_len = int(frac * data_len)
            tmp = indexes[0:part_len]
            rng.shuffle(tmp)
            self.partitions.append(tmp)
            indexes = indexes[part_len:]

    def use(self, partition):
        """ Fetch some partition in the dataset
	    Args
	    partition	index of the partition to be fetched from the dataset
	"""
        return Partition(self.data, self.partitions[partition])

class DatasetManager(object):
    """ Manages training and test sets"""

    def __init__(self, dataset, minibatch, num_workers, size, rank, train_size=None):
        """ Constrctor of DatasetManager Object
	    Args
		dataset		dataset name to be used
		minibatch	minibatch size to be employed by each worker
		num_workers	number of workers employed in the setup
		size		total number of nodes in the deployment
		rank		rank of the current worker
                train_size      number of training samples to use (all if None)
	"""
        if dataset not in datasets_list:
            raise Exception("Existing datasets are: ", datasets_list)
        self.dataset = datasets_list.index(dataset)
        self.batch = minibatch * num_workers
        self.num_workers = num_workers
        self.num_ps = size - num_workers
        self.rank = rank
        self.train_size = train_size

    def fetch_dataset(self, train=True):
        """ Fetch train or test set of some dataset
		Args
		dataset		dataset index from the global "datasets" array
		train		boolean to determine whether to fetch train or test set
	"""
        homedir = str(pathlib.Path.home())
        if self.dataset == MNIST:
            return datasets.MNIST(
              homedir+'/data',
              train=train,
              download=True,
              transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307, ), (0.3081, ))
              ]))

        if self.dataset == CIFAR10:
            if train:
              transforms_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
#                transforms.Resize(299),		#only use with inception
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
#		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),])
              return datasets.CIFAR10(
               homedir+'/data',
               train=True,
               download=True,
               transform=transforms_train)
            else:
              transforms_test = transforms.Compose([
#                transforms.Resize((299,299)),			#only use with inception
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
#		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
              return datasets.CIFAR10(
                homedir+'/data',
                train=False,
                download=True,
                transform=transforms_test)

#            return datasets.CIFAR10(
#               homedir+'/data',
#               train=train,
#               download=train,
#               transform=transforms.Compose(
#                  [transforms.ToTensor(),
#                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        if self.dataset == PIMA:
            logger.debug(f"Using Pima Indians Diabetes dataset")
            return PimaDiabetesDataset(
                    train=train,
                    train_size=self.train_size,
                    )

    def get_train_set(self):
        """ Fetch my partition of the train set"""
        train_set = self.fetch_dataset(train=True)
        size = self.num_workers
        bsz = int(self.batch / float(size))
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_set, partition_sizes)
        partition = partition.use(self.rank - self.num_ps)
        logger.debug(f"Using batch size = {bsz}")
        train_set = torch.utils.data.DataLoader(
            partition, batch_size=bsz, shuffle=False, pin_memory=True, num_workers=2)
        return [sample for sample in train_set]

    def get_test_set(self):
        """ Fetch test set, which is global, i.e., same for all entities in the deployment"""
        test_set = self.fetch_dataset(train=False)
        test_set = torch.utils.data.DataLoader(test_set, batch_size=100, #len(test_set),
		 pin_memory=True, shuffle=False, num_workers=2)
        return test_set
