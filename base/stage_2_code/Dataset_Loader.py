'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pandas as pd
from base.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        f = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name, header=None)
        X = f.iloc[:,1:]
        y = f.iloc[:,0]
        return {'X': X, 'y': y}