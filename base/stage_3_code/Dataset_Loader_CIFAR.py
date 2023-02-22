'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from base.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        print('training set size:', len(data['train']), 'testing set size:', len(data['test']))
        X, y = [], []
        for pair in data['train']:
            #pa = np.array(pair['image'])
            #pa = np.dot(pa[..., :3], [0.2989, 0.5870, 0.1140])
            X.append(pair['image'])
            y.append(pair['label'])

        X = np.transpose(np.array(X), (0, 3, 1, 2))
        X = torch.from_numpy(X).float()
        X, y = X, np.array(y)


        # Use grayscale_image_tensor in your model for prediction
        return {'X': X, 'y': y}