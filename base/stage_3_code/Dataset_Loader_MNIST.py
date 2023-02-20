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

            X.append(pair['image'])
            y.append(pair['label'])

        X, y = torch.from_numpy(np.array(X)).unsqueeze(1).float(), np.array(y) #unsqueeze(1)


        # Use grayscale_image_tensor in your model for prediction
        return {'X': X, 'y': y}