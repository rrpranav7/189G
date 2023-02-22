'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from base.base_class.setting import setting
from base.stage_2_code.Dataset_Loader import Dataset_Loader
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import torch

class Setting_Train_Test(setting):
    fold = 3

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        X_train, y_train = loaded_data['X'], loaded_data['y']

        f = open(self.dataset.dataset_source_folder_path + self.dataset.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        X_test, y_test = [], []
        for pair in data['test']:
            #pa = np.array(pair['image'])
            #pa = np.dot(pa[..., :3], [0.2989, 0.5870, 0.1140])
            X_test.append(pair['image'])
            y_test.append(pair['label'])

        X_test = np.transpose(np.array(X_test), (0, 3, 1, 2))
        X_test = torch.from_numpy(X_test).float()
        X_test, y_test = X_test, np.array(y_test)-1


        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

