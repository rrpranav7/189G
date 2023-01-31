'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from base.base_class.setting import setting
from base.stage_2_code.Dataset_Loader import Dataset_Loader
from sklearn.model_selection import train_test_split
import numpy as np


class Setting_Train_Test(setting):
    fold = 3

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        X_train, y_train = loaded_data['X'], loaded_data['y']

        data_obj = Dataset_Loader('test', '')
        data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
        data_obj.dataset_source_file_name = 'test.csv'

        test_dataset = data_obj.load()

        X_test, y_test = test_dataset['X'], test_dataset['y']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

