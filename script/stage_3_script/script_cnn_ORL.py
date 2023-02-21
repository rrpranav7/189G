from base.stage_3_code.Dataset_Loader_ORL import Dataset_Loader
from base.stage_3_code.Method_CNN_ORL import Method_CNN
from base.stage_3_code.Result_Saver import Result_Saver
from base.stage_3_code.Setting_Train_Test_ORL import Setting_Train_Test
from base.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch

if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('ORLL', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'ORL'


    method_obj = Method_CNN('Convolution Neural Network', '', 40, 1)


    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
    result_obj.result_destination_file_name = 'prediction_result'


    setting_obj = Setting_Train_Test('Train Test present', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, sd_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ', mean_score['accuracy'])
    print('CNN Precision: ', mean_score['precision'])
    print('CNN Recall: ', mean_score['recall'])
    print('CNN F1 score: ', mean_score['F1 Score'])
    print('************ Finish ************')
