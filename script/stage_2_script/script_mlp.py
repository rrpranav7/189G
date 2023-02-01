from base.stage_2_code.Dataset_Loader import Dataset_Loader
from base.stage_2_code.Method_MLP import Method_MLP
from base.stage_2_code.Result_Saver import Result_Saver
from base.stage_2_code.Setting_Train_Test import Setting_Train_Test
from base.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('train ', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name = 'train.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('Train Test present', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, sd_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ', mean_score['accuracy'])
    print('MLP Precision: ', mean_score['precision'])
    print('MLP Recall: ', mean_score['recall'])
    print('MLP F1 score: ', mean_score['F1 Score'])
    print('************ Finish ************')
    # ------------------------------------------------------
    

    