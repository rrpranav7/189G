from base.stage_4_code.Dataset_Loader import Dataset_Loader
from base.stage_4_code.Method_RNN import Method_RNN
from base.stage_4_code.Result_Saver import Result_Saver
from base.stage_4_code.Setting_Train_Test import Setting_Train_Test
from base.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
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
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
    data_obj.dataset_source_file_name = 'train'

    vocab_size = 1000
    embedding_dim = 128
    num_classes = 2

    method_obj = Method_RNN('Recurrent Neural Network', '', vocab_size)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
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
    

    