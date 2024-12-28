import pytest
import os
from .preprocess import *
from .models_stuff import *

"""
 # Describe :  测试load_preprocess模块
 # Parameter :
 # Return :     
""" 
def test_preprocess():
    # 图像读入测试
    #assert isinstance(load_image(os.path.join(test_image_path, '002747.jpg'), 'RGB'), np.ndarray)
    
    # 计算统计量测试
    #print(statistics(os.path.join(test_image_path, '002747.jpg'), 'RGB'))
    #print(statistics(os.path.join(test_image_path, '002649.jpg'), 'RGB'))

    # 批量处理测试
    #print(np.array(batch_processing(test_image_path, 'HSV')).shape)
    #print(np.array(batch_processing(test_image_path, 'RGB')).shape)
    return

"""
 # Describe :  测试models_training
 # Parameter :
 # Return :     
""" 
def test_models():
    # 机器模型训练
    # model_training_ML('RGB')
    # model_training_ML('HSV')

    # 机器模型预测
    # predictions_RGB = model_predictions_ML('../../storage/data/tongue/abnormal', 'RGB')
    # count_error = 0
    # for key, value in predictions_RGB.items():
    #     if value == 1:
    #         count_error += 1
    #         print(key, value)
    # print(f'误判为正常图片有{count_error}个')

    # # 格式化输出
    # predictions_RGB = model_predictions_ML('../../storage/data/tongue/normal', 'RGB')
    # count_error = 0
    # for key, value in predictions_RGB.items():
    #     if value == -1:
    #         count_error += 1
    #         print(key, value)
    # print(f'误判为错误图片有{count_error}个')

    # 人工模型训练及预测
    model_training_HM('../../storage/data/tongue/normal', 'RGB')
    result = model_predicting_HM('../../storage/data/tongue/abnormal', 'RGB')
    for key, value in result.items():
        print(key, value)

    return 
