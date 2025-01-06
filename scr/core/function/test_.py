import pytest
import numpy as np
from .preprocess import *
from .models_stuff import *
from .data_enhancement import *

"""
 # Describe :  测试load_preprocess模块
 # Parameter :
 # Return :     
""" 
def test_preprocess():
    # 图像读入测试
    #assert isinstance(load_image(os.path.join(test_image_path, '002747.jpg'), 'RGB'), np.ndarray)
    
    # 计算统计量测试
    # print(statistics(os.path.join('../../storage/data/tongue/normal', '002747.jpg'), 'RGB'))
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
    # 机器模型训练及预测
    # model_training_IF('../../storage/data/tongue/normal', 'HSV')
    # result = model_predict_ML('../../storage/data/tongue/abnormal', 'RGB')
    # count_error = 0
    # for key, value in result.items():
    #     if value == 1:
    #         count_error += 1
    #         print(key, value)
    # print(f'误判为正常图片有{count_error}个')

    # 人工模型训练及预测
    # model_training_HM('../../storage/data/tongue/normal','RGB')
    # result = model_predict_HM('../../storage/data/tongue/abnormal', 'HSV')
    # count_error = 0
    # for key, value in result.items():
    #    if value == 1:
    #         count_error += 1
    #         print(key, value)
    # print(f'误判为正常图片有{count_error}个')

    # 机器模型评估测试
    evaluations = model_evaluate('../../storage/data/tongue/normal', 
                   '../../storage/data/tongue/abnormal', 'normal', 'abnormal',
                   'IF', 'HSV')

    # 人工模型评估测试
    evaluations = model_evaluate('../../storage/data/tongue/normal_', 
                   '../../storage/data/tongue/abnormal', 'normal', 'abnormal',
                   'HM', 'RGB')
    return 


"""
 # Describe :  测试data_enhancement
 # Parameter :
 # Return :     
""" 
def test_models():
    # augment_images('../../storage/data/tongue/abnormal', '../../storage/data/tongue/abnormal_augmented', 5)
    return

#3参数
# HSV
# test_.py 
# {'TP': 106, 'TN': 17, 'FP': 1, 'FN': 18}
# 0.8661971830985915 0.13380281690140844 0.9177489177489178
# {'TP': 85, 'TN': 17, 'FP': 1, 'FN': 39}
# 0.7183098591549296 0.28169014084507044 0.8095238095238095
# RGB
# test_.py 
# {'TP': 106, 'TN': 17, 'FP': 1, 'FN': 18}
# 0.8661971830985915 0.13380281690140844 0.9177489177489178
# {'TP': 66, 'TN': 16, 'FP': 2, 'FN': 58}
# 0.5774647887323944 0.4225352112676056 0.6875

# 5参数
# HSV
# test_.py .{'TP': 106, 'TN': 0, 'FP': 0, 'FN': 18}
# 0.8732394366197183 0.1267605633802817 0.9217391304347826
# {'TP': 115, 'TN': 9, 'FP': 9, 'FN': 9}
# 0.8732394366197183 0.1267605633802817 0.9274193548387096
# 
# RGB
# test_.py .{'TP': 106, 'TN': 18, 'FP': 0, 'FN': 18}
# 0.8732394366197183 0.1267605633802817 0.9217391304347826
# {'TP': 105, 'TN': 11, 'FP': 7, 'FN': 19}
# 0.8169014084507042 0.18309859154929578 0.8898305084745762