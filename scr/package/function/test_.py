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
    test_image_path = '../../storage/data/tongue/all'

    # 图像读入测试
    assert isinstance(load_image(os.path.join(test_image_path, '002747.jpg'), 'RGB'), np.ndarray)
    
    # 计算统计量测试
    #print(statistics(os.path.join(test_image_path, '002747.jpg'), 'RGB'))
    #print(statistics(os.path.join(test_image_path, '002649.jpg'), 'RGB'))

    # 批量处理测试
    #print(batch_processing(test_image_path, 'RGB'))
    #print(batch_processing(test_image_path, 'HSV'))
    #print(np.array(batch_processing(test_image_path, 'HSV')).shape)
    #print(np.array(batch_processing(test_image_path, 'RGB')).shape)
    return

"""
 # Describe :  测试models_training
 # Parameter :
 # Return :     
""" 
def test_models():
    #model_training()
    output = {key.split('\\')[-1]: value for key, value in model_predictions('HSV').items() if value == -1}
    print(output) #9-8  #15-5   
    count = len(output)
    print(f"字典的个数: {count}")

    return 