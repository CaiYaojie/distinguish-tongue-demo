import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scr.core.function.models_stuff import *
from scr.core.function.data_enhancement import *

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 模型参数
    picture_type = ['RGB', 'HSV']
    model_type = ['IF', 'HM']
    dataset = {
        'normal': os.path.join(current_dir, 'storage/data/tongue/normal'), 
        'abnormal':  os.path.join(current_dir, 'storage/data/tongue/abnormal'),
        'abnormal_augmented':  os.path.join(current_dir, 'storage/data/tongue/abnormal_augmented')
    }
    result = {}
    # 数据增强
    augment_images(dataset['abnormal'], dataset['abnormal_augmented'], 7)

    test_imgs = os.path.join(current_dir, 'storage/cache/test_imgs/')
    split_image_dataset(dataset['normal'], test_imgs, 0.7, 'normal')
    split_image_dataset(dataset['abnormal_augmented'], test_imgs, 0.7, 'abnormal')

    # # 模型训练
    # for i in picture_type:
    #     for j in model_type:
    #         if (j == 'IF'):
    #             model_training_IF(os.path.join(current_dir, 'storage/cache/test_imgs/normal_train'), i)
    #         else:
    #             model_training_HM(os.path.join(current_dir, 'storage/cache/test_imgs/normal_train'), i)
    
    
    # 模型评估
    for i in model_type:
        for j in picture_type:
            print(f'训练模型是{i}, 图像模型是{j}')
            result[f'{i}_{j}'] = model_evaluate(os.path.join(current_dir, 'storage/cache/test_imgs/normal_test'), 
                            os.path.join(current_dir, 'storage/cache/test_imgs/abnormal_test'), 
                            'normal', 'abnormal', i, j)
            print('\n')

    best_model_save(result['IF_HSV'], os.path.join(current_dir, 'storage/cache/IF_model'), 
                    os.path.join(current_dir, 'storage/best_model/IF_model'), 'IF')
    best_model_save(result['HM_HSV'], os.path.join(current_dir, 'storage/cache/HM_model'), 
                    os.path.join(current_dir, 'storage/best_model/HM_model'), 'HM')
    shutil.rmtree(dataset['abnormal_augmented'])
    shutil.rmtree(test_imgs)
    
    return

# 训练模型是IF, 图像模型是RGB
# {'TP': 22, 'TN': 23, 'FP': 3, 'FN': 3}
# A = 0.8823529411764706, P = 0.88, R = 0.88, F = 0.88, MCC = 505.98615384615385


# 训练模型是IF, 图像模型是HSV
# {'TP': 20, 'TN': 26, 'FP': 0, 'FN': 5}
# A = 0.9019607843137255, P = 1.0, R = 0.8, F = 0.888888888888889, MCC = 520.0


# 训练模型是HM, 图像模型是RGB
# {'TP': 13, 'TN': 21, 'FP': 5, 'FN': 12}
# A = 0.6666666666666666, P = 0.7222222222222222, R = 0.52, F = 0.6046511627906976, MCC = 272.90343909008294


# 训练模型是HM, 图像模型是HSV
# {'TP': 22, 'TN': 21, 'FP': 5, 'FN': 3}
# A = 0.8431372549019608, P = 0.8148148148148148, R = 0.88, F = 0.8461538461538461, MCC = 461.97688749182396
if __name__ == '__main__':
    # for i in range(100):
        # print(i)
    main()