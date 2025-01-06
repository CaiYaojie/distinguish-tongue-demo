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

    # 模型训练
    for i in picture_type:
        for j in model_type:
            if (j == 'IF'):
                model_training_IF(os.path.join(current_dir, 'storage/cache/test_imgs/normal_train'), i)
            else:
                model_training_HM(os.path.join(current_dir, 'storage/cache/test_imgs/normal_train'), i)
    
    
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

if __name__ == '__main__':
    main()