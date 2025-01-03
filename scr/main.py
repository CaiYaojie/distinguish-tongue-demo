import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scr.core.function.models_stuff import *

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 模型参数
    picture_type = ['RGB', 'HSV']
    model_type = ['IF', 'HM']
    dataset = {
        'normal': os.path.join(current_dir, 'storage/data/tongue/normal'), 
        'abnormal':  os.path.join(current_dir, 'storage/data/tongue/abnormal')
    }
    test_imgs = os.path.join(current_dir, 'storage/cache/test_imgs/')
    # 模型训练
    for i in picture_type:
        for j in model_type:
            if (j == 'IF'):
                model_training_IF(dataset['normal'], i)
            else:
                model_training_HM(dataset['normal'], i)
    
    # 模型评估
    random_test_imgs(dataset['normal'], dataset['abnormal'], test_imgs, 'normal', 'abnormal', 15)
    for i in model_type:
        for j in picture_type:
            print(f'训练模型是{i}, 图像模型是{j}')
            model_evaluate(os.path.join(test_imgs, 'normal'), os.path.join(test_imgs, 'abnormal'), 'normal', 'abnormal',
                           i, j)
            print('\n')

    shutil.rmtree(test_imgs)
    return

if __name__ == '__main__':
    main()