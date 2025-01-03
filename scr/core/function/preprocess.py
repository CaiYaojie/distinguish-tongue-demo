import numpy as np
import pandas as pd
import cv2
import os
"""
 # Describe :   根据参数加载一张RGB或HSV的图像
 # Parameter :
 #      parameter 1:    img_path 图像路径
 #      parameter 2:    type 载入的图像类型'RGB'或者'HSV',默认值为RGB
 # Return :     返回一个归一化后的ndarry三维数组
""" 
def load_image(img_path, type = 'RGB'):
    if ((type != 'HSV') and (type != 'RGB')) :
        raise Exception('type的参数不能为: {}, 参数只能是RGB或HSV'.format(type))
    img = cv2.imread(img_path)

    if (type == 'HSV') :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    return img


"""
 # Describe :   自定义一个统计函数,用于计算最大值,最小值,平均数,中位数,方差其中一种
 # Parameter :
        parameter 1:    img_array 图像矩阵
        parameter 2:    type 图像类型 
 # Return :     feature_df 返回对应统计特征集
""" 
def statistics(img_path, type ='RGB'):
    img_array = load_image(img_path, type)
    if (type == 'RGB'):
        channels = ['g', 'b', 'r']
        features = {}
        for i, channel in enumerate(channels):
            features[f'mean_{channel}'] = img_array[:, :, i].mean()
            features[f'max_{channel}'] = img_array[:, :, i].max()
            features[f'min_{channel}'] = img_array[:, :, i].min()
            features[f'median_{channel}'] = np.median(img_array[:, :, i])
            features[f'std_{channel}'] = img_array[:, :, i].std()

    elif (type == 'HSV'):
        channels = ['h', 's', 'v']
        features = {}
        for i, channel in enumerate(channels):
            features[f'mean_{channel}'] = img_array[:, :, i].mean()
            features[f'max_{channel}'] = img_array[:, :, i].max()
            features[f'min_{channel}'] = img_array[:, :, i].min()
            features[f'median_{channel}'] = np.median(img_array[:, :, i])
            features[f'std_{channel}'] = img_array[:, :, i].std()
    
    #将特征集转换为一个列表
    features = list(features.values())
    features = np.array(features)
    return features
 

"""
 # Describe :   批量预处理图片
 # Parameter :
        parameter 1:    folder 图像处理的文件夹
        parameter 2:    type 图像类型 
 # Return :     img_features 返回批量预处理图片的对应统计特征集
"""  
def batch_processing(folder, type = 'RGB'):
    img_features=[]
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        statistic = statistics(img_path, type)
        if statistic is not None:
           img_features.append(statistic)
        else:
             print(f"Error loading image {img_path}")
    return np.array(img_features)



        


