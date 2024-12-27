import pickle
from sklearn.ensemble import IsolationForest
from scr.package.function.preprocess import*

"""
 # Describe :   使用孤立森林模型拟合模型
 # Parameter :
 # Return :     
"""  
def model_training():
    # 导入训练集
    normal_data_RGB = np.load('../../storage/cache/img_features_RGB.npy')
    normal_data_HSV = np.load('../../storage/cache/img_features_HSV.npy')
    #模型拟合
    model = IsolationForest(contamination=0.145, random_state=27)
    model_RGB = model.fit(normal_data_RGB)
    model_HSV = model.fit(normal_data_HSV)
    
    #保存模型
    with open('../../storage/cache/model_RGB.pkl', 'wb') as file:
       pickle.dump(model_RGB, file)
    with open('../../storage/cache/model_HSV.pkl', 'wb') as file:
        pickle.dump(model_HSV, file)

"""
 # Describe :   使用孤立森林模型进行预测
 # Parameter :
        parameter 1:    type 图像类型 
 # Return :     predictions 返回一个预测结果字典
"""  
def model_predictions(type):
    test_image_path = '../../storage/data/tongue/abnormal'
    predictions = {}
    # 加载模型
    if(type == 'RGB'):
        with open('../../storage/cache/model_RGB.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        for filename in os.listdir(test_image_path):
            img = os.path.join(test_image_path, filename)
            img_features = np.array(statistics(img, 'RGB')).reshape(1,15)
            predictions[img] = loaded_model.predict(img_features)
            
    elif (type == 'HSV'):
        with open('../../storage/cache/model_HSV.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        for filename in os.listdir(test_image_path):
            img = os.path.join(test_image_path, filename)
            img_features = np.array(statistics(img, 'HSV')).reshape(1,15)
            predictions[img] = loaded_model.predict(img_features)

    return predictions
