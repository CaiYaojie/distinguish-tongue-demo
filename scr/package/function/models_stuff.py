import pickle
from scipy import stats
from sklearn.ensemble import IsolationForest
from scr.package.function.preprocess import*

"""
 # Describe :   使用孤立森林模型拟合模型
 # Parameter :  type 图像模型 
 # Return :     
"""  
def model_training_ML(type = 'RGB'):
    # 导入训练集
    normal_data = np.load(f'../../storage/cache/img_features_{type}.npy')
   
    #模型训练
    model = IsolationForest(contamination=0.145, random_state=27)
    model_IF = model.fit(normal_data)
    
    #保存模型
    with open(f'../../storage/cache/model_{type}.pkl', 'wb') as file:
       pickle.dump(model_IF, file)
    return


"""
 # Describe :   使用孤立森林模型进行预测(机器预测)
 # Parameter :
        parameter 1:    predict_path 预测数据集路径
        parameter 2:    type 图像类型 
 # Return :     predictions 返回一个预测结果字典
"""  
def model_predictions_ML(predict_path, type):
    #字典分类器
    dim_mapping = {
    'RGB': 15,
    'HSV': 9
    }
    dim = dim_mapping.get(type, None) 

    # 加载模型
    with open(f'../../storage/cache/model_{type}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    result={}

    # 模型预测
    for filename in os.listdir(predict_path):
        img = os.path.join(predict_path, filename)
        img_features = np.array(statistics(img, type)).reshape(1,dim)
        result[filename] = loaded_model.predict(img_features)

    return result


"""
 # Describe :   人工设计和训练
 # Parameter :  
 #      parameter1 : training_path 训练集路径
 #      parameter2 : type 图像模型
 # Return :     weight_list 返回训练完的权值
"""  
def model_training_HM(training_path, type):
    """
    # 模型创建: 根据统计特征人工设计一个模型
    # 模型原理: 计算各训练集各参数的的统计特征, 根据模统计特征计算出95%的置信区间,
                再根据训练集各参数在95%的置信区间的个数, 归一化作为权值当作模型
    """  
    # 导入训练集
    train_set = batch_processing(training_path, type)
    trained_range = []
    num_columns = train_set.shape[1]# 列数

    # 95%的置信区间参数
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df = num_columns - 1)  # t分布的临界值
    
    # 对每一列进行统计特征的求取,并求出每一列95%的置信区间
    for column_index in range(num_columns):
        # 计算统计特征
        column_data = train_set[:, column_index] 
        mean = np.mean(column_data)
        std_dev = np.std(column_data, ddof=1)  
        
        # 计算每个参数的95%的置信区间
        margin_of_error = t_critical * (std_dev / np.sqrt(num_columns))
        confidence_interval = (mean - margin_of_error, mean + margin_of_error)
        trained_range.append(confidence_interval)
    
    # 中间变量
    access_train = []  
    access_train_set = [] 
    train_counter = 0 
    num_columns = len(trained_range)

    # 字典分类器
    dim_mapping = {
    'RGB': 15,
    'HSV': 9
    }
    dim = dim_mapping.get(type, None)  
    
    # 用训练集训练得出权值
    for filename in os.listdir(training_path):
        img = os.path.join(training_path, filename)
        predicted_value = np.array(statistics(img, type)).reshape(1,dim)
        
        #训练样本的参数是否在95%置信区间内{'Y': 1,'N': -1}
        for column_index in range(num_columns):
            if (trained_range[column_index][0] <= predicted_value[0, column_index] 
                <= trained_range[column_index][1]):
                access_train.append(1)
            else:
                access_train.append(-1)   
        
        #训练集的access列表
        access_train_set.append(access_train)
        access_train = []
        train_counter = train_counter + 1

    # 统计各参数通过的个数
    access_train_set = np.array(access_train_set)
    column_count = np.sum(access_train_set, axis=0)

    # 权值归一化
    weight_list = [(x - min(column_count)) / (max(column_count) - min(column_count))  
                   for x in column_count]  
    
    # 模型保存
    np.save(f'../../storage/cache/trained_range_HM_{type}.npy', trained_range)
    np.save(f'../../storage/cache/weight_list_HM_{type}.npy', weight_list)

    return 
    

"""
 # Describe :   人工预测训练
 # Parameter :  
 #      parameter1 : predict_path 预测集路径
 #      parameter2 : type 图像模型
 # Return :     result 返回预测结果,类型是一个字典
"""  
def model_predicting_HM(predict_path, type):
    # 模型导入
    weight_list = np.load(f'../../storage/cache/weight_list_HM_{type}.npy')
    trained_range = np.load(f'../../storage/cache/trained_range_HM_{type}.npy')
    num_columns = len(trained_range)

    # 字典分类器
    dim_mapping = {
    'RGB': 15,
    'HSV': 9
    }
    dim = dim_mapping.get(type, None)  

    # 中间变量
    access = []
    result = {}

    # 判别图像
    for filename in os.listdir(predict_path):
        img = os.path.join(predict_path, filename)
        predicted_value = np.array(statistics(img, type)).reshape(1,dim)
        
        #预测样本的参数是否在95置信区间内{'Y': 1,'N': -1}
        for column_index in range(num_columns):
            if trained_range[column_index][0] <= predicted_value[0, column_index] <= trained_range[column_index][1]:
                access.append(1)
            else:
                access.append(-1)
        
        # 权值与通过率进行加权乘积并求和
        score_perdict = sum(np.multiply(weight_list, access))

        # 预测结果
        result[filename] = -1 if score_perdict < 0 else 1

        # 初始化中间变量, 继续进行预测
        access = []
        score_perdict = 0
    return result
