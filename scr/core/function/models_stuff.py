import pickle
import math
from scipy import stats
from sklearn.ensemble import IsolationForest
from scr.core.function.preprocess import*

"""
 # Describe :   使用孤立森林模型拟合模型
 # Parameter :  
 #      parameter1 train_path 训练数据集路径 
 #      parameter2 type 图像模型 
 # Return :     
"""  
def model_training_IF(train_path ,type = 'RGB'):
    # 导入训练集
    train_set = batch_processing(train_path, type)
   
    #模型训练
    model = IsolationForest(contamination=0.145, random_state=27)
    model_IF = model.fit(train_set)
    
    #保存模型
    with open(f'scr/storage/cache/IF_model/model_{type}.pkl', 'wb') as file:
       pickle.dump(model_IF, file)
    return


"""
 # Describe :   使用孤立森林模型进行预测(机器预测)
 # Parameter :
        parameter 1:    predict_path 预测数据集路径
        parameter 2:    type 图像类型 
 # Return :     predictions 返回一个预测结果字典
"""  
def model_predict_IF(predict_path, type):
    # 加载模型
    with open(f'scr/storage/best_model/IF_model/model_{type}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    result={}

    # 模型预测
    for filename in os.listdir(predict_path):
        img = os.path.join(predict_path, filename)
        img_features = statistics(img, type)
        img_features = img_features.reshape(1, -1)
        result[filename] = loaded_model.predict(img_features)
    
    return result


"""
 # Describe :   人工设计和训练
 # Parameter :  
 #      parameter1 : training_path 训练集路径
 #      parameter2 : type 图像模型
 # Return :     weight_list 返回训练完的权值
"""  
def model_training_HM(train_path, type):
    """
    # 模型创建: 根据统计特征人工设计一个模型
    # 模型原理: 计算各训练集各参数的的统计特征, 根据模统计特征计算出95%的置信区间,
                再根据训练集各参数在95%的置信区间的个数, 归一化作为权值当作模型
    """  
    
    train_set = batch_processing(train_path, type)
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
    
    # 用训练集训练得出权值
    for train_data in train_set:    
        #训练样本的参数是否在95%置信区间内{'Y': 1,'N': -1}
        for column_index in range(num_columns):
            if (trained_range[column_index][0] <= train_data[column_index] 
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
    np.save(f'scr/storage/cache/HM_model/trained_range_HM_{type}.npy', trained_range)
    np.save(f'scr/storage/cache/HM_model/weight_list_HM_{type}.npy', weight_list)

    
    return weight_list
    

"""
 # Describe :   人工模型预测
 # Parameter :  
 #      parameter1 : predict_path 预测集路径
 #      parameter2 : type 图像模型
 # Return :     result 返回预测结果,类型是一个字典
"""  
def model_predict_HM(predict_path, type):
    # 模型导入
    weight_list = np.load(f'scr/storage/best_model/HM_model/weight_list_HM_{type}.npy')
    trained_range = np.load(f'scr/storage/best_model/HM_model/trained_range_HM_{type}.npy')
    num_columns = len(trained_range)
    # 中间变量
    access = []
    result = {}

    # 判别图像
    for filename in os.listdir(predict_path):
        img = os.path.join(predict_path, filename)
        predicted_value = np.array(statistics(img, type)).reshape(1, -1)
        
        #预测样本的参数是否在95置信区间内{'Y': 1,'N': -1}
        for column_index in range(num_columns):
            if (trained_range[column_index][0] <= predicted_value[0, column_index] 
                <= trained_range[column_index][1]):
                access.append(1)
            else:
                access.append(-1)
        
        # 权值与通过率进行加权乘积并求和
        score_perdict = sum(np.multiply(weight_list, access))

        # 预测结果
        result[filename] = -1 if score_perdict < 0.3  else 1
        
        # 初始化中间变量, 继续进行预测
        access = []
        score_perdict = 0
    return result


"""
 # Describe :   模型评估
 # Parameter :  
 #      parameter1 : data_set_path1 数据集1路径
 #      parameter2 : data_set_path2 数据集2路径
 #      parameter3 : data_set_type1 数据集类型(正确数据集或错误数据集)
 #      parameter4 : data_set_type2 数据集类型(正确数据集或错误数据集)
 #      parameter5 : models 评估模型的类型
 #      parameter6 : type 图像模型的类型
 # Return :     
"""  
def model_evaluate(data_set_path1, data_set_path2, data_set_type1 = 'normal', 
                   data_set_type2 = 'abnormal', models = 'IF', type = 'RGB'): 
    # 初始化混淆矩阵变量
    Matrix = {
        'TP' : 0,
        'TN' : 0,
        'FP' : 0,
        'FN' : 0
    }
    model_dict = {
        'IF': model_predict_IF,
        'HM': model_predict_HM
    }
    model = model_dict.get(models)
    if model:
        result1 = model(data_set_path1, type)
        result2 = model(data_set_path2, type)
        
    # 根据混淆矩阵定义更新函数
    def update_counts(result, standard, Matrix):
        for value in result.values():
            if (standard == 1):
                if (value == standard):
                     Matrix['TP'] += 1  
                else:
                     Matrix['FN'] += 1  
            else:
                if (value == standard):
                    Matrix['TN'] += 1  
                else:
                    Matrix['FP'] += 1  


   
    # 计算混淆矩阵
    standard1 = 1 if data_set_type1 == 'normal' else -1
    update_counts(result1, standard1, Matrix)
    standard2 = 1 if data_set_type2 == 'normal' else -1
    update_counts(result2, standard2, Matrix)
    print(Matrix)
    
    A = (Matrix['TP'] + Matrix['TN']) / (Matrix['TP'] + Matrix['FN'] + Matrix['FP'] +Matrix['TN'])
    P = Matrix['TP'] / (Matrix['TP'] + Matrix['FP'])
    R = Matrix['TP'] / (Matrix['TP'] + Matrix['FN'])
    F = 2 * P * R / (P + R)
    MCC = (((Matrix['TP'] * Matrix['TN']) - (Matrix['FP'] * Matrix['FN'])) / 
           math.sqrt((Matrix['TP'] + Matrix['FP']) * (Matrix['TP'] + Matrix['FN']) * (Matrix['TN'] + Matrix['FP']) * (Matrix['TN'] + Matrix['FN'])))
    print(f'A = {A}, P = {P}, R = {R}, F = {F}, MCC = {MCC}')

    np.save(f'scr/storage/cache/{models}_model/result_{type}.npy', [A, P, R, F, MCC])
    return [A, P, R, F, MCC]

"""
 # Describe :    保存训练效果最好的模型
 # Parameter :  
 #      parameter1 : result 模型评价结果
 #      parameter2 : input_dir 保存模型路径
 #      parameter3 : save_dir 保存模型路径
 #      parameter3 : type 模型类型
 # Return :     
"""  
def best_model_save(result, input_dir, save_dir, type):
    def copy_files(src_dir, dest_dir):
        # 遍历源目录中的所有文件
        for filename in os.listdir(src_dir):
            # 构建源文件和目标文件的完整路径
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename)

            # 检查是否是文件（而不是目录）
            if os.path.isfile(src_file):
                shutil.copy(src_file, dest_file)

    # 如果save文件夹存在
    if os.path.exists(save_dir):
        if len(os.listdir(save_dir)) == 0:
            copy_files(input_dir, save_dir)
            return
    else:
        # 加载模型结果
        result_temp = np.load(f'scr/storage/cache/{type}_model/result_HSV.npy')
        # 检查条件
        if np.any(result > result_temp):
            copy_files(input_dir, save_dir)

    return
