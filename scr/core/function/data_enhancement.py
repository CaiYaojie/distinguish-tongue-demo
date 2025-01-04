import numpy as np
from PIL import Image, ImageEnhance
import random
import os

"""
 # Describe :   单张图片数据增强
 # Parameter :  
 #      parameter1 : image 图像对象(PIL图像)
 # Return :     image 返回经过增强处理后的图像对象 (PIL图像)
"""  
def augment_image(image):
    # 增加高斯噪声                    
    def add_gaussian_noise(image, mean=0, var=0.01):
        sigma = var ** 0.5
        width, height = image.size

        # 生成高斯噪声，形状为 (高度, 宽度, 通道数)
        gauss = np.random.normal(mean, sigma, (height, width, 3))

        # 将图像转换为 NumPy 数组并添加噪声
        noisy = np.clip(np.array(image) + gauss, 0, 255)
        return Image.fromarray(noisy.astype('uint8'))

    # 旋转图片
    def rotate_image(image, angle):
        return image.rotate(angle)

    # 翻转图片
    def flip_image(image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    # 调整色度
    def adjust_hue(image, factor):
        return ImageEnhance.Color(image).enhance(factor)

    # 调整饱和度
    def adjust_saturation(image, factor):
        return ImageEnhance.Color(image).enhance(factor)

    # 调整亮度
    def adjust_brightness(image, factor):
        return ImageEnhance.Brightness(image).enhance(factor)

    if random.random() < 0.3:  # 添加高斯噪声
        image = add_gaussian_noise(image)

    if random.random() < 0.4:  # 逆时针旋转
        angle = random.uniform(-5, 5)
        image = rotate_image(image, angle)

    if random.random() < 0.5:  # 左右翻转
        image = flip_image(image)

    if random.random() < 0.1:  # 调整色度
        hue_factor = random.uniform(0.8, 1.3)
        image = adjust_hue(image, hue_factor)

    if random.random() < 0.1:  # 调整饱和度
        saturation_factor = random.uniform(0.7, 1.2)
        image = adjust_saturation(image, saturation_factor)

    if random.random() < 0.5:  # 调整亮度
        brightness_factor = random.uniform(0.8, 1.2)
        image = adjust_brightness(image, brightness_factor)

    return image

"""
 # Describe :   多张图片数据增强
 # Parameter :  
 #      parameter1 : input_folder 输入图像路径
 #      parameter2 : output_folder 输出图像路径
 #      parameter3 : num_outputs 单张图片增强次数
 # Return :    
"""  
def augment_images(input_folder, output_folder, num_outputs):
    # 创建目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历'input_folder'对图像进行数据增强
    image_files = [f for f in os.listdir(input_folder)]
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)

        for i in range(num_outputs):
            augmented_image = augment_image(image)
            output_path = os.path.join(output_folder, f'augmented_{i}_{image_file}')
            augmented_image.save(output_path)
    return