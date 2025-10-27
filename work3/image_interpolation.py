import numpy as np
from PIL import Image
import os

def linear_interpolation_1d(arr, scale_factor):
    """
    一维线性插值
    
    参数:
        arr: 输入数组
        scale_factor: 缩放因子（例如2表示放大2倍）
    
    返回:
        插值后的数组
    """
    # 获取原始长度
    n = len(arr)
    # 计算新长度
    new_n = int(n * scale_factor)
    # 创建新数组
    new_arr = np.zeros(new_n)
    
    # 计算原始数组的索引对应的新索引
    for i in range(new_n):
        # 计算对应的原始索引位置（浮点数）
        pos = i / scale_factor
        # 计算整数部分和小数部分
        index = int(pos)
        frac = pos - index
        
        # 处理边界情况
        if index >= n - 1:
            new_arr[i] = arr[n - 1]
        else:
            # 线性插值：I = I0 * (1-t) + I1 * t
            new_arr[i] = arr[index] * (1 - frac) + arr[index + 1] * frac
    
    return new_arr

def resize_height_linear(image_array):
    """
    使用线性插值将影像像素行数提升1倍
    
    参数:
        image_array: 输入图像数组 [height, width, channels]
    
    返回:
        插值后的图像数组
    """
    height, width, channels = image_array.shape
    
    # 创建新数组，行数为原来的2倍
    new_height = height * 2
    resized = np.zeros((new_height, width, channels), dtype=image_array.dtype)
    
    # 对每一列进行线性插值
    for c in range(channels):
        for w in range(width):
            # 提取该列的所有像素值
            column = image_array[:, w, c]
            # 对该列进行线性插值
            new_column = linear_interpolation_1d(column, 2.0)
            resized[:, w, c] = new_column
    
    return resized

def resize_width_linear(image_array):
    """
    使用线性插值将影像像素列数提升1倍
    
    参数:
        image_array: 输入图像数组 [height, width, channels]
    
    返回:
        插值后的图像数组
    """
    height, width, channels = image_array.shape
    
    # 创建新数组，列数为原来的2倍
    new_width = width * 2
    resized = np.zeros((height, new_width, channels), dtype=image_array.dtype)
    
    # 对每一行进行线性插值
    for c in range(channels):
        for h in range(height):
            # 提取该行的所有像素值
            row = image_array[h, :, c]
            # 对该行进行线性插值
            new_row = linear_interpolation_1d(row, 2.0)
            resized[h, :, c] = new_row
    
    return resized

def bilinear_interpolation(image_array):
    """
    使用双线性插值将影像像素行数与列数均提升1倍
    
    参数:
        image_array: 输入图像数组 [height, width, channels]
    
    返回:
        插值后的图像数组
    """
    height, width, channels = image_array.shape
    
    # 创建新数组，尺寸为原来的2倍
    new_height = height * 2
    new_width = width * 2
    resized = np.zeros((new_height, new_width, channels), dtype=image_array.dtype)
    
    # 对每个新像素位置进行双线性插值
    for c in range(channels):
        for new_h in range(new_height):
            for new_w in range(new_width):
                # 计算在原图中的对应位置（浮点数坐标）
                pos_h = new_h / 2.0
                pos_w = new_w / 2.0
                
                # 获取四个最近的像素点
                h1 = int(np.floor(pos_h))
                w1 = int(np.floor(pos_w))
                h2 = min(h1 + 1, height - 1)
                w2 = min(w1 + 1, width - 1)
                
                # 计算插值权重
                dh = pos_h - h1
                dw = pos_w - w1
                
                # 双线性插值公式
                # I = I11*(1-dh)*(1-dw) + I12*(1-dh)*dw + I21*dh*(1-dw) + I22*dh*dw
                resized[new_h, new_w, c] = (
                    image_array[h1, w1, c] * (1 - dh) * (1 - dw) +
                    image_array[h1, w2, c] * (1 - dh) * dw +
                    image_array[h2, w1, c] * dh * (1 - dw) +
                    image_array[h2, w2, c] * dh * dw
                )
    
    return resized

def main():
    # 读取原始图像
    input_path = "img.png"
    print(f"正在读取图像: {input_path}")
    img = Image.open(input_path)
    img_array = np.array(img)
    
    # 获取原始图像信息
    original_height, original_width = img_array.shape[:2]
    print(f"原始图像尺寸: {original_width} x {original_height}")
    
    # 任务1: 将影像像素行数提升1倍
    print("\n任务1: 使用线性插值将行数提升1倍...")
    result1 = resize_height_linear(img_array)
    result1_img = Image.fromarray(result1.astype(np.uint8))
    result1_img.save("result_height_doubled.png")
    print(f"结果图像尺寸: {result1.shape[1]} x {result1.shape[0]}")
    print("已保存为: result_height_doubled.png")
    
    # 任务2: 将影像像素列数提升1倍
    print("\n任务2: 使用线性插值将列数提升1倍...")
    result2 = resize_width_linear(img_array)
    result2_img = Image.fromarray(result2.astype(np.uint8))
    result2_img.save("result_width_doubled.png")
    print(f"结果图像尺寸: {result2.shape[1]} x {result2.shape[0]}")
    print("已保存为: result_width_doubled.png")
    
    # 任务3: 采用双线性插值，将影像像素行数与列数均提升1倍
    print("\n任务3: 使用双线性插值将行数和列数均提升1倍...")
    result3 = bilinear_interpolation(img_array)
    result3_img = Image.fromarray(result3.astype(np.uint8))
    result3_img.save("result_bilinear.png")
    print(f"结果图像尺寸: {result3.shape[1]} x {result3.shape[0]}")
    print("已保存为: result_bilinear.png")
    
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()

