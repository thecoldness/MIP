import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_histogram(image_array):
    """
    计算图像的灰度直方图
    
    参数:
        image_array: 输入图像数组
    
    返回:
        hist: 直方图数组（256个元素）
    """
    # 如果图像是彩色的，先转换为灰度图
    if len(image_array.shape) == 3:
        # 使用加权平均法转换为灰度图：Gray = 0.299*R + 0.587*G + 0.114*B
        gray = np.dot(image_array[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = image_array
    
    # 计算直方图：统计每个灰度级别(0-255)的像素数量
    hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    return hist

def calculate_cumulative_histogram(hist):
    """
    计算灰度累积直方图（累积分布函数CDF）
    
    参数:
        hist: 灰度直方图
    
    返回:
        cum_hist: 累积直方图
    """
    # 对直方图进行累积求和
    cum_hist = np.cumsum(hist)
    return cum_hist

def histogram_equalization(image_array):
    """
    直方图均衡化处理
    
    算法原理：
    1. 计算原始图像的灰度直方图 H(r)
    2. 计算累积分布函数 CDF(r) = Σ(i=0 to r) H(i)
    3. 将CDF归一化到[0, 255]范围: s(r) = round(255 * CDF(r) / 总像素数)
    4. 使用归一化的CDF作为映射函数，将原图的每个像素值映射到新值
    
    参数:
        image_array: 输入图像数组
    
    返回:
        equalized: 均衡化后的图像
    """
    # 如果图像是彩色的，先转换为灰度图
    if len(image_array.shape) == 3:
        gray = np.dot(image_array[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = image_array
    
    gray_uint8 = gray.astype(np.uint8)
    
    # 步骤1: 计算原始图像的灰度直方图
    hist = calculate_histogram(gray_uint8)
    
    # 步骤2: 计算累积分布函数CDF
    cdf = calculate_cumulative_histogram(hist)
    
    # 步骤3: 归一化CDF到[0, 255]范围
    # 避免除以0的情况
    cdf_normalized = np.where(cdf > 0, 
                              np.round(255 * cdf / cdf[-1]), 
                              0).astype(np.uint8)
    
    # 步骤4: 使用映射函数对图像进行变换
    equalized = cdf_normalized[gray_uint8.flatten()]
    equalized = equalized.reshape(gray_uint8.shape)
    
    return equalized

def plot_histograms(original, equalized, save_prefix="result"):
    """
    绘制原始图像和均衡化后图像的直方图和累积直方图
    
    参数:
        original: 原始图像
        equalized: 均衡化后的图像
        save_prefix: 保存文件的前缀
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体或其他中文支持的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    
    # 计算原始图像的直方图和累积直方图
    hist_original = calculate_histogram(original)
    cum_hist_original = calculate_cumulative_histogram(hist_original)
    
    # 计算均衡化后图像的直方图和累积直方图
    hist_equalized = calculate_histogram(equalized)
    cum_hist_equalized = calculate_cumulative_histogram(hist_equalized)
    
    # 创建子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 原始图像的灰度统计直方图
    axes[0, 0].bar(range(256), hist_original, color='blue', width=1.0)
    axes[0, 0].set_title('原始图像灰度统计直方图', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('灰度值', fontsize=12)
    axes[0, 0].set_ylabel('像素数量', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 原始图像的灰度累积直方图
    axes[0, 1].plot(range(256), cum_hist_original, color='red', linewidth=2)
    axes[0, 1].set_title('原始图像灰度累积直方图', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('灰度值', fontsize=12)
    axes[0, 1].set_ylabel('累积像素数量', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 均衡化后图像的灰度统计直方图
    axes[1, 0].bar(range(256), hist_equalized, color='green', width=1.0)
    axes[1, 0].set_title('均衡化后灰度统计直方图', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('灰度值', fontsize=12)
    axes[1, 0].set_ylabel('像素数量', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 均衡化后图像的灰度累积直方图
    axes[1, 1].plot(range(256), cum_hist_equalized, color='purple', linewidth=2)
    axes[1, 1].set_title('均衡化后灰度累积直方图', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('灰度值', fontsize=12)
    axes[1, 1].set_ylabel('累积像素数量', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_histograms.png', dpi=300, bbox_inches='tight')
    print(f"直方图已保存为: {save_prefix}_histograms.png")
    plt.show()

def main():
    # 读取原始图像
    input_path = "img.png"
    print(f"正在读取图像: {input_path}")
    img = Image.open(input_path)
    img_array = np.array(img)
    
    print(f"原始图像尺寸: {img_array.shape}")
    
    # 任务1: 画出影像的灰度统计直方图
    print("\n任务1: 计算灰度统计直方图...")
    hist = calculate_histogram(img_array)
    print("灰度直方图计算完成")
    
    # 任务2: 画出影像的灰度累积直方图
    print("\n任务2: 计算灰度累积直方图...")
    cum_hist = calculate_cumulative_histogram(hist)
    print("灰度累积直方图计算完成")
    
    # 任务3: 直方图均衡化处理
    print("\n任务3: 执行直方图均衡化处理...")
    equalized = histogram_equalization(img_array)
    
    # 保存均衡化后的图像
    equalized_img = Image.fromarray(equalized)
    equalized_img.save("result_equalized.png")
    print("均衡化后的图像已保存为: result_equalized.png")
    
    # 绘制并保存所有直方图
    print("\n正在绘制直方图...")
    plot_histograms(img_array, equalized, "result")
    
    print("\n所有任务完成！")
    print("\n生成的文件：")
    print("  - result_equalized.png: 均衡化后的图像")
    print("  - result_histograms.png: 包含4个子图的对比直方图")

if __name__ == "__main__":
    main()

