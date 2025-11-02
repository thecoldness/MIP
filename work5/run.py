import nibabel as nib
import numpy as np

# 请确保您的文件路径正确
# 如果您下载的是 MINC 文件，NiBabel 通常可以直接读取
INPUT_FILE = './t1_icbm_normal_1mm_pn0_rf0.mnc.gz' 
THRESHOLD_VALUE = 150 # 灰度值阈值

def extract_isovalue_surface(input_path, threshold):
    """
    对3D影像数据进行阈值分割，提取灰度值大于等于指定阈值的体素，并保存为Mask文件。
    """
    
    try:
        # 1. 读取 3D 影像文件
        img = nib.load(input_path)
        data = img.get_fdata() 
        affine = img.affine # 获取仿射矩阵，保存空间信息
        
        # 2. 阈值分割（创建二值 Mask）
        # 注释：找到所有灰度值大于等于 150 的体素，并将它们的值设为 1。
        mask_data = (data >= threshold).astype(np.uint8) 
        
        # 3. 创建新的 NIfTI 图像用于保存 Mask
        mask_img = nib.Nifti1Image(mask_data, affine)
        
        # 4. 保存 Mask 文件
        output_filename = f'isovalue_{threshold}_mask.nii.gz'
        nib.save(mask_img, output_filename)
        
        print(f"Mask 已成功生成并保存至: {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"数据处理失败，请检查文件路径和格式: {e}")
        return None

# --- 主程序执行 ---
output_mask_file = extract_isovalue_surface(INPUT_FILE, THRESHOLD_VALUE)