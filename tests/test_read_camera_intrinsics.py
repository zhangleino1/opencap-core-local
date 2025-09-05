import pickle
import numpy as np
import os

def read_camera_intrinsics(file_path):
    """
    读取相机内参pickle文件并输出内容
    
    Args:
        file_path (str): pickle文件路径
    
    Returns:
        dict: 相机内参数据
    """
    with open(file_path, 'rb') as f:
        camera_intrinsics = pickle.load(f)
    
    print("相机内参数据内容：")
    print("=" * 50)
    
    for key, value in camera_intrinsics.items():
        print(f"\n{key}:")
        if isinstance(value, np.ndarray):
            print(f"  形状: {value.shape}")
            print(f"  数据类型: {value.dtype}")
            print(f"  值:")
            print(value)
        else:
            print(f"  值: {value}")
    
    return camera_intrinsics

def test_read_iphone_intrinsics():
    """测试读取iPhone17,5相机内参"""
    file_path = r"CameraIntrinsics\iPhone17,5\Deployed_720_60fps\cameraIntrinsics.pickle"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    return read_camera_intrinsics(file_path)

if __name__ == "__main__":
    # 切换到项目根目录
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # 读取iPhone17,5相机内参
    intrinsics = test_read_iphone_intrinsics()

    # distortion (畸变系数)
    #   intrinsicMat (内参矩阵):