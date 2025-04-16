import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_excel_coordinates(file_path, lon_col='Longitude', lat_col='Latitude', sample_interval=5):
    """
    读取Excel文件中的经度和纬度数据并绘制散点图，颜色按顺序渐变，减少点数。
    
    参数:
        file_path (str): Excel文件路径
        lon_col (str): 经度列名，默认为'Longitude'
        lat_col (str): 纬度列名，默认为'Latitude'
        sample_interval (int): 采样间隔，默认为5（每5个点取1个）
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 获取经度和纬度数据
    longitude = df[lon_col]
    latitude = df[lat_col]
    
    # 采样数据以减少点数
    longitude = longitude[::sample_interval]
    latitude = latitude[::sample_interval]
    
    # 创建颜色渐变
    num_points = len(longitude)
    colors = plt.cm.Blues(np.linspace(0.2, 1, num_points))  # 从浅到深的蓝色
    
    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(longitude, latitude, c=colors, alpha=0.5)
    plt.title('Longitude vs Latitude Scatter Plot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    
    # 显示图形
    plt.show()

# 示例调用
if __name__ == "__main__":
    plot_excel_coordinates('413719000.xls', '经度', '纬度', sample_interval=5)