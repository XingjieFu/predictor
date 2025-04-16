import argparse
from dataread import data_test_raw, ShipTrajData
import torch
import numpy as np
from transformer.Models import Transformer
from torch.nn import functional as F
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from torch.utils.data import DataLoader

def DrawTrajectory(tra_pred, tra_true):
    tra_pred[:, :, 0] = tra_pred[:, :, 0] * 0.00063212 + 110.12347
    tra_true[:, :, 0] = tra_true[:, :, 0] * 0.00063212 + 110.12347
    tra_pred[:, :, 1] = tra_pred[:, :, 1] * 0.000989464 + 20.023834
    tra_true[:, :, 1] = tra_true[:, :, 1] * 0.000989464 + 20.023834
    
    # 检查轨迹点数量
    if tra_true.shape[0] == 0 or tra_true.shape[1] == 0:
        print("Error: No valid trajectory points to plot.")
        return
    
    idx = random.randrange(0, tra_true.shape[0])
    idx =1
    plt.figure(figsize=(9, 6), dpi=150)
    pred = tra_pred[idx, :, :].cpu().detach().numpy()
    true = tra_true[idx, :, :].cpu().detach().numpy()
    np.savetxt('pred_true.txt', np.vstack((pred, true)))
    print("A track includes a total of {0} detection points, and their longitude and latitude differences are".format(pred.shape[0]))
    for i in range(pred.shape[0]):
        print("{0}: ({1} degrees, {2} degrees)".format(i + 1, abs(pred[i, 0] - true[i, 0]), abs(pred[i, 1] - true[i, 1])))
    print('\n')

    # 获取点的数量
    n_points = pred.shape[0]
    if n_points == 0:
        print("Error: No points in the selected trajectory.")
        return

    # 创建颜色渐变
    pred_cmap = cm.get_cmap('YlOrRd')  # 预测轨迹：浅黄到深红
    true_cmap = cm.get_cmap('YlGnBu')  # 历史轨迹：浅黄到深蓝
    pred_colors = [pred_cmap(i / (n_points - 1)) for i in range(n_points)]  # 渐变平滑
    true_colors = [true_cmap(i / (n_points - 1)) for i in range(n_points)]

    # 逐点绘制散点图，完全不透明
    for i in range(n_points):
        plt.scatter(pred[i, 0], pred[i, 1], c=[pred_colors[i]], marker='o', s=100,
                    label="Predicted" if i == 0 else None)  # 圆点，不透明
        plt.scatter(true[i, 0], true[i, 1], c=[true_colors[i]], marker='*', s=150,
                    label="Historical" if i == 0 else None)  # 五角星，不透明

    # 绘制连接线，完全不透明
    plt.plot(pred[:, 0], pred[:, 1], "r-", linewidth=1)
    plt.plot(true[:, 0], true[:, 1], "b-", linewidth=1)

    # 设置坐标轴格式为普通数字
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

    # 设置图例：右上角，历史为蓝色，预测为红色
    plt.legend(handles=[
        plt.Line2D([0], [0], color='r', marker='o', linestyle='-', label='Predicted'),
        plt.Line2D([0], [0], color='b', marker='*', linestyle='-', label='Historical')
    ], loc='upper right')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Predicted vs Historical Trajectory")
    plt.grid(True, alpha=0.5)  # 添加网格线
    # plt.savefig(f"trajectory_{idx}.png", dpi=150, bbox_inches='tight')  # 保存图形
    plt.show()

# 通过计算预测值和真值之间的误差来作为loss
def cal_performance(tra_pred, tra_true):
    return F.mse_loss(tra_pred, tra_true)

def test(model, dataloader, device):
    total_loss = 0
    for idx, data in enumerate(dataloader):
        tra_pred = model(input_data=data.to(device).to(torch.float32), device=device)
        loss = cal_performance(tra_pred, data[:, 1:, :].to(device).to(torch.float32))
    total_loss += loss.item()
    DrawTrajectory(tra_pred, data[:, 1:, :])
    print("Test Finish, total_loss = {}".format(total_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default=140)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-do_eval', type=bool, default=True)

    opt = parser.parse_args()
    opt.d_word_vec = opt.d_model
    device = "cuda:0"
    # device = "cpu"

    if opt.do_eval:
        data_test = ShipTrajData(data_test_raw)
        test_loader = DataLoader(dataset=data_test, batch_size=opt.batch_size, shuffle=False)
        model = torch.load('model.pt', weights_only=False).to(device)  # 加载模型
        for name, param in model.named_parameters():
            print(f"Parameter {name}: {param[:5]}")
            break  # 只打印第一个参数
        test(
            model=model,
            dataloader=test_loader,
            device=device
        )