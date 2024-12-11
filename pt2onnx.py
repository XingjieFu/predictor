import torch.onnx
import netron
import torch
import torch.nn as nn
# 包装类
class WrapperModel(nn.Module):
    def __init__(self, model, device):
        super(WrapperModel, self).__init__()
        self.model = model
        self.device = device

    def forward(self, input_data):
        # 调用实际模型前，传递 device 参数
        return self.model(input_data, self.device)
    
model=torch.load('model.pt').to("cpu")
# 指定设备
device = torch.device('cpu')
# 包装模型
wrapped_model = WrapperModel(model, device)
# 创建输入数据
d = torch.randn(140, 11, 4).to(device)

torch.onnx.export(wrapped_model,d,'model.onnx',opset_version=10)
netron.start('model.onnx')