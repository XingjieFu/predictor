# 项目操作指南

## 配置环境

在开始之前，请确保已安装项目所需的依赖项。运行以下命令以安装依赖：

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py
tensorboard --logdir="./logs" #使用 TensorBoard 查看日志
```

## 评估

```bash
python eval.py
