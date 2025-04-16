import pickle
from torch.utils.data import Dataset
import numpy as np
import torch

f=open('./data/413719000/413719000.pkl','rb')
data=pickle.load(f)
# 筛选出速度（discrete_sog）大于或等于1的记录。
data=data[data['速度']>=1].values
data_train_raw, data_test_raw = [],[]
i=0

# 这儿要加10的目的是为了每一包数据和
while i<1400:
    data_train_raw.append(data[i:i+11])
    # print("len(data[i:i+11]) = ", len(data[i:i+11]))
    # print("--------------------------------\n data[i:i+11] = ", data[i:i+11])
    i+=8
i=1400

while i<2600:
    data_test_raw.append(data[i:i+11])
    i+=8
pass

class ShipTrajData(Dataset):
    def __init__(self, data): # x,y,v,theta
        data = np.array(data)
        self.data = torch.from_numpy(data).to(torch.float32)
    def __getitem__(self, index):
        return self.data[index, :, :]
    def __len__(self):
        return self.data.size(0)
        # return data.size(0)
