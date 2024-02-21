import torch

# 定义PreLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class PreLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label, transforms=None):
        self.data = data_root
        self.label = data_label
        self.transforms = transforms
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.transforms is not None:
            data = self.transforms(data)   # 在这里做transform，转为tensor等等
        return data, label
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
