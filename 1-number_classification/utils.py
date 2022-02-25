import torch
from torch.utils.data import Dataset
import struct
import numpy as np
import os


def read_data(img_path, label_path):
    with open(label_path, 'rb') as flabel:
        magic, n = struct.unpack('>II', flabel.read(8))
        labels = np.fromfile(flabel, dtype=np.uint8)
    with open(img_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack('>IIII', fimg.read(16))
        images = np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

class BatchData(Dataset):
    def __init__(self, img_list, label_list):
        self.img_list = img_list
        self.label_list = label_list
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, item):
        img = self.img_list[item].reshape([1, 28, 28])
        img = torch.tensor(img, dtype=torch.float).to(self.device)
        label = self.label_list[item]
        label = torch.tensor(label, dtype=torch.long).to(self.device)
        return {'img': img, 'label': label}



def eval(model, val_loader):
    model.eval()
    eval_acc_list = []
    for eval_batch in val_loader:
        output = model(eval_batch)

        _, prediction = torch.max(output, 1)
        correct = (prediction == eval_batch['label']).sum()
        eval_acc = correct.float() / len(eval_batch['label'])
        eval_acc_list.append(eval_acc.cpu())
    eval_acc = np.array(eval_acc_list).mean()

    return eval_acc


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


