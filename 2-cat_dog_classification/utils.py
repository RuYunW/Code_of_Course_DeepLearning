import os
import torch
from torch.utils.data import  Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


def img_reader(path, label):
    data_list = []
    img_dir_list = os.listdir(path)
    for img_name in img_dir_list:
        img_path = path + img_name
        # label = img_name.split('.')[0]
        data_list.append({'img_path': img_path, 'label': label})
    return data_list



class BatchData(Dataset):
    def __init__(self, all_data, img_size=112):
        self.all_data = all_data
        self.img_size = img_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __len__(self):
        return len(self.all_data)
    def __getitem__(self, idx):
        data = self.all_data[idx]
        img_path = data['img_path']
        label = torch.tensor(0) if data['label'] == 'cat' else torch.tensor(1)  # cat = 1  dog = 0
        torch_img = self.transfer_img(img_path)
        img_info = {'tens': torch_img.to(self.device), 'label': label.to(self.device)}
        return img_info

    def transfer_img(self, img_path):
        mode = Image.open(img_path).convert('RGB')
        input_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        torch_img = input_transform(mode)
        mode.close()
        return torch_img


def val(model, val_loader, criteria, data_num):
    model.eval()
    correct_num = []
    with torch.no_grad():
        loss_all = []
        acc_all = []
        for data in val_loader:
            output = model(data)
            loss = criteria(output, data['label'])
            _, prediction = torch.max(output, 1)
            train_correct = (prediction == data['label']).sum()
            # train_acc = train_correct.float() / len(data['label'])
            correct_num.append(train_correct.float().cpu())
            # print(correct_num)
            # exit()
            loss_all.append(loss)
            # acc_all.append(train_acc)
        loss = sum(loss_all) / len(loss_all)
        # acc = sum(acc_all) / len(acc_all)
    acc = np.array(correct_num).sum() / data_num
    model.train()
    return loss, acc


