import torch
from torch.utils.data import  DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model import SimpleCNN
from tqdm import tqdm
from utils import img_reader, val, BatchData
import json
import random
import torch.nn as nn
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')

config_path = './config.json'
with open(config_path, 'r') as f_conf:
    conf = json.load(f_conf)

## Config
cat_data_dir = conf['cat_data_dir']
dog_data_dir = conf['dog_data_dir']
num_epochs = conf['num_epochs']
batch_size = conf['batch_size']
lr = conf['lr']
img_size = conf['img_size']
num_train = conf['num_train']
num_val = conf['num_val']
num_test = conf['num_test']
num_print_step = conf['num_print_step']


## Read Data
cat_data_list = img_reader(cat_data_dir, 'cat')
dog_data_list = img_reader(dog_data_dir, 'dog')
data_list = cat_data_list + dog_data_list
random.shuffle(data_list)
train_data_list = data_list[:num_train]
val_data_list = data_list[num_train: num_train+num_val]

train_dataset = BatchData(train_data_list, img_size)
val_dataset = BatchData(val_data_list, img_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Build Model
model = SimpleCNN()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(model)
exit()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1]) # multi-GPU
criteria = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)


# Params
train_acc_list = []
train_loss_list = []
train_step_list = []
val_loss_list = []
val_acc_list = []
val_step_list = []
step = 0
train_acc = 0
loss = 0


## Train
model.train()
for epoch in tqdm(range(num_epochs)):
    # val_loss, val_acc = val(model, val_loader, criteria, num_val)
    for data in train_loader:
        step += 1
        output = model(data)
        label = data['label']
        loss = criteria(output, label)
        # cal acc
        _, prediction = torch.max(output, 1)
        train_correct = (prediction == data['label']).sum()
        train_acc = train_correct.float() / len(data['label'])
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc_list.append(round(float(train_acc), 2))
        train_loss_list.append(round(float(loss.item()), 2))
        train_step_list.append(step)
        if step % num_print_step == 0:
            print('Train: |    Epoch: {}|    Step: {}|    Acc: {:.2f} %|    Loss: {:.4f}'.format(epoch, step, train_acc*100, loss.item()))
    # val
    val_loss, val_acc = val(model, val_loader, criteria, num_val)
    val_loss_list.append(round(float(val_loss), 2))
    val_acc_list.append(round(float(val_acc), 2))
    val_step_list.append(step)
    print('Val: |    Epoch: {}|    Acc = {:.2f} %|    Loss = {:.4f}'.format(epoch, val_acc*100, val_loss))
with open('./results/train_step_list.pkl', 'wb') as f_1:
    pkl.dump(train_step_list, f_1)
with open('./results/train_loss_list.pkl', 'wb') as f_2:
    pkl.dump(train_loss_list, f_2)
with open('./results/val_step_list.pkl', 'wb') as f_3:
    pkl.dump(val_step_list, f_3)
with open('./results/val_loss_list.pkl', 'wb') as f_4:
    pkl.dump(val_loss_list, f_4)
with open('./results/train_acc_list.pkl', 'wb') as f_5:
    pkl.dump(train_acc_list, f_5)
with open('./results/val_acc_list.pkl', 'wb') as f_6:
    pkl.dump(val_acc_list, f_6)

## Test
test_data_list = data_list[-num_test:]
test_dataset = BatchData(test_data_list, img_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loss, test_acc = val(model, test_loader, criteria, num_test)
print('Test: |    Acc: {:.2f} %|    Loss: {:.4f}'. format(test_acc*100, test_loss))



