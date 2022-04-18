import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import time
import json
from utils import read_data, BatchData, eval, create_dir_not_exist
from model import SimpleCNN
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
import pickle as pkl


torch.manual_seed(1)
time_flag = time.strftime("%Y-%m-%d %H:%M:%S")
logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='log/train_' + str(time_flag) + '_log.log',
                    filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    # a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    # 日志格式
                    )
conf_p = 'config/config.json'
with open(conf_p, 'r') as f:
    conf = json.load(f)
print(conf)


## Config
train_img_path = conf['train_img_path']
train_label_path = conf['train_label_path']
batch_size = conf['batch_size']
num_epoch = conf['num_epoch']
val_num = conf['val_num']
lr = conf['lr']
num_print = conf['num_print']


## Read Data
imgs, labels = read_data(train_img_path, train_label_path)
train_imgs, val_imgs = imgs[:-val_num], imgs[-val_num:]
train_labels, val_labels = labels[:-val_num], labels[-val_num:]

train_dataset = BatchData(train_imgs, train_labels)
val_dataset = BatchData(val_imgs, val_labels)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


## Model
model = SimpleCNN()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(model)
logging.info(model)

criteria = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

logging.info(conf)
logging.info(optimizer)
model_save_dir = 'checkpoints/' + time_flag + '/'
create_dir_not_exist(model_save_dir)
logging.info('Model will be saved into: ' + model_save_dir)
print('Model will be saved into: ' + model_save_dir)


val_acc = 0
train_acc = 0
results = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
## Train
for epoch in tqdm(range(num_epoch)):
    # train
    for step, train_batch in enumerate(train_loader):
        output = model(train_batch)
        label = train_batch['label']
        loss = criteria(output, label)

        _, prediction = torch.max(output, 1)
        train_correct = (prediction == train_batch['label']).sum()
        train_acc = train_correct.float() / len(train_batch['label'])  # divided by batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % num_print == 0:
            train_info = 'Epoch: {} / {}|    Step: {} / {}|    Loss: {}|    Acc: {} % |'.format(epoch, num_epoch, step, len(train_loader), loss, train_acc*100)
            print(train_info)
            logging.info(train_info)

    # val
    val_acc, val_loss = eval(model, criteria, val_loader)
    results['train_acc'].append(train_acc.cpu())
    results['train_loss'].append(loss.item())
    results['val_acc'].append(val_acc)
    results['val_loss'].append(val_loss)

    val_info = 'Epoch: {} / {}|    Eval: |    Acc: {} % |    Loss: {}'.format(
        epoch, num_epoch, float('%.2f' % (val_acc*100)), float('%.2f' % val_loss))
    print(val_info)
    logging.info(val_info)
    model.train()

model_name = 'MNIST_acc_' + str(val_acc)
model_path = model_save_dir + model_name
state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lr': lr}
torch.save(state, model_path)
logging.info('Model has been saved into: ' + model_path)

# save results for plt
results_save_path = './log/results_' + time_flag + '.pkl'
with open(results_save_path, 'wb') as f_results:
    pkl.dump(results, f_results)
