import json
import torch
import numpy as np
from model import SimpleCNN
from utils import read_data, BatchData, eval, create_dir_not_exist
from torch.utils.data import DataLoader
# from tqdm import tqdm
import pickle as pkl
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt




conf_p = 'config/config.json'
with open(conf_p, 'r') as f:
    conf = json.load(f)

test_img_path = conf['test_img_path']
test_label_path = conf['test_label_path']
checkpoint_path = conf['checkpoint_path']
batch_size = conf['batch_size']
time_flag = checkpoint_path.split('/')[-2]
results_save_path = './log/results_' + time_flag + '.pkl'


## Read Data
imgs, labels = read_data(test_img_path, test_label_path)
test_dataset = BatchData(imgs, labels)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

## Load Model
print('Load model...')
model = SimpleCNN()
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model, device_ids=[0, 1]) # multi-GPU
model_param = torch.load(checkpoint_path)
model.load_state_dict(model_param['model'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
criteria = CrossEntropyLoss()
test_acc, test_loss = eval(model, criteria, test_loader)
print('Test: Acc: {} %|    Loss: {}|'.format(float('%.2f' % float(test_acc*100)), float('%.2f' % float(test_loss))))

# PLT
with open(results_save_path, 'rb') as f_results:
    results = pkl.load(f_results)

plt.figure()
plt.plot(np.array(results['train_acc']))
plt.plot(np.array(results['val_acc']))
plt.plot(np.array(results['train_loss']))
plt.plot(np.array(results['val_loss']))
plt.title('Changes of acc and loss scores through training.')
plt.xlabel('Epochs')
plt.ylabel('Values of acc and loss')
plt.legend(['Train Acc', 'Val Acc', 'Train Loss', 'Val Loss'])
plt.savefig('./log/plt_'+time_flag+'.png')


