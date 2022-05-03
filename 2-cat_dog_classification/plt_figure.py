import matplotlib.pyplot as plt
import pickle as pkl


with open('./results/train_step_list.pkl', 'rb') as f_1:
    train_step_list = pkl.load(f_1)
with open('./results/train_loss_list.pkl', 'rb') as f_2:
    train_loss_list = pkl.load(f_2)
with open('./results/val_step_list.pkl', 'rb') as f_3:
    val_step_list = pkl.load(f_3)
with open('./results/val_loss_list.pkl', 'rb') as f_4:
    val_loss_list = pkl.load(f_4)
with open('./results/train_acc_list.pkl', 'rb') as f_5:
    train_acc_list = pkl.load(f_5)
with open('./results/val_acc_list.pkl', 'rb') as f_6:
    val_acc_list = pkl.load(f_6)

## Plt Figure
# fig = plt.figure()
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# ax1 = fig.add_subplot(111)
plt.plot(train_step_list, train_loss_list, 'r')
plt.plot(val_step_list, val_loss_list, 'b')
plt.ylabel('loss', fontsize=12)
plt.legend(['train set', 'val set'], loc='upper left')
plt.title('The change of Loss value during training.')
plt.xlabel('Steps')
plt.savefig('./figs/results_loss.png')
plt.close()
# ax2 = ax1.twinx()
# fig = plt.figure()
# ax2 = fig.add_subplot(111)
plt.plot(train_step_list, train_acc_list, 'r')
plt.plot(val_step_list, val_acc_list, 'b')
plt.ylabel('Acc.', fontsize=12)
plt.legend(['train set', 'val set'], loc='upper right')
plt.title('The change of Acc. value during training.')
plt.xlabel('Steps')
plt.savefig('./figs/results_acc.png')