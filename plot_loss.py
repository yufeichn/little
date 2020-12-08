import os
import matplotlib.pyplot as plt

files_path = "/data4t/data/zhangzihan/log.txt"
if not os.path.exists(files_path):
    print("log.txt不存在")
    exit(1)
log_list = []
with open(files_path) as read:
    for line in read.readlines():  # 可切片read.readlines()[1921:]
        if 'loss_box_reg' in line:
            log_list.append(line.strip().split(':'))

print('iter nums: ', len(log_list) * 20)
print("File is: ", files_path)
iter = []
loss = []
loss_box_reg = []
loss_classifier = []
loss_objectness = []
loss_rpn_box_reg = []
lr = []
for log in log_list:
    print(log)
    # iter.append(int(log[7][:-4]))
    # loss.append(float(log[8][1:7]))
    # loss_box_reg.append(float(log[9][1:7]))
    # loss_classifier.append(float(log[10][1:7]))
    # loss_objectness.append(float(log[11][1:7]))
    # loss_rpn_box_reg.append(float(log[12][1:7]))
    # lr.append(float(log[15][1:9]))
    iter.append(int(log[7][:-4]))
    loss.append(float(log[8][1:7]))
    loss_box_reg.append(float(log[9][1:7]))
    loss_classifier.append(float(log[10][1:7]))
    loss_objectness.append(float(log[12][1:7]))
    loss_rpn_box_reg.append(float(log[13][1:7]))
    lr.append(float(log[16][1:9]))

plt.figure()
plt.suptitle('log_path: %s    iter: %d' % (files_path, len(log_list) * 20))
plt.subplot(231)
plt.plot(iter, loss, label='loss')
plt.ylim(0, 0.1)
plt.legend()
plt.subplot(232)
plt.plot(iter, loss_box_reg, label='loss_box_reg')
plt.legend()
plt.subplot(233)
plt.plot(iter, loss_classifier, label='loss_classifier')
plt.legend()
plt.subplot(234)
plt.plot(iter, loss_objectness, label='loss_objectness')
plt.ylim(0, 0.01)
plt.legend()
plt.subplot(235)
plt.plot(iter, loss_rpn_box_reg, label='loss_rpn_box_reg')
plt.legend()
plt.subplot(236)
plt.plot(iter, lr, label='lr')
plt.ylim(0, 0.015)
plt.legend()
plt.show()
