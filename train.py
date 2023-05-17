# 模板序号008
# 开发时间 2022/11/19 18:39
import sys
[sys.path.append(i) for i in ['.', '..']]

import argparse
import os
from datetime import time
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#import conf.global_settings as settings
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import itertools
CUDA_LAUNCH_BLOCKING=1
import os,shutil
from PIL import Image
from torchvision.transforms import transforms
import random
# transform=transforms.Compose([transforms.Resize((160,192)),transforms.RandomRotation(15),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],
#                                                                          std=[0.29,0.224,0.225])])
# train_db=ImageFolder(root=r'E:\pycharm_project_pytorch\Face_emotion\genki4k\face\train',transform=transform)
# val_db=ImageFolder(root=r'E:\pycharm_project_pytorch\Face_emotion\genki4k\face\test',transform=transform)

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_data_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        n_iter = (epoch - 1) * len(training_data_loader) + batch_index + 1
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        if batch_index==390:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(training_data_loader.dataset)))
        writer.add_scalar('Train/loss', loss.item(), n_iter)       # update training loss
        if epoch <= args.warm:
            warmup_scheduler.step()    #  start warmup

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    finish = time.time()               # runing time
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()
    test_loss = 0.0
    correct = 0.0
    for (images, labels) in test_data_loader:
        #print('+++:',labels,type(labels))
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_data_loader.dataset),
        correct.float() / len(test_data_loader.dataset),
        finish - start))
    print()

    if tb:                       # Write information to tensorboard
        writer.add_scalar('Test/Average loss', test_loss / len(test_data_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_data_loader.dataset), epoch)
    ########################################################################################################
    return correct.float() / len(test_data_loader.dataset), test_loss / len(test_data_loader.dataset)     # test Acc
    #########################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='ECA_ResNet_50', help='net type')      # Selection model
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')   # Whether to use CUDA. If no, set this parameter to false
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')     # batch_size
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')    # learning rate
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()
    net = get_network(args)
    # data preprocessing:
    training_data_loader = get_training_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=0,
        batch_size=args.b,
        shuffle=True)

    test_data_loader = get_test_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=0,
        batch_size=args.b,
        shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_data_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')
        checkpoints_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        checkpoints_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    checkpoints_path = os.path.join(checkpoints_path, '{net}-{epoch}-{type}.pth')
    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            #########################################################################################################
            best_acc = eval_training(tb=False)[0]  # Validation function
            #########################################################################################################
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))
        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    ###############################################################
    val_acc=[]
    val_loss=[]
    ##############################################################
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)
        ###########################################################
        val_acc.append(acc[0].cpu())
        print('acc[0]:',type(acc[0]))
        print(acc[0])
        print('acc[1]:', type(acc[1]))
        print(acc[1])
        val_loss.append(acc[1])
        ###########################################################
        if epoch > settings.MILESTONES[1] and best_acc < acc[0]:
            weights_path = checkpoints_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc[0]
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoints_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
history = {'val_loss': val_loss, 'val_acc': val_acc}
#history = train_and_val(epoch, net, train_loader, val_loader, loss_function, optimizer)
# Print accuracy and loss curve
def plot_loss(x, history):
    plt.plot(x, history['val_loss'], label='val', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(x, history):
    plt.plot(x, history['val_acc'], label='val_acc', marker='x')
    plt.title('Score per epoch')
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

plot_loss(np.arange(0,epoch), history)
plot_acc(np.arange(0,epoch), history)

# Check the accuracy of each category #
#model= torch.load('./best.pth', map_location=torch.device('cuda:0'))   # load model optional
model=net
classes = ('bladder','bowel','gallbladder','kidney','liver','spleen','muscule','benign', 'maligant','normal')     # Category name, change as needed
#classes = ('bladder','bowel','gallbladder','kidney','liver','spleen')
class_correct = [0.] * len(classes)                         # Multiply by the number of categories
class_total = [0.] * len(classes)                           # Multiply by the number of categories
y_test, y_pred = [], []
X_test = []
y_score1=[]

with torch.no_grad():
    for images, labels in  test_data_loader:    # The verification set retrieves the picture and label
        X_test.extend([_ for _ in images])  # Insert a list at the end of the list, and the list is the picture data
        outputs = model(images.cuda())   # Predictive output
        #_, predicted = torch.max(outputs, 1)  # Remove the predicted maximum value, that is, remove the label
        score, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu()         # Taking the processing equipment from another device, such as a gpu, to a cpu will not change the type of variable -- it will remain Tensor
        # print('predicted：',predicted)
        # print('labels:',labels)
        # print(predicted==labels)
        c = (predicted == labels).squeeze()  # Get a list of True and False
        for i, label in enumerate(labels):   # Traversal tag
            class_correct[label] += c[i].item()   # class_correct[label]为10个类别列表中的第label个位置，c[i]遍历判断后的True和False的列表元素，True为1，False为0，class_correct[label] += c[i].item()为计算每个类别预测准确的数量
            class_total[label] += 1               # Calculate the total number of labels for each category
        y_pred.extend(predicted.numpy())          # Add the prediction tag to the list
        y_test.extend(labels.cpu().numpy())       # Add the GT tag to the list
        y_score1.extend(score.cpu().numpy())
for i in range(len(classes)): ##########################################################################################################################改
    print(f"Acuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:2.0f}%")   #  class_correct[i] / class_total[i]Calculate the accuracy of each classification

# 查看precision，recall和f1-score#################################################################################################################################################
from sklearn.metrics import confusion_matrix, classification_report
lable1= range(len(classes))  #############################################################################################################################改

ac = accuracy_score(y_test, y_pred)     # accuracy_score It has been imported above

cmtx = confusion_matrix(y_test, y_pred)    # The confusion matrix is calculated
cr = classification_report(y_test, y_pred,labels=lable1,target_names=classes,digits=4)
print("Accuracy is :",ac)
print(cr)

pat=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)
f=open(pat+'/confusion.txt','w')
f.writelines(str(history['val_loss'])+'\n')
f.writelines(str(history['val_acc'])+'\n')
f.writelines(str(classes)+'\n')
f.writelines(str(cmtx )+'\n')
f.writelines(str(y_test)+'\n')
f.writelines(str(y_pred)+'\n')
f.writelines(str(cr))
f.close()

# View confusion matrix#####################################################################################################################################################################
import seaborn as sns, pandas as pd

labels = pd.DataFrame(cmtx).applymap(lambda v: f"{v}" if v!=0 else f"")   # pd.DataFrame是由多种类型的列构成的二维标签数据结构
# plt.figure(figsize=(7,5))
# plt.xticks(rotation=45)           # x label 45 degree display
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
sns.heatmap(cmtx, annot=labels, fmt='s', xticklabels=classes, yticklabels=classes, linewidths=0.1 )  # 绘制热力图
title='Confusion matrix'
cmap=plt.cm.Blues
# plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)    # x label 45 degree display
plt.yticks(tick_marks, classes,rotation=45)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

###############################################################################
# Draw the confusion matrix
def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure
# Add confusion matrix to tensorboard
def add_confusion_matrix(
    writer,
    cmtx,
    num_classes,
    global_step=None,
    subset_ids=None,
    class_names=None,
    tag="Confusion Matrix",
    figsize=None,
):
    """
    Calculate and plot confusion matrix to a SummaryWriter.
    Args:
        writer (SummaryWriter): the SummaryWriter to write the matrix to.
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        global_step (Optional[int]): current step.
        subset_ids (list of ints): a list of label indices to keep.
        class_names (list of strs, optional): a list of all class names.
        tag (str or list of strs): name(s) of the confusion matrix image.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    """
subset_ids=None
num_classes=len(classes)
class_names=list(classes)
tag="Confusion Matrix"
figsize=None
global_step=None


sub_cmtx = plot_confusion_matrix(
    cmtx,
    num_classes=len(classes),
    class_names=class_names,
    figsize=figsize,
)
print('begining...')
from matplotlib import image
plt.figure(figsize=(8,10))
plt.yticks(tick_marks, classes,rotation=45)
plt.show()
    # Add the confusion matrix image to writer.
writer.add_figure(tag="Confusion Matrix", figure=sub_cmtx, global_step=global_step)






