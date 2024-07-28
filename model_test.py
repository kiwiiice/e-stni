import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda import amp
import time
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,auc,f1_score
from dir_utils import model_perfromance
from ctni.dataset_st import CustomDataset
from ctni.efficientnet_feature_st import e_stni


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('.-csv dir', type=str, default='ctn/ctni/st is 0.csv', help='Directory for data dir')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num classes', type=int, default=1, help='Num of diagnostic classes')
    parser.add_argument('--lr','.learning-rate',type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--test batch size",type=int, default=128, help='Test Batch size')
    parser.add_argument('--num workers', type=int, default=0, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phaseu train or test')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--resume', default=False, help='Resume')
    parser.add_argument('--use gpu', default=True,help='Use GPU')
    parser.add_argument('--model path', type=str, default='Save Model', help='Path to saved model')
    parser.add_argument("--log interval", type=str, default=30, help='log interval')

    return parser.parse_args()


def negBpow(data, name, label, seqflag=False):
    dc = len(data)
    if seqflag:
        poscount = len(data[data == 1])
        negcount = len(data[data == 0])
    else:
        poscount = len(data[data[label] == 1])
        negcount = len(data[data[label] == 0])
    pos_rate, neg_rate = poscount / dc, negcount / dc
    print(' {}:正样本数:{}，负样本数:{}，正负祥本比:{}:{}'.\
          format(name, poscount, negcount,1,np.around(negcount/poscount,decimals=4)))
    return poscount, negcount


def main():
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
        device_ids = [i for i in range(torch.cuda.device_count())]

        torch.cuda.set_device(0)
        print("===> using gpu {}".format(device_ids))
    else:
        device = torch.device("cpu")
        print("===> using cpu!!!")


    ##### tni dataset #####
    data = pd.read_csv(args.csv_dir)
    _,_  = negBpow(data, 'origin_data','ctni_class')

    data_X = data.drop('ctni_class', axis=1)
    data_Y = data['ctni_class']

    train_data,test_data,y_train,y_test = train_test_split(data_X,data_Y,test_size=0.3,random_state=123,stratify=data_Y)
    train_data, val_data, y_train, y_val = train_test_split(train_data, y_train, test_size=0.15, random_state=123,
                                                              stratify=y_train)

    transform = transforms.

    train_dataset = CustomDataset(train_data,y_train,transform)
    validation_dataset =CustomDataset(val_data,y_val,transform)
    test_dataset = CustomDataset(test_data, y_val, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,
                              num_workers=args.num_workers,pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size,shuffle=True,
                              num_workers=args.num_workers,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=True,
                              num_workers=args.num_workers,pin_memory=True)

    print("===> loading datasets done")

    #### model ####
    PATH = ''
    best_model = torch.load(PATH)
    model = efficientnet.to(device)
    # print(best_model['best_epoch'])
    model.load_state_dict(best_model['state_dict'])
    youden_index = best_model['Youden Index']


    #### optim ####
    new_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=new_lr,betas=(0.9,0.999),eps=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

    pos_weight = torch.tensor([1 / 1],dtype=torch.float32)
    criterion = torch.nn.BCELoss(weight=torch.FloatTensor(pos_weight).to(device))
    print("===> loading model done")

    grad_scaler = amp.GradScaler()
    start_epoch = 1
    since = time.time()
    results_dict = {'epoch': [],
                    'train_loss': [],
                    'val_loss': [],
                    'lr': [],
                    # 'val_loss':[],
                    # 'val_accuracy':[]
                    "micro_f1_train_subs": [],
                    "micro_f1_val_subs": [],
                    "macro_f1_train_cls": [],
                    "macro_f1_val_cls": [],

                    "val_micro_roc": [],
                    "val_macro_roc": [],
                    "val_micro_ap": [],
                    "val_macro_ap": [],
                    }
    torch.autograd.set_detect_anomaly(True)
    epoch_start_time = time.time()
    true_labels,target_labels, prob_score, pred_labels = np.array([]),np.array([]),np.array([]),np.array([])

    #### test ####
    model.eval()

    test_loss = 0
    corret = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for batch_idx, (data, features, target, II) in enumerate(test_loader):
            data, target = data.to(device,dtype=torch.float),target.to(device,dtype=torch.float)
            features = features.to(device,dtype=torch.float)

            output = model(data, features)
            # output = model(II, features)

            ### pred ###
            pred = (torch.sigmoid(output) > youden_index).int()

            ### loss ###
            loss = criterion(torch.sigmoid(output).squeeze(-1), target)
            test_loss += loss.item()
            corret += pred.eq(target.view_as(pred)).sum().item()

            true_labels = np.append(true_labels, np.array(target.cpu()))
            target_score = np.append(target_score, np.array(output.cpu()))
            prob_score = np.append(prob_score, np.array(torch.sigmoid(output).cpu()))
            pred_labels = np.append(pred_labels, np.array(pred.cpu()))

            print("test: [{} / {}]".format(batch_idx, len(test_loader)-1))

        test_auroc = roc_auc_score(true_labels, prob_score)
        test_auprc = average_precision_score(true_labels, prob_score, average=None)
        FPR,TPR,threshold = roc_curve(true_labels, prob_score)
        auc = auc(FPR,TPR)


        ### confusion matrix ###
        cm_plot(true_labels, prob_score)

        ### calculate ###
        _,_,_,_,sen,spe,ppv,npv = model_performance(true_labels, prob_score)
        f1 = f1_score(true_labels, prob_score)

    t_loss /= len(test_loader)
    t_acc = 100. * corret / total

    print('-----------------------------------------------')
    print("test loss: {:.6f} \t LearningRate {:.8f}".format(test_loss, scheduler.get_lr()[0]))
    print('test set: AUROC: {:.4f}'.format(test_auroc))
    print('Valid set: AUPRC:{:.4f}'.format(test_auprc))
    print('tést set: Fl score: {:.4f}'.format(f1))
    print('test set:Sensitivity=Recall:{:.4f}'.format(sen))
    print('test set:Specificity: {:.4f}'.format(spe))
    print('test set:Precision = PPV:{:.4f}'.format(ppv))
    print('test set:NPV:f:.4f}'.format(npv))
    print('test set:Youden Index:f:.4f}'.format(youden_index))



if __name__ == "__main__":
    args = parse_args()
    main()

