import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dataloader import generate_dataloader
from utils import Logger, adjust_learning_rate, CraateLogger,create_cosine_learing_schdule,encode_test_label,set_seed
from model.KEDR import FusionNet
from dependency import *
from torch import optim
from torchcontrib.optim import SWA
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

class_names = ['NEV', 'BCC', 'MEL','MISC','SK']
    
def criterion(logit, truth):

    loss = nn.CrossEntropyLoss()(logit, truth)

    return loss

def metric(logit, truth):
    # prob = F.sigmoid(logit)
    _, prediction = torch.max(logit.data, 1)

    acc = torch.sum(prediction == truth)
    return acc

def train(net,train_dataloader,model_name):

    #net.set_mode('train')
    train_loss = 0
    train_dia_acc = 0  
    train_sps_acc = 0
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(train_dataloader):
        opt.zero_grad()
        
        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()
#         meta_data    = meta_data.cuda()
        
        # Diagostic label
        diagnosis_label = label[0].long().cuda()

        #print()
        logit_diagnosis_fusion, features, mine_loss, c_loss,_,_,_ = net((clinic_image, derm_image), diagnosis_label, missing=False)
        
        loss_fusion = criterion(logit_diagnosis_fusion, diagnosis_label)           
        loss =  loss_fusion + mine_loss+ c_loss

        dia_acc_fusion = torch.true_divide(metric(logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))

        dia_acc = dia_acc_fusion

        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_dia_acc += dia_acc.item()
#         train_sps_acc += sps_acc.item()

    train_loss = train_loss / (index + 1) # Because the index start with the value 0f zero
    train_dia_acc = train_dia_acc / (index + 1)
#     train_sps_acc = train_sps_acc / (index + 1)

    return train_loss,train_dia_acc

def validation(net,val_dataloader,model_name, epoch):
    net.eval()
    val_loss = 0
    val_dia_acc = 0
    vaL_sps_acc = 0

    
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []

    original_missing_all = []
    reconstructed_all = []

    # 收集 TCP confidence 值
    fundus_conf_list = []
    oct_conf_list = []    
    fusion_conf_list = []
    
    
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(val_dataloader):

        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()
#         meta_data    = meta_data.cuda()

        diagnosis_label = label[0].long().cuda()

        with torch.no_grad():
          
          
            logits, features, mine_loss, c_loss,TCPConfidence_clic, TCPConfidence_derm, TCPConfidence_shared = net((clinic_image, derm_image), diagnosis_label, missing=True)
            
            loss = criterion(logits, diagnosis_label)
            loss = loss + mine_loss+ c_loss
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits.data, 1)
            
            
            acc = torch.true_divide(metric(logits, diagnosis_label), clinic_image.size(0))
            
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(diagnosis_label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())

            # 提取每个 batch 的 TCP confidence
            fundus_conf_list.extend(TCPConfidence_clic.view(-1).cpu().numpy())
            oct_conf_list.extend(TCPConfidence_derm.view(-1).cpu().numpy())
            fusion_conf_list.extend(TCPConfidence_shared.view(-1).cpu().numpy())
  

        val_loss += loss.item()
        val_dia_acc += acc.item()
#         vaL_sps_acc += sps_acc.item()


    
    val_loss = val_loss / (index + 1)
    val_dia_acc = val_dia_acc / (index + 1)
    
    log.write('\nValidation Metrics:\n')
    for metric_name, value in metrics_dict.items():
        log.write(f'{metric_name}: {value:.4f}\n')

    return val_loss,val_dia_acc


def run_train(model_name,mode,i):
    log.write('** start training here! **\n')
    #best_acc = 0
    es = 0
    patience = 50
    best_mean_acc = 0 
    best_loss = 300
    
    for epoch in range(epochs):
        swa_lr = cosine_learning_schule[epoch]
        adjust_learning_rate(opt, swa_lr)

        # train_mode
        train_loss,train_dia_acc = train(net, train_dataloader,model_name)
        log.write('Round: {}, epoch: {}, Train Loss: {:.4f}, Train Dia Acc: {:.4f}\n'.format(i, epoch, train_loss,
                                                                                                         train_dia_acc
                                                                                                         ))

        # validation mode
        val_loss,val_dia_acc = validation(net, val_dataloader,model_name, epoch)
        
        val_acc = val_dia_acc
        val_mean_acc = val_dia_acc
        
        log.write('Round: {}, epoch: {}, Valid Loss: {:.4f}, Valid Dia Acc: {:.4f}\n'.format(i, epoch, val_loss,
                                                                                                         val_dia_acc
                                                                                                         ))

     
        if val_mean_acc > best_mean_acc:
            es = 0
            best_mean_acc = val_mean_acc
            #torch.save(net, out_dir + '/checkpoint/{diag_label_guided_gating}_best_model.pth')
            log.write('Current Best Mean Acc is {}'.format(best_mean_acc))
        #  else:
        #      es += 1
        #      print("Counter {} of {}".format(es,patience))
          
        #      if es > patience:
        #          print("Early stopping with best_mean_acc: {:.4f}".format(best_mean_acc), "and val_mean_acc for this epoch: {:.4f}".format(val_mean_acc))
        #          break
  
        #if epoch == 150:
            #torch.save(net, out_dir + '/checkpoint/{diag_label_guided_gating}_model.pth')
        if epoch > (epochs - swa_epoch) and epoch % 1 == 0:
            opt.update_swa()
            log.write('SWA Epoch: {}'.format(epoch))

    #torch.save(net, out_dir+'/swa_{}_resnet50_model.pth')

        
if __name__ == '__main__':
    # Hyperparameters
    
    mode = 'multimodal'
    model_name = 'missing=0.8'
    shape = (224, 224)
    batch_size = 32
    num_workers = 8
    data_mode = 'Normal'
    deterministic = True
    if deterministic:
        if data_mode == 'Normal':
          random_seeds = 42
        elif data_mode == 'self_evaluated':
          random_seeds = 183
    rounds = 1
    lr = 1e-5
    epochs = 250
    swa_epoch = 50

    train_dataloader, val_dataloader = generate_dataloader(shape, batch_size, num_workers, data_mode)
    
    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)
      # create logger
        print(random_seeds+i)
        log, out_dir = CraateLogger(mode, model_name,i,data_mode)
        net = FusionNet(num_classes=5, p_missing=0.8, dataset_name="SPC").cuda()
        #net.initialize_memory(train_dataloader)
      # create optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt = SWA(optimizer)
      # create learning schdule
        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        run_train(model_name,mode,i)
