import torch.nn as nn 
import torch 
import torch.nn.functional as F

import numpy as np 
import pandas as pd
import scipy.sparse as sp
import time

from sklearn.metrics import roc_auc_score
from model.utils import Sampling

def train(model, optimizer, train_loader, test_loader, params,interacted_items,neg_df): 
    device = params['device']
    best_AUC = 0
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // params['batch_size']
    if len(train_loader.dataset) % params['batch_size'] != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))
    
    # if params['enable_tensorboard']:
    #     writer = SummaryWriter()

    for epoch in range(params['max_epoch']):
        model.train() 
        start_time = time.time()

        for batch, x in enumerate(train_loader): # x: tensor:[users, pos_items]
            users, pos_items, neg_items = Sampling(x,params['item_num'],interacted_items,params['negative_weight'],neg_df)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            model.zero_grad()
            loss = model(users, pos_items, neg_items)
            # if params['enable_tensorboard']:
                # writer.add_scalar("Loss/train_batch", loss, batches * epoch + batch)
            loss.backward()
            optimizer.step()
        
        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        # if params['enable_tensorboard']:
            # writer.add_scalar("Loss/train_epoch", loss, epoch)
        print(f'{epoch} epoch, {train_time} time')

        need_test = True
        if epoch < 50 and epoch % 5 != 0:
            need_test = False
            
        if need_test:
            start_time = time.time()
            AUC = test(model, test_loader)
            # if params['enable_tensorboard']:
            #     writer.add_scalar('Results/recall@20', Recall, epoch)
            #     writer.add_scalar('Results/ndcg@20', NDCG, epoch)
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
            
            print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            print("Loss = {:.5f}, AUC: {:5f}".format(loss.item(),AUC))

            if AUC > best_AUC:
                best_AUC = AUC
                early_stop_count = 0
                best_epoch = epoch
                # torch.save(model.state_dict(), params['model_save_path'])

            else:
                early_stop_count += 1
                if early_stop_count == params['early_stop_epoch']:
                    early_stop = True
        
        if early_stop:
            print('##########################################')
            print('Early stop is triggered at {} epochs.'.format(epoch))
            print('Results:')
            print('best epoch = {}, best auc = {}'.format(best_epoch, best_AUC))
            # print('The best model is saved at {}'.format(params['model_save_path']))
            break

    # writer.flush()

    print('Training end!')
    
def test(model, test_loader):
    AUC = 0 
    with torch.no_grad():
        model.eval()
        for idx, (batch_users,batch_items, label) in enumerate(test_loader):
            
            batch_users = batch_users.to(model.get_device())
            batch_items = batch_items.to(model.get_device())
            label = label.to(model.get_device())
            pred=model.test_foward(batch_users,batch_items)
            
            AUC += roc_auc_score(label.cpu().numpy(),pred.cpu().numpy())
    AUC = AUC / len(test_loader)

    return AUC