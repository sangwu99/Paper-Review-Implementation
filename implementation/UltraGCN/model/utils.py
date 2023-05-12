import numpy as np 
import torch
import pandas as pd
import scipy.sparse as sp

def Sampling(pos_train_data,item_num,interacted_items,sample_num,neg_df):
    neg_items = []
    neg_candidates = np.arange(item_num)
    
    for i in pos_train_data[0]:
        if int(i) not in neg_df.index:
            probs = np.ones(item_num)
            probs[interacted_items[int(i)]] = 0
            probs /= np.sum(probs)
            neg_items.append(np.random.choice(neg_candidates, size = sample_num, p = probs, replace = True).reshape(-1))
        else:    
            if len(neg_df.loc[int(i),'itemID']) < sample_num:
                tmp_list = neg_df.loc[int(i),'itemID']
                size = sample_num - len(neg_df.loc[int(i),'itemID'])
                probs = np.ones(item_num)
                probs[interacted_items[int(i)]] = 0
                probs /= np.sum(probs)
                
                n_neg_items = list(np.random.choice(neg_candidates, size = size, p = probs, replace = True).reshape(-1))
                tmp_list.extend(n_neg_items)
                neg_items.append(np.array(tmp_list))
            else:
                neg_items.append(np.array(neg_df.loc[int(i),'itemID'][:sample_num]))
                         
    return pos_train_data[0], pos_train_data[1], torch.from_numpy(np.array(neg_items))

