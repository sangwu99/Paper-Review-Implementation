import pandas as pd 
import scipy.sparse as sp
import torch 
import numpy as np
import torch.utils.data as dataload

def make_dataset(neighbor_num,batch_size):
    data_dir = '../../data/'

    train = pd.read_csv(data_dir + 'train_data.csv')
    test = pd.read_csv(data_dir + 'test_data.csv')

    data = pd.concat([train, test])
    
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]

    train_data.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last", inplace=True)
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid2index = {v: i for i, v in enumerate(userid)}
    itemid2index = {v: i for i, v in enumerate(itemid)}
    id2index = dict(userid2index, **itemid2index)
    
    train_data['userID']=train_data['userID'].map(userid2index)
    train_data['assessmentItemID']= train_data['assessmentItemID'].map(itemid2index)
    train_data.reset_index(drop=True,inplace=True)
    
    val_user = train_data.drop_duplicates(subset='userID', keep="last")['userID']
    val_item = train_data.drop_duplicates(subset='userID', keep="last")['assessmentItemID']
    val_label = train_data.drop_duplicates(subset='userID', keep="last")['answerCode']
    val_data = pd.DataFrame({'userID': val_user, 'assessmentItemID': val_item, 'answerCode': val_label})
    
    train_idx = [i for i in range(len(train_data)) if i not in val_data.index]
    train_data = train_data.iloc[train_idx]
    
    train_df = train_data.copy()
    val_df = val_data.copy()
    
    train_data = []
    train_negative =[]
    val_data = []

    for i in range(len(train_df)):
        if train_df['answerCode'].iloc[i] ==0:
            train_negative.append([train_df['userID'].iloc[i], train_df['assessmentItemID'].iloc[i]])
        else:
            train_data.append([train_df['userID'].iloc[i], train_df['assessmentItemID'].iloc[i]])
            
    for i in range(len(val_df)):
        val_data.append([val_df['userID'].iloc[i], val_df['assessmentItemID'].iloc[i],val_df['answerCode'].iloc[i]])
        
    interacted_items = [[] for _ in range(n_user)]
    for (u, i) in train_data:
        interacted_items[u].append(i)

    train_mat = sp.dok_matrix((n_user, n_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0],x[1]] = 1
        
    items_D = np.sum(train_mat, axis = 0).reshape(-1)
    users_D = np.sum(train_mat, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
    A = train_mat.T.dot(train_mat)

    res_mat = torch.zeros((n_item,neighbor_num))
    res_sim_mat = torch.zeros((n_item,neighbor_num))
    
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    
    for i in range(n_item):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, neighbor_num)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))
            
    res_mat = res_mat.long()
    res_sim_mat = res_sim_mat.float()
    
    neg_df = pd.DataFrame()
    sample_user = []
    sample_item = []
    for i,j in train_negative:
        sample_user.append(i)
        sample_item.append(j)
    neg_df['userID'] = sample_user
    neg_df['itemID'] = sample_item
    neg_df = neg_df.groupby('userID').agg({'itemID': lambda x: list(x)})
    
    train_loader = dataload.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=5)
    test_loader = dataload.DataLoader(val_data, batch_size=batch_size, shuffle=True,num_workers=5)

    return train_loader, test_loader, constraint_mat, res_mat, res_sim_mat, neg_df, n_user, n_item, interacted_items
