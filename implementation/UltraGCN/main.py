import os
import argparse
import torch
from dataset.dataset import make_dataset
from model.UltraGCN import UltraGCN
from model.train import train

from args import parse_args

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, constraint_mat, res_mat, res_sim_mat, neg_df, n_user, n_item, interacted_items = make_dataset(params['num_neighbor'],params['batch_size'])
    params['user_num'] = n_user
    params['item_num'] = n_item
    params['device'] = device
    
    Ultra = UltraGCN(params, constraint_mat, res_sim_mat, res_mat)
    Ultra = Ultra.to(device)
    
    optimizer = torch.optim.Adam(Ultra.parameters(), lr=params['learning_rate'])
    
    train(Ultra, optimizer, train_loader, test_loader, params, interacted_items, neg_df)
    
    
if __name__ == "__main__":
    args = parse_args()
    params = vars(args)
    main(params)