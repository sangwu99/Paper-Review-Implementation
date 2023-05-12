import argparse


def parse_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--embedding_dim", default=32, type=int, help="")
    parser.add_argument("--w1", default=1e-8, type=float, help="")
    parser.add_argument("--w2", default=1, type=float, help="")
    parser.add_argument("--w3", default=1, type=float, help="")
    parser.add_argument("--w4", default=1e-8, type=float, help="")
    
    parser.add_argument("--gamma", default=1e-4, type=float, help="")
    parser.add_argument("--lambda", default=2.75, type=float, help="")    
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="")
    parser.add_argument("--negative_weight", default=20, type=int, help="")
    parser.add_argument("--initial_weight", default=1e-4, type=float, help="")
    parser.add_argument("--early_stop_epoch", default=30, type=int, help="")
    parser.add_argument("--max_epoch", default=500, type=int, help="")
    parser.add_argument("--batch_size", default=512, type=int, help="")
    parser.add_argument("--num_neighbor", default=10, type=int, help="")

    args = parser.parse_args()

    return args
