import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-directory", type=str, default="data/en")
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-window_size", type=int, default=3)
    parser.add_argument("-num_neg_samples", type=int, default=5)
    parser.add_argument("-embedding_dim", type=int, default=10),
    parser.add_argument("-learning_rate", type=float, default=0.01)
    parser.add_argument("-epochs", type=int, default=5)
    parser.add_argument("-log_every", type=int, default=100)
    return parser.parse_args()