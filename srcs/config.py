import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-directory", 
        type=str, 
        default="data/en"
        )
    
    return parser.parse_args()