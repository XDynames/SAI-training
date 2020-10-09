import argparse
import json

'''
    Records predictions and ground truth pairs for an image as a JSON file
'''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-dir",
        help="path to gt annotation json",
    )
    parser.add_argument(
        "--anno-dir",
        help="path to human annotation json",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output",
    )
    return parser
 
 def read_json(self, filepath):
    with open(filepath) as file:
        raw_json = json.load(file)
    return raw_json

if __name__=='__main__':
    args = get_parser().parse_args()
    raw_json = read_json(args.anno_dir)
    