import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-path",
        default="datasets/barley/stoma/annotations",
        help="path to annotation xmls",
    )
    parser.add_argument(
        "--image-path",
        default="datasets/barley/stoma/images",
        help="path to images",
    )
    parser.add_argument(
        "--output-path",
        default="datasets/barley/stoma/annotations",
        help="A file or directory to save the converted annotations.",
    )
    parser.add_argument(
        "--num-train", type=int, default=18, help="Number of training images"
    )
    parser.add_argument(
        "--shuffled-splits",
        action="store_true",
        help="Sets split generation mode to deterministic uniform sampling",
    )
    parser.add_argument(
        "--arabidopsis", action="store_true", help="Create arabidopsis dataset"
    )
    parser.add_argument(
        "--json", action="store_true", help="Write annotations as separate json files"
    )
    return parser
