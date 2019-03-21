import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        default='None',
        help='The Configuration file',
        required=True)

    train_or_visualize = argparser.add_mutually_exclusive_group()

    train_or_visualize.add_argument(
        "-m", "--mode",
        dest="mode",
        choices=["train", "vis"],
        default="train",
        help="Train or visualize network."
    )
    args = argparser.parse_args()
    return args
