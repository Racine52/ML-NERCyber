from config import *
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='ML-NERCyber')
    parser.add_argument('action', choices=['train', 'predict', 'validate'], help='Action to perform')
    args = parser.parse_args()

    if args.action == 'train':
        os.system('python ./code/huggingFace_train.py')
    elif args.action == 'predict':
        os.system('python ./code/huggingFace_test.py')
    elif args.action == 'validate':
        os.system('python ./code/huggingFace_val.py')

if __name__ == '__main__':
    main()