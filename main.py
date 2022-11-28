import sys
import argparse
from loguru import logger #pip install loguru
from Image_Process import ImageProcess

def param_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--type", default="image", help="Your dataset type")
    parser.add_argument('-d', "--dataset", default="tusimple", help="Choose 'custom' or 'tusimple'")
    parser.add_argument('-p', "--path", default="./dataset/image", help="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = param_parsing()
    if args.type == "image":
        logger.info("Path must direct Image path")
        IP = ImageProcess(state="image", path="./dataset/image", dataset="tusimple")
        IP.process()
        pass
    elif args.type=="video":
        logger.info("Path must direct Video file")
        pass
    else:
        logger.error("Bad type")
