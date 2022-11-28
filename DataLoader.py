import sys
import os
import cv2
import numpy
import numpy as np
from glob import glob
from loguru import logger
from natsort import natsorted

class DataLoader:
    def __init__(self, state:str, path:str, dataset:str):
        self.state = state
        self.path = path
        self.dataset = dataset

    def ImageLoader(self):
        if not os.path.isdir(self.path):
            logger.error("Image directory does not exist")

        if len(os.listdir(self.path)) == 0:
            logger.error("Image directory is empty")

        if self.dataset == "tusimple":
            tmp = []
            image = sorted(glob(self.path + "/*"))
            for i in range(len(image)):
                image[i] = sorted(glob(image[i] + "/*"))
                for j in range(len(image[i])):
                    image[i][j] = natsorted(
                        glob(image[i][j] + "/*.jpg") + glob(image[i][j] + "/*.png") + glob(image[i][j] + "/*.jpeg"))
                    tmp.append(image[i][j])
            ImgSet = np.array(tmp).ravel()
            return ImgSet

    def VideoLoader(self):
        return self.path



if __name__ == "__main__":
    DL = DataLoader(path="./dataset/image", dataset="tusimple")
    DL.ImageLoader()