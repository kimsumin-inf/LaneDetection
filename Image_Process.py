import sys
import os
import cv2
import numpy as np
import math
from loguru import logger
from DataLoader import DataLoader
from BinaryProcess import BinaryProcess
from BEV_transform import BEV
from SlidingWindow import SlidingWindow

class ImageProcess:
    def __init__(self, state:str, path:str, dataset:str):
        self.Image = DataLoader(state, path, dataset).ImageLoader()
        self.MasterBreak = False

    def process(self):
        for i in range(len(self.Image)):
            if not self.MasterBreak:
                try:
                    self.frame = cv2.imread(self.Image[i])
                    resize_img = self.resize(self.frame)
                    roi_img = self.ROI(resize_img)
                    roi_img = self.PrevFiltering(roi_img)
                    self.show("rp", roi_img,1)
                    bin_img = BinaryProcess(roi_img).return_binary_frame()
                    warp_img = BEV(bin_img, state="warp").BEV_transform()
                    unwarp_img = BEV(warp_img,state="unwarp").BEV_transform()
                    ransac_left, ransac_right = SlidingWindow(unwarp_img).line_fit()

                    line_left_y = np.arange(0, roi_img.shape[0] - 1, 1)
                    line_left_x = np.array(list(map(int, ransac_left.predict(line_left_y[:, np.newaxis]))))
                    left = (line_left_x, line_left_y)


                    line_right_y = np.arange(0, roi_img.shape[0] - 1, 1)
                    line_right_x = np.array(list(map(int, ransac_right.predict(line_right_y[:, np.newaxis]))))
                    right = (line_right_x, line_right_y)

                    for i in range(roi_img.shape[0]-1):
                        if line_left_x[i]>= 0 and line_left_x[i]< roi_img.shape[1]:
                            roi_img = cv2.circle(roi_img,(line_left_x[i],line_left_y[i]), 4, (0,255,0), -1)

                        if line_right_x[i]>= 0 and line_right_x[i]< roi_img.shape[1]:
                            roi_img = cv2.circle(roi_img, (line_right_x[i], line_right_y[i]), 4, (255, 0, 0), -1)

                        if (line_left_x[i]>= 0 and line_left_x[i]< roi_img.shape[1]) and (line_right_x[i]>= 0 and line_right_x[i]< roi_img.shape[1]) :
                            roi_img = cv2.circle(roi_img,( round((line_left_x[i]+line_right_x[i])/2),i), 4, (0, 0, 255), -1)



                    self.show("img", roi_img ,1)
                except:
                    self.show("img", roi_img, 1)
                    logger.error("Detect Error")


    def linear_line(self, frame, pt):
        min_y, max_y = frame.shape[0], 0
        alpha = (pt[1][1] - pt[0][1]) / (pt[1][0] - pt[0][0])
        new_min_x = (min_y - pt[0][1]) / alpha + pt[0][0]
        new_max_x = (max_y - pt[0][1]) / alpha + pt[0][0]
        if new_min_x < 0:
            new_min_x = 0
        if new_max_x > frame.shape[1]:
            new_max_x = frame.shape[1]
        return ((int(new_min_x), min_y), (int(new_max_x), max_y))

    def resize(self, frame):
        height, width = frame.shape[:2]
        ratio = round(height/width, 3)
        hoped_width = 720
        hoped_height = round(hoped_width*ratio)
        #logger.info(f"resize shape : {hoped_height, hoped_width}")
        return cv2.resize(frame, (hoped_width, hoped_height))

    def ROI(self, frame):
        height, width = frame.shape[:2]
        frame = frame[round(height/3):, : ]
        return frame

    def PrevFiltering(self, frame):
        filtered = cv2.bilateralFilter(frame, 3, 100, 100)
        return filtered

    def show(self, frame_name, frame, waitkey):
        cv2.imshow(frame_name, frame)
        ch = cv2.waitKey(waitkey)
        if ch ==ord('q') or ch == 27:
            logger.warning(f"The Master Break has been activated.")
            self.MasterBreak = True






if __name__ == "__main__":
    IP= ImageProcess(state="image", path="./dataset/image", dataset="tusimple")
    IP.process()