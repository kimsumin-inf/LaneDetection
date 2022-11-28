import cv2
import numpy as np
from LineFitting import LineFitting
from sklearn.linear_model import LinearRegression
from sklearn. linear_model import RANSACRegressor

class SlidingWindow:
    def __init__(self, frame):
        self.frame = frame
        self.line_fit()
        self.left = []
        self.right = []


    def line_fit(self):
        out_img = (np.dstack((self.frame, self.frame, self.frame))).astype('uint8')
        tmp_frame = np.zeros_like(out_img)
        midpoint = round(self.frame.shape[1]/2)
        left_y, left_x= self.frame[:,0:midpoint].nonzero()
        left_x = left_x.reshape(-1,1)
        left_y = left_y.reshape(-1,1)
        ransac_left = RANSACRegressor()#LinearRegression(), max_trials=100, min_samples=60, loss='absolute_error', residual_threshold=10, random_state=42)
        ransac_left.fit(left_y, left_x)



        right_y, right_x = self.frame[:, midpoint:].nonzero()
        right_x = right_x.reshape(-1, 1) + midpoint
        right_y = right_y.reshape(-1, 1)
        ransac_right = RANSACRegressor()#LinearRegression(), max_trials=100, min_samples=60, loss='absolute_error', residual_threshold=10, random_state=42)
        ransac_right.fit(right_y, right_x)

        return ransac_left, ransac_right




    def show(self, frame_name, frame, waitkey):
        cv2.imshow(frame_name, frame)
        ch = cv2.waitKey(waitkey)

    def func(self, X, a, b, c):
        return a * (X ** 2) + b * (X) + c