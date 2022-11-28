import cv2
import numpy as np

class BinaryProcess:
    def __init__(self, frame):
       self.frame = frame

    def get_lb_frame(self, sigma=0.33):
        l_frame = cv2.split(cv2.cvtColor(self.frame, cv2.COLOR_BGR2LAB))[0]
        MidValue = self.get_midvalue(l_frame)
        HighValue = (1.0 + sigma) * MidValue
        if HighValue > 250:
            HighValue = 250
        _, lb_frame = cv2.threshold(l_frame, HighValue, 255, cv2.THRESH_BINARY)

        return lb_frame

    def get_sb_frame(self, sigma=0.6):
        s_frame = cv2.split(cv2.cvtColor(self.frame, cv2.COLOR_BGR2HLS))[2]
        MidValue = self.get_midvalue(s_frame)
        LowValue = (1.0 - sigma) * MidValue
        if LowValue < 40:
            LowValue = 40

        _, sb_frame = cv2.threshold(s_frame, LowValue, 255, cv2.THRESH_BINARY)

        return sb_frame

    def return_binary_frame(self):
        lb_img = self.get_lb_frame()
        sb_img = self.get_sb_frame()
        return self.xor_process(lb_img, sb_img, iteration=1)


    def show(self, frame_name, frame, waitkey):
        cv2.imshow(frame_name, frame)
        ch = cv2.waitKey(waitkey)

    def xor_process(self,lb_frame, sb_frame, iteration=1):
        xor_frame = None
        lb = lb_frame
        sb = sb_frame


        for _ in range(iteration):
            xor_frame = cv2.bitwise_xor(lb, sb)
            sb = cv2.subtract(sb, xor_frame)
            xor_frame = cv2.bitwise_xor(lb, sb)

            lb = cv2.subtract(lb, xor_frame)


        result = cv2.bitwise_or(sb, lb)
        #result = cv2.dilate(result,(1,3), iterations=3)
        #result = cv2.dilate(result, (3, 3), iterations=3)
        return result


    def get_midvalue(self, frame):
        return round(np.median(list(set(frame.ravel()))))