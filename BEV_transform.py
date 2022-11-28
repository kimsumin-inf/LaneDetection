import cv2
import numpy as np

class BEV:
    def __init__(self, frame, state):
        self.frame = frame
        self.state = state

    def BEV_transform(self):
        height, width = self.frame.shape[:2]

        src = np.float32(
            [[width/3, height/6],
            [0, height],
            [width, height],
            [width*2/3,height/6]]
        )

        dst = np.float32(
            [[-width/5,0],
            [width*1/7, height],
            [width*6/7, height],
            [width*6/5, 0]]
        )


        if self.state=="warp":
            warped = cv2.warpPerspective(self.frame, cv2.getPerspectiveTransform(src, dst), (width, height),
                                         flags=cv2.INTER_LINEAR)
            return warped

        elif self.state=="unwarp":
            unwarped = cv2.warpPerspective(self.frame, cv2.getPerspectiveTransform(dst, src), (width, height),
                                           flags=cv2.INTER_LINEAR)
            return unwarped



