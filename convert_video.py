import cv2 as cv
import os
import numpy as np
from config import *


if __name__ == "__main__":
    fps = demo_video_fps
    size = (resize_w, resize_h)
    video_writer = cv.VideoWriter(save_dir+'/vo_pitch_est_res.mp4', cv.VideoWriter_fourcc(*'H264'), fps, size, True)
    images_dir = save_dir+'/matches'
    # images_dir = save_dir+'/epilines'
    file_list = os.listdir(images_dir)
    file_list.sort()
    print(file_list)
    for item in file_list:
        if item.endswith('.jpg'): 
            file = os.path.join(images_dir, item)
            img = cv.imread(file)
            h, w = img.shape[:2]
            if (h != resize_h or w != resize_w): print('size is not correct!')
            video_writer.write(img)
    video_writer.release()