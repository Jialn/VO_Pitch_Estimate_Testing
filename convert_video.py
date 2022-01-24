from anyio import start_blocking_portal
import cv2 as cv
import os
import numpy as np
from config import *

image_start_position = 70

if __name__ == "__main__":
    fps = demo_video_fps
    images_dir = save_dir+'/matches'
    # images_dir = save_dir+'/epilines'
    
    # get image size
    file_list = os.listdir(images_dir)
    file_list.sort()
    print(file_list)
    file = os.path.join(images_dir, file_list[1])
    img = cv.imread(file)
    h, w = img.shape[:2]
    size = (w, h)
    # set video writer
    video_writer = cv.VideoWriter(save_dir+'/vo_pitch_est_res.mp4', cv.VideoWriter_fourcc(*'H264'), fps, size, True)

    iter_cnt = 1
    for item in file_list:
        if item.endswith('.jpg'): 
            if iter_cnt >= image_start_position:
                file = os.path.join(images_dir, item)
                img = cv.imread(file)
                video_writer.write(img)
            iter_cnt += 1
    video_writer.release()