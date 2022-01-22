import cv2 as cv
import os
import numpy as np
from config import *


if __name__ == "__main__":
    fps = demo_video_fps
    size = (resize_w, resize_h)
    # video_enco_format = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(save_dir+'/vo_pitch_est_res.mp4', cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    images_dir = save_dir+'/matches'
    file_list = os.listdir(images_dir)
    file_list.sort()
    print(file_list)
    for item in file_list:
        if item.endswith('.jpg'): 
            file = os.path.join(images_dir, item)
            img = cv.imread(file)
            video_writer.write(img)
    video_writer.release()