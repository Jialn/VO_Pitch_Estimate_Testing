
import os

######################### Path Variables ##################################################
curr_dir_path = os.getcwd()
images_dir = curr_dir_path + '/data/images/observatory'
save_dir = curr_dir_path + '/data/images'
calibration_file_dir = curr_dir_path + '/data/calibration'
###########################################################################################

use_orb = False # use orb or sift
cali_horizon_y_percent = 52.5
image_h_fov = 38 # deg
horizon_y_offset_percent = 0.0

max_feature_num = 9999
# resize_w, resize_h = 1600, 900
resize_w, resize_h = 1280, 720
# resize_w, resize_h = 800, 450
# resize_w, resize_h = 640, 360
pitch_filter_gamma = 0.01

demo_video_fps = 15
draw_distance_to_image = False
