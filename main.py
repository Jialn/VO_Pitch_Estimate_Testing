import cv2 as cv
import os
import numpy as np
import transform_helper as tf
from bundle_adjustment import bundle_adjustment
from plot_utils import draw_epipolar_lines, draw_3d_distance
from config import *

def get_camera_intrinsic_params():
    # np.savetxt(f,data)
    K=np.loadtxt(calibration_file_dir + '/cameras.txt')
    print(K)
    return K.reshape(3,3)

def rep_error_fn(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3,4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))
    rep_error = []
    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])
        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]
        # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])


if __name__ == "__main__":
    # Variables 
    iter = 0
    prev_img = None
    prev_kp = None
    prev_desc = None
    K = np.array(get_camera_intrinsic_params(), dtype=np.float)
    R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    R_t_1 = np.empty((3,4))
    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3,4))
    pts_4d = []
    X = np.array([])
    Y = np.array([])
    Z = np.array([])
    total_valid_pts = 0

    # for opencv with lower version
    # feature_det = cv.xfeatures2d.SIFT_create()
    if use_orb:
        feature_det = cv.ORB_create()
        matcher = cv.BFMatcher(cv.NORM_HAMMING)
    else:
        feature_det = cv.SIFT_create(nfeatures=max_feature_num) 
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=100)
        matcher = cv.FlannBasedMatcher(index_params,search_params)
    for filename in os.listdir(images_dir):
        file = os.path.join(images_dir, filename)
        print(file)
        img = cv.imread(file)

        resized_img = cv.resize(img, (resize_w, resize_h))
        print("detectAndCompute feature")
        kp, desc = feature_det.detectAndCompute(cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY),None)
        
        if iter == 0:
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc
        else:
            # FLANN parameters
            print("find Matcher")
            matches = matcher.knnMatch(prev_desc,desc,k=2)
            pts1 = []
            pts2 = []
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:  #  0.7*n.distance:
                    pts1.append(prev_kp[m.queryIdx].pt)
                    pts2.append(kp[m.trainIdx].pt)
                    
            pts1 = np.array(pts1)
            pts2 = np.array(pts2)
            print("findFundamentalMat")
            F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC, ransacReprojThreshold=1) #
            print("The fundamental matrix \n" + str(F))

            # We select only inlier points
            pts1 = pts1[mask.ravel()==1]
            pts2 = pts2[mask.ravel()==1]
            
            E = np.matmul(np.matmul(np.transpose(K), F), K)

            print("The new essential matrix is \n" + str(E))

            retval, R, t, mask = cv.recoverPose(E, pts1, pts2, K)

            # print("Mullllllllllllll \n" + str(np.matmul(R, R_t_0[:3,:3])))

            R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
            R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())

            print("The R_t_0 \n" + str(R_t_0))
            print("The R_t_1 \n" + str(R_t_1))

            P2 = np.matmul(K, R_t_1)

            # print("The projection matrix 1 \n" + str(P1))
            # print("The projection matrix 2 \n" + str(P2))

            pts1_t = np.transpose(pts1)
            pts2_t = np.transpose(pts2)

            print("Shape pts 1\n" + str(pts1_t.shape))
            total_valid_pts += pts1_t.shape[1]

            points_3d = cv.triangulatePoints(P1, P2, pts1_t, pts2_t)
            points_3d /= points_3d[3]
            # P2, points_3D = bundle_adjustment(points_3d, pts2_t, resized_img, P2)
            # print("done BA!")
            opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
            num_points = len(pts2_t[0])
            rep_error_fn(opt_variables, pts2_t, num_points)

            X = np.concatenate((X, points_3d[0]))
            Y = np.concatenate((Y, points_3d[1]))
            Z = np.concatenate((Z, points_3d[2]))

            R_t_0 = np.copy(R_t_1)
            P1 = np.copy(P2)
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc

            rpy = np.array(tf.mat2euler(R)) * 180.0/3.14159265
            if iter <=2 : rpy[0] = 0
            if rpy[0] > 90:     rpy[0] = rpy[0] - 180.0
            if rpy[0] < -90:    rpy[0] = rpy[0] + 180.0
            if abs(rpy[0]) > 5.0:   rpy[0] = 0
            absrpy = np.array(tf.mat2euler(R_t_1)) * 180.0/3.14159265
            
            img_epiline, img_match = draw_epipolar_lines(pts1, pts2, prev_img, resized_img, F)
            if display_size_distance:
                img_match = draw_3d_distance(img_match, pts1, pts2, points_3d)

            cv.putText(img_match,   "relative rx ry rz:"+str(rpy), (20, 20), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
            cv.putText(img_match,   "abs rx ry rz:"+str(absrpy), (20, 40), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
            cv.putText(img_match,   "relative xyz::"+str(t), (20, 60), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
            
            h, w = img_match.shape[:2]
            horizon_y_offset_percent = horizon_y_offset_percent * (1-pitch_filter_gamma)
            horizon_y_offset_percent += (100*rpy[0]/image_h_fov)

            cv.putText(img_match,   "estimated Pitch::"+str(image_h_fov*horizon_y_offset_percent/100.0), (20, 80), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)

            h_line = (int)(h * (cali_horizon_y_percent - horizon_y_offset_percent) / 100)
            cv.line(img_match, (0, h_line), (w, h_line), [0, 255, 0],1)
            cv.line(img_match, (0, (int)(h *cali_horizon_y_percent/100)), (w, (int)(h *cali_horizon_y_percent/100)), [255, 255, 255],1)
            print("relative RPY:"+str(rpy))
            print("relative zyz:"+str(t))
            zeros_to_append = (4 - len(str(iter))) * '0'
            cv.imwrite(save_dir+'/matches/'+ zeros_to_append + str(iter)+'.jpg', img_match)
            cv.imwrite(save_dir+'/epilines/'+ zeros_to_append+ str(iter)+'.jpg', img_epiline)

        iter = iter + 1
        print("iter:" + str(iter))
    print("average valid pts after ransac:" + str(total_valid_pts / (iter-1)))
