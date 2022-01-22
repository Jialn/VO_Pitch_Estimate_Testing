import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def viz_3d_matplotlib(pt_3d):
    X = pt_3d[0,:]
    Y = pt_3d[1,:]
    Z = pt_3d[2,:]

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X,
               Y,
               Z,
               s=1,
               cmap='gray')
    
    plt.show()

def draw_3d_distance(img, pts1, pts2, points_3d):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    itera = 0
    for pt1,pt2 in zip(pts1,pts2):
        # color = tuple(np.random.randint(0,255,3).tolist())
        # p3d_str = "%.1f, %.1f, %.1f" % (points_3d[0][itera], points_3d[1][itera], points_3d[2][itera])
        distance = points_3d[0][itera]*points_3d[0][itera] + points_3d[1][itera]*points_3d[1][itera] + points_3d[2][itera]*points_3d[2][itera]
        distance = np.sqrt(distance)
        p3d_str = "%.1f" % distance
        cv.putText(img, p3d_str, (pt1[0], pt1[1]), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
        itera += 1
    return img

def draw_epipolar_lines(pts1, pts2, img1, img2, F):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = draw_epilines(img2.copy(),img1.copy(),lines2,pts2,pts1)
    img_match = drawmatch_img1(img2,img1,pts2,pts1)
    # plt.subplot(223)
    # plt.imshow(img3)
    # plt.subplot(224)
    # plt.imshow(img5)
    # plt.subplot(221)
    # plt.imshow(img_match)
    # plt.show()
    return img3, img_match

def drawmatch_hstack(img1, img2, pts1, pts2):
    ''' draw the corresponding matches '''
    img = np.hstack((img1, img2))
    h, w = img1.shape[:2]
    for pt1,pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        img = cv.line(img, tuple(pt1), (w+pt2[0],pt2[1]), color,1)
    return img

def drawmatch_img1(img1, img2, pts1, pts2):
    ''' draw the corresponding matches '''
    # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    # img1[:,:,0] = img1[:,:,0]*img2[:,:,0]//255
    # img1 = (img1 * 3 // 4 + img2 // 4).astype(np.uint8)
    color_cnt = 0
    for pt1,pt2 in zip(pts1,pts2):
        color_cnt += 1
        color = [63,63,255]
        img1 = cv.line(img1, tuple(pt1), (pt2[0],pt2[1]), color,2)
        img1 = cv.circle(img1,tuple(pt1),4,color,1)
    return img1
    
def draw_epilines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2