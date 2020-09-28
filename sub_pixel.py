import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from skimage import measure
from scipy.spatial import distance as dist
import argparse
import glob
from math import log10, sqrt 

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('vid1.h264')
#_,frame1=cap.read()

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def ssim(imageA, imageB):
	err = measure.compare_ssim(test, marker)
	return err

def psnr(imageA, imageB): 
    mse = np.mean((imageA - imageB) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    err = 20 * log10(max_pixel / sqrt(mse)) 
    return err 

original = cv2.imread("background.jpg", 0)
#original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
parameters =  cv2.aruco.DetectorParameters_create()

score=[]
res=[]
ang=np.arange(-5,5,0.01)

while 1:
    #frame=frame1.copy()
    _,frame=cap.read()
    frame = frame[240:480,540:800]
    score=[]
    score_s = []
    score_p = []
    #frame=cv2.imread('IMG_20200926_143121.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids==1:
        i0,j0=np.argwhere(ids==1)[0][0],np.argwhere(ids==1)[0][1]
        x1=corners[i0][j0][0][0]
        x2=corners[i0][j0][1][0]
        x3=corners[i0][j0][2][0]
        x4=corners[i0][j0][3][0]

        y1=corners[i0][j0][0][1]
        y2=corners[i0][j0][1][1]
        y3=corners[i0][j0][2][1]
        y4=corners[i0][j0][3][1]

        min_x=int(np.min(np.array([x1,x2,x3,x4])))
        max_x=int(np.max(np.array([x1,x2,x3,x4])))
        min_y=int(np.min(np.array([y1,y2,y3,y4])))
        max_y=int(np.max(np.array([y1,y2,y3,y4])))        

        frame=cv2.rectangle(frame, (min_x,min_y), (max_x,max_y), (0,255,0), 2) 

        marker = gray[min_y:max_y,min_x:max_x]
        h,w=marker.shape
        original = cv2.resize(original,(w,h))
        for z in ang:
            test=imutils.rotate_bound(original, z)
            test = cv2.resize(test,(w,h))
            score.append(round(mse(test, marker),2))
            score_s.append(round(ssim(test, marker),2))
            score_p.append(round(psnr(test, marker),2))
##        appr=[]
##        r=np.polyfit(ang,score,4)
##        for ii in np.arange(-5,5,0.01):
##            #appr.append(r[0]*i**3+r[1]*i**2+r[2]*i+r[3])
##            appr.append(r[0]*ii**4+r[1]*ii**3+r[2]*ii**2+r[3]*ii+r[4])
##        k=np.where(np.array(appr)==np.min(appr))[0][0]
##        res=round(np.arange(-5,5,0.01)[k],3)
        #plt.plot(ang, score, 'o')
        fig, fig_mse = plt.subplots()
        fig_mse.set(title='mse')
        fig_mse.plot(ang, score, 'b-')
        #plt.plot(ang, score, 'b-')
        fig, fig_ssim = plt.subplots()
        fig_ssim.plot(ang, score_s, 'b-')
        fig_ssim.set(title='ssim')
        #plt.plot(ang, score_s, 'b-', color = 'red')
        fig, fig_psrn = plt.subplots()
        fig_psrn.plot(ang, score_p, 'b-')
        fig_psrn.set(title='psrn')
        #plt.plot(ang, score_p, 'b-', color = 'green')
        #plt.plot(np.arange(-5,5,0.01),appr, 'r-')
        #plt.title('res='+str(res))
        plt.show()

            
        #print('score', score)
##    cv2.putText(frame, str(res), (min_x, min_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
##    #frame=cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
##    cv2.imshow('frame', frame)
##    key = cv2.waitKey(1) & 0xFF
##    if key == ord("q"):
##        break

##appr=[]
##r=np.polyfit(ang,score,4)
##for ii in np.arange(-5,5,0.01):
##    #appr.append(r[0]*i**3+r[1]*i**2+r[2]*i+r[3])
##    appr.append(r[0]*ii**4+r[1]*ii**3+r[2]*ii**2+r[3]*ii+r[4])
##
##k=np.where(np.array(appr)==np.min(appr))[0][0]
##res=round(np.arange(-5,5,0.01)[k],2)
##plt.plot(ang, score, 'b-')
##plt.plot(np.arange(-5,5,0.01),appr, 'r-')
##plt.title('res='+str(res))
##plt.show()



##cv2.destroyAllWindows()
cap.release()


