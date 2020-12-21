"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import urllib.request
import cv2
import numpy as np
import time
URL = "http://192.168.1.3:8080/video" #ip from IP Webcam app


def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)



def show_webcam(mirror=False):
    gate = 1
    samples = []
    i=0
    cam = cv2.VideoCapture(URL)

    #get rescale values to deal with camera size
    ret_val, img = cam.read()
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        if(len(samples)<gate):
            samples.append(img)
        else:
            samples[i] = img
        i=(i+1)%gate

        dst = samples[0]
        for j in range(len(samples)):
        	if j == 0:
        		pass
        	else:
        		alpha = 1.0/(j + 1)
        		beta = 1.0 - alpha
        		dst = cv2.addWeighted(samples[j], alpha, dst, beta, 0.0)

        normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        normalized2 = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
        gc = cv2.bitwise_not(adjust_gamma(cv2.bitwise_not(img), 0.1))
        gc2 = cv2.bitwise_not(adjust_gamma(cv2.bitwise_not(dst), 0.1))
        normalized3 = cv2.normalize(gc2, None, 0, 255, cv2.NORM_MINMAX)
        top = np.concatenate((cv2.resize(dst,dim), cv2.resize(normalized2,dim)), axis = 1) #resize and add to grid
        bottom = np.concatenate((cv2.resize(gc2,dim), cv2.resize(normalized3,dim)), axis = 1) #resize and add to grid
        grid = np.concatenate((top, bottom), axis=0)
        cv2.imshow("Nightvision", grid)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=False)


if __name__ == '__main__':
    main()
