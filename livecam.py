"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)



def show_webcam(mirror=False):
    gate = 5
    samples = []
    i=0
    cam = cv2.VideoCapture(0)
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
        #cv2.imshow('my webcam - base', img)
        #cv2.imshow('my webcam - normalized', normalized)
        #top = np.concatenate((img, normalized, gc), axis=1)
        #bottom = np.concatenate((dst, normalized2, gc2), axis=1)
        top = np.concatenate((dst, normalized2), axis = 1)
        bottom = np.concatenate((gc2, normalized3), axis = 1)
        grid = np.concatenate((top, bottom), axis=0)
        cv2.imshow("Nightvision", grid)
        #cv2.imshow('my webcam - stacked', dst)
        #cv2.imshow('my webcam - stacked normalized', normalized2)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
