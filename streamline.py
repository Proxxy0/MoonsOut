#!/usr/bin/env python3

__author__ = "Noah Worley"
__copyright__ = "Copyright 2020"
__credits__ = [""]
__license__ = "GPLv3"
__version__ = "0.1.0"
__maintainer__ = "Noah Worley"
__email__ = "noah.worley921@gmail.com"
__status__ = "development"


'''import libraries'''
import urllib.request
import cv2
import numpy as np
from pylab import *

'''global values'''
URL                    = "http://192.168.1.3:8080/video" #ip webcam address
gate                   = 1 # number of images in the sample set
alpha				   = 3
beta				   = 0
gamma				   = 0.1
font                   = cv2.FONT_HERSHEY_COMPLEX
bottomLeftCornerOfText = (10,20)
fontScale              = 0.7
fontColor              = (255,255,255)
lineType               = 1


'''functions'''
#adjusts the gamma of an image
def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

#rescales an image
def rescale(img, scale=None, width=None, height=None):
	if(width == None and height == None):
		width = int(img.shape[1] * scale)
		height = int(img.shape[0] * scale)
		dim = (width, height)

		return cv2.resize(img,dim)
	else:
		if(width == None):
			heightscale = float(height/img.shape[0])
			width = int(img.shape[1] * heightscale)
			dim = (width, height)

			return cv2.resize(img,dim)
		else:
			widthscale = float(width/img.shape[1])
			height = int(img.shape[0] * widthscale)
			dim = (width, height)

			return cv2.resize(img,dim)

#stacks a series of images to reduce noise
def stack(samples):
	dst = samples[0]
	for j in range(len(samples)):
		if j == 0:
			pass
		else:
			alpha = 1.0/(j + 1)
			beta = 1.0 - alpha
			dst = cv2.addWeighted(samples[j], alpha, dst, beta, 0.0)

	return dst

#adjusts the brightness and contrast of an image using alpha/beta values
def brightContrast(img, alpha, beta):
	return cv2.addWeighted(img,alpha,np.zeros(img.shape, img.dtype),0,beta)

#normalizes the pixel values to range over 0-255
def normalize(img):
	return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

#inverts, adjusts gamma, and reinverts an image
def doubleGC(img, gamma):
	return cv2.bitwise_not(adjust_gamma(cv2.bitwise_not(img), 0.1))

#adds a text overlay to an image
def putText(img,text):
	return cv2.putText(img,text,
					bottomLeftCornerOfText,
					font, fontScale, fontColor,
					lineType)

#shows a live feed of the webcam
def show_webcam(mirror=False, mobile = False):
	#get the camera to be used
	cam = cv2.VideoCapture(URL) if mobile else cv2.VideoCapture(0)

	#initialize sample and index
	samples = []
	i=0

	#run continuously until breakpoint is hit
	while True:
		#gets a frame from the camera
		ret_val, img = cam.read()
		#mirrors image if needed
		if mirror: img = cv2.flip(img, 1)

		#sets up samples for stacking
		if(len(samples)<gate): samples.append(img)
		else: samples[i] = img
		i=(i+1)%gate



		#stacks samples
		dst = rescale(stack(samples),height = 200)

		#split channels and apply transforms
		b,g,r = cv2.split(dst)
		r2=brightContrast(r,alpha,beta); g2=brightContrast(g,alpha,beta); b2=brightContrast(b,alpha,beta)
		r3=normalize(r); g3=normalize(g); b3=normalize(b);
		r4=doubleGC(r,gamma); g4=doubleGC(g,gamma); b4=doubleGC(b,gamma);
		r5=normalize(r4); g5=normalize(g4); b5=normalize(b4);

		#remerge channels and apply transforms to base
		rm = cv2.merge([b,g,r]); rm2 = cv2.merge([b2,g2,r2]); rm3 = cv2.merge([b3,g3,r3]); rm4 = cv2.merge([b4,g4,r4]); rm5 = cv2.merge([b5,g5,r5])
		dst2 = brightContrast(dst,alpha,beta); dst3 = normalize(dst); dst4 = doubleGC(dst, gamma); dst5 = normalize(dst4)

		#caption the feeds
		r=putText(r,'Red'); g=putText(g,'Green'); b=putText(b,'Blue')
		r2=putText(r2,'Red | Cont. boost'); g2=putText(g2,'Green | Cont. boost'); b2=putText(b2,'Blue | Cont. boost')
		r3=putText(r3,'Red | Normalized'); g3=putText(g3,'Green | Normalized'); b3=putText(b3,'Blue | Normalized')
		r4=putText(r4,'Red | DGC'); g4=putText(g4,'Green | DGC'); b4=putText(b4,'Blue | DGC')
		r5=putText(r5,'Red | DGC Norm.'); g4=putText(g5,'Green | DGC Norm.'); b4=putText(b5,'Blue | DGC Norm.')
		rm=putText(rm,"RGB Merged");rm2=putText(rm2,"Cont. boost RBG Merged");rm3=putText(rm3,"Normalized RBG Merged");rm4=putText(rm4,"DGC RBG Merged");rm5=putText(rm5,"DGC Norm. RBG Merged")
		dst=putText(dst,"Base");dst2=putText(dst2,"Cont. boost Base");dst3=putText(dst3,"Normalized Base");dst4=putText(dst4,"DGC Base");dst5=putText(dst5,"DGC Norm. Base")



		#create grid of images from split channels and show it
		cv2.imshow("Color channels", vstack((hstack((r,		g,	b)),
											 hstack((r2,	g2,	b2)),
											 hstack((r3,	g3,	b3)),
											 hstack((r4,	g4,	b4)),
											 hstack((r5,	g5,	b5)))))

		#create grid of images from remerged and base images and show it
		cv2.imshow("Merged comparison", vstack((hstack((rm,		dst)),
											 	hstack((rm2,	dst2)),
											 	hstack((rm3,	dst3)),
											 	hstack((rm4,	dst4)),
											 	hstack((rm5,	dst5)))))

		#show summed split channels
		#cv2.imshow("summed channels", (r+g+b)*6)

		#breakpoint
		if cv2.waitKey(1) == 27:
			break  # esc to quit
	cv2.destroyAllWindows()

#main function
def main():
	show_webcam(mirror=False, mobile=False)

'''run on startup'''
if __name__ == '__main__':
	main()
