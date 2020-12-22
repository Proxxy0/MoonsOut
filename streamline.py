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
fontColor              = (0,0,255)
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
def brightContrast(img):
	return cv2.addWeighted(img,alpha,np.zeros(img.shape, img.dtype),0,beta)

#normalizes the pixel values to range over 0-255
def normalize(img):
	return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

#inverts, adjusts gamma, and reinverts an image
def doubleGC(img):
	return cv2.bitwise_not(adjust_gamma(cv2.bitwise_not(img), 0.1))

#adds a text overlay to an image
def putText(img,text):
	return cv2.putText(img,text,
					bottomLeftCornerOfText,
					font, fontScale, fontColor,
					lineType)

'''splits an image into color channels, applies filters to each channel and base,
remerges channels and converts colorspaces of each channel, returning each
variation'''
def splitFilter(img, actions = None, desc = None):
	base  = img.copy()
	b,g,r = cv2.split(img)

	s = "Base"
	bs = "Blue"
	gs = "Green"
	rs = "Red"
	ms = "Merged"

	if(actions != None):
		for action in actions:
			base = action(base)
			b = action(b)
			g = action(g)
			r = action(r)

		s+=" | "+desc
		bs+=" | "+desc
		gs+=" | "+desc
		rs+=" | "+desc
		ms+=" | "+desc

	m = cv2.merge([b,g,r])

	b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
	g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
	r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

	base = putText(base,s)
	b = putText(b,bs)
	g = putText(g,gs)
	r = putText(r,rs)
	m = putText(m,ms)

	return base,b,g,r,m

'''applies filters and adds a caption to an image'''
def filter(img, actions = None, desc = None):
	base = img.copy()

	s = "Base"
	if(actions != None):
		for action in actions:
			base = action(base)

		s+=" | "+desc

	base = putText(base,s)

	return base



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
		dst = rescale(stack(samples),height = 180)

		#split channels and apply transforms
		base1,b1,g1,r1,merge1 = splitFilter(dst)
		base2,b2,g2,r2,merge2 = splitFilter(dst, actions = [brightContrast], 	desc = "Cont. boost")
		base3,b3,g3,r3,merge3 = splitFilter(dst, actions = [normalize], 		desc = "Norm.")
		base4,b4,g4,r4,merge4 = splitFilter(dst, actions = [doubleGC],			desc = "DGC")
		base5,b5,g5,r5,merge5 = splitFilter(dst, actions = [doubleGC,normalize],desc = "DGC Norm.")

		#create grid of images from split channels and show it
		cv2.imshow("MoonsOut | Dark Vision", vstack((hstack((r1,	g1,	b1,	zeros((dst.shape[0],3,3),	np.uint8),	merge1,		base1)),
											 		 hstack((r2,	g2,	b2,	zeros((dst.shape[0],3,3),	np.uint8),	merge2,		base2)),
											 	 	 hstack((r3,	g3,	b3,	zeros((dst.shape[0],3,3),	np.uint8),	merge3,		base3)),
											 	 	 hstack((r4,	g4,	b4,	zeros((dst.shape[0],3,3),	np.uint8),	merge4,		base4)),
											 	 	 hstack((r5,	g5,	b5,	zeros((dst.shape[0],3,3),	np.uint8),	merge5,		base5)))))

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
