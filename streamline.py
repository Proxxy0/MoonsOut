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

def splitFilter(b,g,r, actions = None, desc = None):
	bs = "Blue"
	gs = "Green"
	rs = "Red"
	ms = "Merged"
	if(actions != None):
		for action in actions:
			b = action(b)
			g = action(g)
			r = action(r)

		bs+=" | "+desc
		gs+=" | "+desc
		rs+=" | "+desc
		ms+=" | "+desc

	m = cv2.merge([b,g,r])


	b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
	g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
	r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

	b = putText(b,bs)
	g = putText(g,gs)
	r = putText(r,rs)
	m = putText(m,ms)

	return b,g,r,m

def filter(img, actions = None, desc = None):
	s = "Base"
	if(actions != None):
		for action in actions:
			img = action(img)

		s+=" | "+desc

	img = putText(img,s)

	return img



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
		b2,g2,r2,rm2 = splitFilter(b,g,r, actions = [brightContrast], desc = "Cont. boost")
		b3,g3,r3,rm3 = splitFilter(b,g,r, actions = [normalize], desc = "Norm.")
		b4,g4,r4,rm4 = splitFilter(b,g,r, actions = [doubleGC], desc = "DGC")
		b5,g5,r5,rm5 = splitFilter(b,g,r, actions = [doubleGC,normalize], desc = "DGC Norm.")

		#remerge channels and apply transforms to base
		rm = cv2.merge([b,g,r])
		dst2 = filter(dst, actions = [brightContrast], desc = "Cont. boost")
		dst3 = filter(dst, actions = [normalize], desc = "Norm.")
		dst4 = filter(dst, actions = [doubleGC], desc = "DGC")
		dst5 = filter(dst, actions = [doubleGC,normalize], desc = "DGC Norm.")

		#caption the feeds
		b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
		g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
		r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
		r=putText(r,'Red'); g=putText(g,'Green'); b=putText(b,'Blue')
		rm=putText(rm,"RGB Merged")
		dst=putText(dst,"Base")



		#create grid of images from split channels and show it
		cv2.imshow("MoonsOut | Dark Vision", vstack((hstack((r,		g,	b,	zeros((dst.shape[0],3,3),	np.uint8),	rm,		dst)),
											 		 hstack((r2,	g2,	b2,	zeros((dst2.shape[0],3,3),	np.uint8),	rm2,	dst2)),
											 	 	 hstack((r3,	g3,	b3,	zeros((dst3.shape[0],3,3),	np.uint8),	rm3,	dst3)),
											 	 	 hstack((r4,	g4,	b4,	zeros((dst4.shape[0],3,3),	np.uint8),	rm4,	dst4)),
											 	 	 hstack((r5,	g5,	b5,	zeros((dst5.shape[0],3,3),	np.uint8),	rm5,	dst5)))))

		# #create grid of images from remerged and base images and show it
		# cv2.imshow("Merged comparison", vstack((hstack((rm,		dst)),
		# 									 	hstack((rm2,	dst2)),
		# 									 	hstack((rm3,	dst3)),
		# 									 	hstack((rm4,	dst4)),
		# 									 	hstack((rm5,	dst5)))))

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
