from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QPixmap,QImage
import sys
import cv2
import numpy as np
from layout import Ui_Dialog

class Ui(QtWidgets.QDialog, Ui_Dialog):
	def __init__(self, parent=None):
		super(Ui, self).__init__(parent) # Call the inherited classes __init__ method
		self.setupUi(self)
		#uic.loadUi('layout.ui', self) # Load the .ui file

		self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
		self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)

		self.filbox = ['Default','Brightness/Contrast Boost','Gamma Boost','Normalize','Double Inverted Gamma Correction (DGC)', 'DGC Normalize']
		self.cambox = ['Base','Red','Green','Blue','Remerged']
		self.filterBox.addItems(self.filbox)
		self.cameraBox.addItems(self.cambox)

		self.alpha = float(self.contrastLine.text())
		self.beta = float(self.brightnessLine.text())
		self.gamma = float(self.gammaLine.text())

		self.gate = float(self.gateLine.text())
		self.i = 0

		self.samples = []

		self.source = self.sourceLine.text()
		if(self.source.isnumeric()):
			self.source = int(self.source)
		self.camera = cv2.VideoCapture(self.source)

		self.initUI()

		self.show() # Show the GUI

	def initUI(self):
		self.updateSelf()

		self.my_timer = QtCore.QTimer()
		self.my_timer.timeout.connect(self.updateSelf)
		self.my_timer.start(1e3/30)

		self.sourceButton.clicked.connect(self.swapCam)

	def swapCam(self):
		oldcam = self.camera

		self.source = self.sourceLine.text()
		if(self.source.isnumeric()):
			self.source = int(self.source)

		self.camera=cv2.VideoCapture(self.source)
		retval = self.camera.read()[0]

		if not retval:
			self.camera = oldcam
		else:
			self.samples = []

	def isNumber(self,string):
	    try:
	        float(string)
	        return True
	    except ValueError:
	        return False

	#adjusts the gamma of an image
	def adjust_gamma(self,image):
		invGamma = 1.0 / self.gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
		return cv2.LUT(image, table)

	#stacks a series of images to reduce noise
	def stack(self):
		dst = self.samples[0]
		for j in range(len(self.samples)):
			if j == 0:
				pass
			else:
				alpha = 1.0/(j + 1)
				beta = 1.0 - alpha
				dst = cv2.addWeighted(self.samples[j], alpha, dst, beta, 0.0)

		return dst

	#adjusts the brightness and contrast of an image using alpha/beta values
	def brightContrast(self,img):
		return cv2.addWeighted(img,self.alpha,np.zeros(img.shape, img.dtype),0,self.beta)

	#normalizes the pixel values to range over 0-255
	def normalize(self,img):
		return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

	#inverts, adjusts gamma, and reinverts an image
	def doubleGC(self,img):
		return cv2.bitwise_not(self.adjust_gamma(cv2.bitwise_not(img)))

	'''splits an image into color channels, applies filters to each channel and base,
	remerges channels and converts colorspaces of each channel, returning each
	variation'''
	def splitFilter(self, img, actions = None,):
		base  = img.copy()
		b,g,r = cv2.split(img)
		if(actions != None):
			for action in actions:
				base = action(base)
				b = action(b)
				g = action(g)
				r = action(r)

		m = cv2.merge([b,g,r])

		b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
		g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
		r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

		return base,b,g,r,m

	def updateSelf(self):
		ret_val, frame = self.camera.read()
		if(ret_val):
			if(len(self.samples)<self.gate):
				self.samples.append(frame)
			else:
				self.samples[int(self.i)] = frame
			self.i=(self.i+1)%self.gate

		imgstack = self.stack()

		filInd = self.filterBox.currentIndex()
		camInd = self.cameraBox.currentIndex()

		base,b,g,r,m = self.splitFilter(imgstack)


		if (filInd == 1): #bc boost
			base,b,g,r,m = self.splitFilter(imgstack,actions = [self.brightContrast])
		elif (filInd == 2): #gamma boost
			base,b,g,r,m = self.splitFilter(imgstack,actions = [self.adjust_gamma])
		elif (filInd == 3): #normalize
			base,b,g,r,m = self.splitFilter(imgstack,actions = [self.normalize])
		elif (filInd == 4): #DGC
			base,b,g,r,m = self.splitFilter(imgstack,actions = [self.doubleGC])
		elif (filInd == 5): #DGC Norm.
			base,b,g,r,m = self.splitFilter(imgstack,actions = [self.doubleGC,self.normalize])

		if (camInd == 0):
			img = base
		elif (camInd == 1): #red
			img = r
		elif (camInd == 2): #green
			img = g
		elif (camInd == 3): #blue
			img = b
		elif (camInd == 4): #remerged
			img = m


		height, width, channel = img.shape
		bytesPerLine = 3 * width
		qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_BGR888)
		map = QPixmap(qImg).scaled(self.display.size(), QtCore.Qt.KeepAspectRatio,
                                          		   		QtCore.Qt.SmoothTransformation)

		self.display.setPixmap(map)


		if(self.isNumber(self.contrastLine.text())):
			self.alpha = float(self.contrastLine.text())
		if(self.isNumber(self.brightnessLine.text())):
			self.beta = float(self.brightnessLine.text())
		if(self.isNumber(self.gammaLine.text())):
			if(float(self.gammaLine.text())!=0):
				self.gamma = float(self.gammaLine.text())
		if(self.isNumber(self.gateLine.text())):
			if(int(self.gateLine.text())>0):
				if(self.gate!=int(self.gateLine.text())):
					self.samples = []
				self.gate = int(self.gateLine.text())

		self.update()




app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the application
