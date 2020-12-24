from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QPixmap,QImage
import sys
import cv2

class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('layout.ui', self) # Load the .ui file
        box1 = ['Default','Brightness/Contrast Boost','Gamma Boost','Normalize','Double Inverted Gamma Correction (DGC)', 'DGC Normalize']
        box2 = ['Base','Red','Green','Blue','Remerged']
        self.filterBox.addItems(box1)
        self.cameraBox.addItems(box2)

        self.camera = cv2.VideoCapture(0)
        self.initUI()

        self.show() # Show the GUI

    def initUI(self):
        self.updateSelf()

        self.my_timer = QtCore.QTimer()
        self.my_timer.timeout.connect(self.updateSelf)
        self.my_timer.start(1) #1 min intervall


    def updateSelf(self):
        ret_val, img = self.camera.read()

        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_BGR888)
        map = QPixmap(qImg)

        self.display.setPixmap(map)

        self.update()




app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the application
