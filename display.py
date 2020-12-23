from PyQt5 import QtWidgets, uic
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
        self.show() # Show the GUI

        def setData(self, **index**, value):




app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the application
