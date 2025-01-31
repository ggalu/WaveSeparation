'''
Created on Jun 20, 2017

@author: gcg
'''
import numpy as np
import os, sys
from scipy import misc, signal, stats
import glob, pickle
import copy

from PyQt5 import QtWidgets  

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
from trackROI_chiSquare import TrackROI
import skimage

# only edit lines below:
path = "/home/gcg/Projekte/21_WaveSeparation/2025-01-30_Waveseparation/02_PC"; fac=1
smooth_sigma = 3.0
flipX = False # flip horizontal axis



params = [{'name': 'px2mm', 'type': 'float', 'value':  fac, 'step': 1.0e-3, 'siPrefix': False, 'suffix': 'mm/px'},
          {'name': 'scan rate', 'type': 'float', 'value': 200, 'step': 1, 'suffix': ' kHz', 'siPrefix': False},
          {'name': 'width ROI 1 in px', 'type': 'int', 'value': 1, 'suffix': ' px', 'siPrefix': False},
          {'name': 'width ROI 1 in mm', 'type': 'float', 'value': 1, 'suffix': ' mm', 'siPrefix': False},
          {'name': 'velocity ROI 1', 'type': 'float', 'value': 0, 'suffix': ' m/s', 'siPrefix': False},
          {'name': 'velocity ROI 2', 'type': 'float', 'value': 0, 'suffix': ' m/s', 'siPrefix': False},
          {'name': 'width ROI 2 in px', 'type': 'int', 'value': 1, 'suffix': ' px'},
          {'name': 'width ROI 2 in mm', 'type': 'int', 'value': 1, 'suffix': ' mm'},
          {'name': "distance ROI1 -- ROI2", 'type': 'float', 'value': 0, 'suffix': ' mm'},
          {'name': "ROI1 start", 'type': 'int', 'value': 485, 'suffix': ' px'},
          {'name': "ROI1 stop", 'type': 'int', 'value': 1740, 'suffix': ' px'},
          {'name': "ROI2 start", 'type': 'int', 'value': 3170, 'suffix': ' px'},
          {'name': "ROI2 stop", 'type': 'int', 'value': 3240, 'suffix': ' px'},
          {'name': "smoothing sigma", 'type': 'float', 'value': smooth_sigma, 'suffix': ' px'},
          ]


def read_linescan_data(path):
    """ read one image which contains multiple lines.
    Each line belongs to a different state of time."""
    
    os.chdir(path)
    files = glob.glob("*.bmp") + glob.glob("*.tif")
    print(files)
    if not (len(files) == 1):
        print("There are have multiple .bmp files in the working directory.")
        print("This program expects only a single .bmp image file containing linescan data.")
        sys.exit(1)
    
    print(files[0])
    
    #import imageio
    from matplotlib.pyplot import imread
    image_data = imread(os.path.join(path,files[0]))
    image_data = skimage.filters.gaussian(image_data, sigma=(0, smooth_sigma))

    if flipX:
        image_data = np.flip(image_data, axis=1)
    #, flatten=1
    print(np.shape(image_data))
    
    return image_data
    # return image_data[9500:15000:,:]
    
    # average every two subsequent lines
#    nx, ny = np.shape(image_data)
#    new_data = np.zeros((int(nx/2), ny))
#    for i in xrange(nx/2 - 1):
#        j = 2*i
#        k = j + 1
#        new_data[i,:] = 0.5 * (image_data[j,:] + image_data[k,:])
#        
#    print("*** AVERAGED EVERY TWO LINES ***")
    
    
    #return new_data

def updatePlot():
    global x
    print("---------------- in updatePlot() -----------------------")
    p["ROI1 start"], p["ROI1 stop"] = ROI1.getRegion()
    p["width ROI 1 in px"] = p["ROI1 stop"] - p["ROI1 start"]
    p["width ROI 1 in mm"] = p["width ROI 1 in px"] * p["px2mm"]
    
    p["ROI2 start"], p["ROI2 stop"] = ROI2.getRegion()
    p["width ROI 2 in px"] = p["ROI2 stop"] - p["ROI2 start"]
    p["width ROI 2 in mm"] = p["width ROI 2 in px"] * p["px2mm"]
    
    p["distance ROI1 -- ROI2"] = (np.mean(ROI2.getRegion()) - np.mean(ROI1.getRegion())) * p["px2mm"]
    print("---------------- end updatePlot() -----------------------")
    

def updateVelocity():
    """ Called whenever the velocity range region selector is changed
    """
    xmin, xmax = velRange1.getRegion()
    x, y = curve1.getData()
    indices = np.where(np.logical_and(x > xmin, x < xmax))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[indices],y[indices])
    print("linear velocity of ROI1 in selected range is ", slope)
    p["velocity ROI 1"] = slope
    curve_LinearVelocity.setData((xmin, xmax), (intercept + slope * xmin, intercept + slope * xmax))
    x, y = curve2.getData()
    indices = np.where(np.logical_and(x > xmin, x < xmax))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[indices],y[indices])
    print("linear velocity of ROI2 in selected range is ", slope)
    p["velocity ROI 2"] = slope
    #plotTrack1.clear()
    #plotTrack1.plot((xmin, xmax), (intercept + slope * xmin, intercept + slope * xmax))
    
    
    
def makeLayout():
    ## Create a grid layout to manage the widgets size and position
    #layout = QtGui.QGridLayout()
    layoutH0 = QtWidgets.QHBoxLayout()
    w.setLayout(layoutH0)
    
    layoutVL = QtWidgets.QVBoxLayout()
    layoutH0.addLayout(layoutVL)
    
    layoutVR = QtWidgets.QVBoxLayout()
    layoutH0.addLayout(layoutVR)
    
    layoutVL.addWidget(t)   # parameter tree
    layoutVL.addWidget(btnROI1)   # button goes in upper-left
    layoutVL.addWidget(btnROI2)   # button goes in upper-left
    layoutVL.addWidget(btnSave)
    layoutVL.addWidget(btnSaveConfig)
    layoutVL.addWidget(btnRestoreConfig)
    #layout.addWidget(listw)  # list widget goes in bottom-left
    layoutVR.addWidget(plotAll)
    layoutVR.addWidget(plot)  # plot goes on right side, spanning 3 rows
    layoutVR.addWidget(plotTrack1)  # plot goes on right side, spanning 3 rows
    layoutVR.addWidget(plotTrack2)  # plot goes on right side, spanning 3 rows
    

def clickedBtnROI1(btnROI1):
    print("button clicked!", ROI1.getRegion())
    with pg.BusyCursor():
        global t1
        t1 = TrackROI(data, int(ROI1.getRegion()[0]), int(ROI1.getRegion()[1]))
        global curve1
        curve1.setData(x, t1.displacements * p["px2mm"])
        
        
        # --- indicate the path of displaced ROI1 on image ---
        #global checkROI1
        x0 = np.mean(ROI1.getRegion())
        checkROI1.clear()
        #checkROI1 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))

        spots = []
        for i in range(numFrames):
            spot = {'pos': (x0 + t1.displacements[i], i)}
            spots.append(spot)
        
        checkROI1.addPoints(spots)
        #plotAll.addItem(checkROI1)
                
        
        
        # also update difference of ROI1 and ROI2        
        x1, y1 = curve1.getData()
        x2, y2 = curve2.getData()
        

        #curve3.setData(x, t1.subPixelShifts)
        
def clickedBtnROI2(btnROI2):
    print("button 2 clicked!", ROI2.getRegion())
    #from trackROI import TrackROI
    with pg.BusyCursor():
        t2 = TrackROI(data, int(ROI2.getRegion()[0]), int(ROI2.getRegion()[1]))
        global curve2
        curve2.setData(x, t2.displacements * p["px2mm"])

        # also update difference of ROI1 and ROI2        
        x1, y1 = curve1.getData()
        x2, y2 = curve2.getData()
        curve3.setData(x, (y2 - y1))
        
        # --- indicate the path of displaced ROI1 on image ---
        #global checkROI2
        x0 = np.mean(ROI2.getRegion())
        checkROI2.clear()
        #checkROI2 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 120))

        spots = []
        for i in range(numFrames):
            spot = {'pos': (x0 + t2.displacements[i], i)}
            spots.append(spot)
        
        checkROI2.addPoints(spots)
        #plotAll.addItem(checkROI2)
        

def saveToFile(btn):
    x1, y1 = curve1.getData() # displacement of ROI1
    x2, y2 = curve2.getData() # displacement of ROI2
    x3, y3 = curve3.getData() # relative displacement
    
    #print x
    
    L0 = p["distance ROI1 -- ROI2"]
    print("Saving with L0 = ", L0)
    strain = (y2 - y1) / L0  
    
    header = "linescan frequency = %f kHz\npixel length = %f mm\ngauge length = %f\ntime [msec], ROI1 displacement [mm], ROI2 displacement [mm], relative displacement [mm],  strain [-]" % (p["scan rate"], p["px2mm"], L0*p["px2mm"])
    np.savetxt(os.path.join(path, "linescan_analysis.dat"), np.column_stack((x, t1.displacements)), header=header)
    print("wrote output file linescan_analysis.dat to directory [%s]" % (path))
    
def saveState():
    state = p.saveState()
    outfile = os.path.join(path, "configuration.pkl")
    output = open(outfile, 'wb')
    pickle.dump(state, output)
    output.close()
    print("saved configuration state to file %s" % (outfile))
    
def loadState():
    #global p
    global ROI1
    infile = os.path.join(path, "configuration.pkl")
    pkl_file = open(infile, 'rb')
    state = pickle.load(pkl_file)
    p.restoreState(state) #, addChildren=False, removeChildren=False)
    pkl_file.close()
    print("--------------------------------------------------")
    print("restored configuration state from file %s" % (infile))
    print(p["ROI1 start"])
    ROI1.setRegion([p["ROI1 start"], p["ROI1 stop"]])
    ROI2.setRegion([p["ROI2 start"], p["ROI2 stop"]])
    print("restored RO1: ", ROI1.getRegion())
    print("restored RO2: ", ROI2.getRegion())
    print("--------------------------------------------------")
    updatePlot()
    
    
#def restore():
#    global state
#    p.restoreState(state, addChildren=add) 
    
def change(param, changes):
    """ called whenever a value in the parameter tree is changed
    """
    global x
    x =  np.arange(N) * (1.0 / p["scan rate"])
    
    #updatePlot() # ... does not seem to work
    
    for param, change, data in changes:
        path = p.childPath(param)
        if path is not None:
            childName = '.'.join(path)
        else:
            childName = param.name()
        print('*  path: %s' % path)
        print('*  parameter: %s' % childName)
        print('*  change:    %s' % change)
        print('*  data:      %s' % str(data))
        print('  ----------')
        
        if childName == "Save Stress/Strain":
            writeData()
    
def show_exception_and_exit(exc_type, exc_value, tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, tb)
    raw_input("Press key to exit.")
    sys.exit(-1)

import sys
sys.excepthook = show_exception_and_exit
    

data = read_linescan_data(path)
numFrames = len(data)
frame0 = data[0,:]
numPix = len(frame0)



frame0_extended = np.zeros((numPix,2))
for i in range(2):
    frame0_extended[:,i] = frame0
img = pg.ImageItem(frame0_extended)

## Always start by initializing Qt (only once per application)
#app = QtGui.QApplication([])
app = QtWidgets.QApplication([])

## Define a top-level widget to hold everything
#w = QtGui.QWidget()
w = QtWidgets.QWidget() # correct 2024-01-27
#frm = QtGui.QFrame ()
#win = QtGui.QMainWindow ()
#win.setCentralWidget (w)
w.resize(1200, 800)

p = Parameter.create(name='params', type='group', children=params)
p.sigTreeStateChanged.connect(change)
t = ParameterTree()
t.setParameters(p, showTop=False)

## Create some widgets to be placed inside.QHBoxLayout()
btnROI1 = QtWidgets.QPushButton('track ROI 1')
btnROI1.clicked.connect(clickedBtnROI1)
btnROI2 = QtWidgets.QPushButton('track ROI 2')
btnROI2.clicked.connect(clickedBtnROI2)

btnSave = QtWidgets.QPushButton('save results to file')
btnSave.clicked.connect(saveToFile)

btnSaveConfig = QtWidgets.QPushButton('save config to file')
btnSaveConfig.clicked.connect(saveState)

btnRestoreConfig = QtWidgets.QPushButton('load config from file')
btnRestoreConfig.clicked.connect(loadState)

plotAll = pg.PlotWidget(title="all linescan frames")
imgAll = pg.ImageItem(np.transpose(data))
checkROI1 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(255,0,0), brush=pg.mkBrush(200, 255, 0, 120))
checkROI2 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))
plotAll.addItem(imgAll)
plotAll.addItem(checkROI1)
plotAll.addItem(checkROI2)
plotAll.setLabel('left', "frame no.")
plotAll.setLabel('bottom', "pixel no.")

plot = pg.PlotWidget(title="Linescan frame at time 0. Select ROIs!")
plot.setLabel('left', "no unit")
plot.setLabel('bottom', "pixel no.")

ROI1 = pg.LinearRegionItem([p["ROI1 start"],p["ROI1 stop"]])
ROI2 = pg.LinearRegionItem([p["ROI2 start"],p["ROI2 stop"]])
ROI1.setZValue(10)
ROI2.setZValue(10)
plot.addItem(ROI1)
plot.addItem(ROI2)
plot.addItem(img)
ROI1.sigRegionChanged.connect(updatePlot)
ROI2.sigRegionChanged.connect(updatePlot)

# plot pixel displacements for each ROI
plotTrack1 = pg.PlotWidget(title="displacements of ROI1 and ROI2")
N = numFrames
x = np.arange(N) * (1.0 / p["scan rate"])
y = np.zeros(N)
curve1 = plotTrack1.plot(x, y, pen=(0,255,0), name="displacement ROI1")
curve_LinearVelocity = plotTrack1.plot(x, y, pen=(255,255,255), name="velocity curve")
curve2 = plotTrack1.plot(x, y, pen=(255,255,0), name="displacement ROI2")
plotTrack1.setLabel('left', "displacement", units='mm')
plotTrack1.setLabel('bottom', "time", units='ms')
velRange1 = pg.LinearRegionItem([0.4,2.0])
plotTrack1.addItem(velRange1)
velRange1.sigRegionChanged.connect(updateVelocity)

# plot relatiev displacmenst between ROI1 and ROI2
plotTrack2 = pg.PlotWidget(title="relative displacement")
curve3 = plotTrack2.plot(x, y, pen=(255,0,255), name="Red curve")
plotTrack2.setLabel('left', "displacement", units='mm')
plotTrack2.setLabel('bottom', "time", units='ms')


makeLayout()
w.show()
updatePlot()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()


