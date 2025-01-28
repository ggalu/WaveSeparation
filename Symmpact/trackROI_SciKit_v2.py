# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-27 17:50:34
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-28 10:49:00
import numpy as np
import copy
from skimage.feature import match_template
import skimage
import pylab as plt

class TrackROI():
    def __init__(self, image_data, ROI_start, ROI_stop, subpixel_refinement):
        
        """
        track the displacement of a region of interest (ROI) over several frames 
        """
        
        self.ROI_start = ROI_start
        self.ROI_stop = ROI_stop
        self.subpixel_refinement = subpixel_refinement
        
        # load files
        self.allFrames = copy.deepcopy(image_data)
        self.numFrames = np.shape(image_data)[0]
        #self.numFrames = 100
        print("number of frames in image: ", self.numFrames)
        
        # load first frame and define ROI
        self.refFrame = self.yield_frame(0)
        self.numPixels = len(self.refFrame)
        
        # single out ROI by setting everything else in the first frame to zero
        self.ROI = np.zeros(self.refFrame.shape, dtype=np.bool)
        self.ROI[self.ROI_start:self.ROI_stop,:] = True
        
        # create arrays to save data calculated by this program
        self.displacements = np.zeros(self.numFrames)
        self.subPixelShifts = np.zeros(self.numFrames)
        self.optshift = 0
        
        # iterate over all frames and find the shift of the reference image (template) to the current image.
        # for efficiency, do not consider the full current image, but only a range centerad around the
        # shift determind during the last iteration.
        #
        # in a second step, use a subpixel-accurate method to improve the shift
        for i in range(self.numFrames):
            
            currentFrame = self.yield_frame(i)
            #subPixelShift, error, diffphase = skimage.registration.phase_cross_correlation(template, currentFrame, upsample_factor=self.subpixel_refinement)
            pixelShift_float, error, diffphase = skimage.registration.phase_cross_correlation(
                self.refFrame, currentFrame, reference_mask=self.ROI, upsample_factor=100, disambiguate=True)
            pixelShift = int(pixelShift_float[0])
            #print( "pixel shift:", pixelShift_float[0], pixelShift)

            self.displacements[i] = pixelShift

            # now do sub-pixel refinement.
            refFrame_windowed = self.refFrame[self.ROI_start:self.ROI_stop,:]
            curFrame_windowed =  currentFrame[self.ROI_start-pixelShift:self.ROI_stop-pixelShift,:]
            assert len(refFrame_windowed) == len(curFrame_windowed)
            subPixelShift_float, error, diffphase = skimage.registration.phase_cross_correlation(
                refFrame_windowed, curFrame_windowed, upsample_factor=100)
            #print("subpixel shift is", subPixelShift_float)

            print( "pixel shift:", pixelShift_float[0], pixelShift, subPixelShift_float[0])
            self.subPixelShifts[i] = subPixelShift_float[0]

            #plt.plot(subPixelShift_float[0])
            #plt.show()
            
        
       
    def yield_frame(self, i):
        """ read frame number i """
        
        image = self.allFrames[i,:]
        
        # duplicate row to have a 2d image as required for SciKit-Image
        shape = np.shape(image)
        image2d = np.zeros((shape[0], 2))
        image2d[:,0] = image
        image2d[:,1] = image
        
        return image2d
        #return signal.detrend(image)
