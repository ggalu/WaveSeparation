# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-27 17:50:34
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-27 18:26:42
import numpy as np
import copy
from skimage.feature import match_template
import skimage

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
        self.ROI = self.refFrame
        self.ROI[:self.ROI_start] = 0.0
        self.ROI[self.ROI_stop:] = 0.0
        
        # create arrays to save data calculated by this program
        self.displacements = np.zeros(self.numFrames)
        self.optshift = 0
        
        
        template = self.refFrame[self.ROI_start:self.ROI_stop]
        lenTemplateArea = self.ROI_stop - self.ROI_start
        halo = 20
        newShift = 0
        rightEnd = leftEnd = False
        
        # iterate over all frames and find the shift of the reference image (template) to the current image.
        # for efficiency, do not consider the full current image, but only a range centerad around the
        # shift determind during the last iteration.
        #
        # in a second step, use a subpixel-accurate method to improve the shift
        for i in range(self.numFrames):
            
            # dont allow indices to leave image area
            shiftedStart = max(0, self.ROI_start + newShift - halo)
            shiftedStop = min(4095, self.ROI_stop + newShift + halo)
            
            if shiftedStop == 4095:
                rightEnd = True
            else:
                rightEnd = False
            
            if shiftedStart == 0:
                leftEnd = True
            else: leftEnd = False
            
            print( "----")
            print( "rightEnd", rightEnd)
            print( "leftEnd", leftEnd)
            print( "template length is           %d " % (self.ROI_stop-self.ROI_start))
            print( "search from %d to %d, length %d " % (shiftedStart, shiftedStop, shiftedStop - shiftedStart))
            
            currentFrame = self.yield_frame(i)[shiftedStart:shiftedStop]
            if (len(currentFrame) < len(template) +1):
                self.displacements[i] = newShift
            else:
                
                
                # First get pixel-accurate estimate of where template is within current frames
               
                
                # need to exit if 
                
                res = match_template(currentFrame, template, pad_input=True)[:,1]
                relative_shift = np.argmax(res) - halo - int(0.5 * len(template)) # this is relative to the already shifted current image
                newShift += relative_shift
                print( "frame %d, displacement is %d, match score is %f" % (i, newShift, np.max(res)))
                
                # now perform subpixel detection:
                subPixelShift = 0.0
                if self.subpixel_refinement > 1:
                    # need to template and current image to have exactly the same size, so
                    # do not include halo like above.
                    shiftedStart = self.ROI_start + newShift
                    shiftedStop = self.ROI_stop + newShift
                    
                    # cannot do subpixel alignment if search region leaves image area
                    if shiftedStart >= 0 and shiftedStop <= 4095:
                        print( "subpixel: shiftedStart = ", shiftedStart)
                        print( "subpixel: shiftedStop = ", shiftedStop)
                    
                        #assert (shiftedStart >= 0)
                        #assert (shiftedStop <= 4095)
                        currentFrame = self.yield_frame(i)[shiftedStart:shiftedStop]

                        subPixelShift, error, diffphase = skimage.registration.phase_cross_correlation(template, currentFrame, upsample_factor=self.subpixel_refinement)
                        #subPixelShift = register_translation(currentFrame, template,upsample_factor=self.subpixel_refinement)[0][0]
                        print( "sub-Pixel shift:", subPixelShift)
                
                self.displacements[i] = newShift + subPixelShift[0]
            
        
       
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
