# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-27 17:50:34
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-28 13:55:15
import numpy as np
import copy
import pylab as plt
import itk
import sys


class TrackROI():
    def __init__(self, image_data, ROI_start, ROI_stop):
        
        """
        track the displacement of a region of interest (ROI) over several frames 
        """
        
        self.ROI_start = ROI_start
        self.ROI_stop = ROI_stop
        
        # load files
        self.allFrames = copy.deepcopy(image_data)
        self.numFrames = np.shape(image_data)[0]
        #self.numFrames = 100
        print("number of frames in image: ", self.numFrames)
        
        # load first frame and define ROI
        self.refFrame = self.yield_frame(0)
        self.refFrame = itk.image_view_from_array(self.refFrame)

        #self.numPixels = len(self.refFrame)
        
        # single out ROI by setting everything else in the first frame to zero
        self.ROI = np.zeros(self.refFrame.shape, dtype=np.uint8)
        print("ROI shape:", self.ROI.shape)
        #sys.exit()

        self.ROI[:,self.ROI_start:self.ROI_stop] = 1
        mask = itk.image_view_from_array(self.ROI)
        #print("data type:", self.mask.dtype)
        region = mask.GetLargestPossibleRegion()
        size = region.GetSize()
        print(f"ITK image data size after conversion from NumPy = {size}\n")

        #sys.exit()
        
        # create arrays to save data calculated by this program
        self.displacements = np.zeros(self.numFrames)

        par_obj = itk.ParameterObject.New()
        par_map = par_obj.GetDefaultParameterMap('rigid')
        par_obj.AddParameterMap(par_map)
        
        # iterate over all frames and find the shift of the reference image (template) to the current image.
        # for efficiency, do not consider the full current image, but only a range centerad around the
        # shift determind during the last iteration.
        #
        # in a second step, use a subpixel-accurate method to improve the shift
        for i in range(self.numFrames):
            
            currentFrame = self.yield_frame(i)
            currentFrame = itk.image_view_from_array(currentFrame)

            #itk.LogToConsoleOn()

            result_image, rtp = itk.elastix_registration_method(
                           self.refFrame,
                           currentFrame,
                           fixed_mask=mask,
                           moving_mask=mask,
                           parameter_object=par_obj)
            
            
            
            
            parameter_map = rtp.GetParameterMap(0)
            transform_parameters = np.array(parameter_map['TransformParameters'], dtype=float)

            print("transform", transform_parameters)
            
            self.displacements[i] = transform_parameters[1]

        plt.plot(self.displacements)
        plt.show()

    
       
    def yield_frame(self, i):
        """ read frame number i """
        
        image = self.allFrames[i,:]
        
        # duplicate row to have a 2d image as required for SciKit-Image
        shape = np.shape(image)
        nTile = 128
        image2d = np.tile(image, (nTile, 1))
        
        #plt.imshow(image2d)
        #plt.show()
        
        return image2d
        #return signal.detrend(image)


if __name__ == "__main__":
    filename = "/home/gcg/Projekte/21_WaveSeparation/2024-01-27_Waveseparation/01/01.bmp"
    from matplotlib.pyplot import imread
    image_data = imread(filename)

    #plt.imshow(image_data)
    #plt.show()


    image_data = image_data[2300:2400,:].astype(np.float32)


    tracker = TrackROI(image_data, 485, 1740)
    

