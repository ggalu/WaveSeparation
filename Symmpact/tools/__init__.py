import numpy as np
from scipy import stats

def find_TTL(y, direction="positive", level=2.5):
    """
    find a TTL signal, retrun index in y
    """
    for i in range(len(y)):
        if y[i] > level:
            return i
    return -1

def zeroSignal(y, range):
    maxIdx = int(len(y) * range)
    average = np.mean(y[:maxIdx])
    y -= average
    
def SelectRange(data, start, stop):
    """
    slice a time subset of data
    """
    time = data[:,0]
    indices = np.where(np.logical_and(time > start, time < stop))
    data_subset = data[indices,:]
    print(np.shape(data_subset))
    return data_subset[0]
    
def Slope(xin, yin, zero_intercept=True, plot=False, label=""):
    """
    Given input data vectors xin, yin,
    perform linear regression.
    Plot data versus linear function.
    Return the slope.
    """
    x = xin
    y = yin
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #print "slope = ", slope
    #print "intercept = ", intercept
    if zero_intercept:
        y -= intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        #print "slope after zeroing intercept = ", slope
        #print "intercept, should be zero = ", intercept
    
    
    
    if plot:
        import pylab as plt
        plt.clf()
        plt.plot(x, y, label=label)
        plt.xlabel("KMZ voltage")
        plt.ylabel("DMS voltage")
        x0 = x[0]
        x1 = np.max(x)
        y0 = intercept + slope * x0
        y1 = intercept + slope * x1
        plt.plot((x0,x1), (y0,y1), "r-")
        plt.legend()
        plt.show()
        
    return slope


import numpy as n
import scipy.interpolate
import scipy.ndimage

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [n.float64, n.float32]:
        a = n.cast[float](a)

    m1 = n.cast[int](minusone)
    ofs = n.cast[int](centre) * 0.5
    old = n.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print( "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = n.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = n.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = n.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = n.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [n.arange(i, dtype = n.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = n.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = n.mgrid[nslices]

        newcoords_dims = range(n.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (n.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print( "Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None