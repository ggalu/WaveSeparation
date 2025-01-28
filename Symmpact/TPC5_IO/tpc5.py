import numpy as np

''' ******************************************************************************************** '''
''' Python TPC5 Helper Modules for reading HDF5 generated by TranAX Application Software         '''
''' Copyright 2017 Elsys AG      '''
''' ******************************************************************************************** '''


''' Get Raw Dataset Block'''
def getDataSetName(channel, block = 1):
    blockString   = "%08d" % block
    channelString = "%08d" % channel
    name = '/measurements/00000001/channels/' + channelString + '/blocks/' + blockString + '/raw'
    return name

def getChannelGroupName(channel):
    channelString = "%08d" % channel   
    return '/measurements/00000001/channels/' + channelString +'/'

def getBlockName(channel, block):
    blockString   = "%08d" % block
    channelString = "%08d" % channel
    name = '/measurements/00000001/channels/' + channelString + '/blocks/' + blockString + '/'
    return name

def getVoltageData(fileRef, channel, block = 1):
    channel_group           = fileRef[getChannelGroupName(channel)]
    dataset_name            = getDataSetName(channel,block)

    ''' Get Scaling Parameters '''
    binToVoltageFactor      = channel_group.attrs['binToVoltFactor']
    binToVoltageConstant    = channel_group.attrs['binToVoltConstant']

    ''' Get Analog and Digital Mask for Data separation '''
    analogMask              = channel_group.attrs['analogMask']
    markerMask              = channel_group.attrs['markerMask']
    #print "analog mask", analogMask
    #print "marker mask", markerMask

    analogData              = fileRef[dataset_name] & analogMask
    markerData              = fileRef[dataset_name] & markerMask # GCG 24Jun2018
    
    #print "shape of analog data", np.shape(analogData)
    #print "shape of marker data", np.shape(markerData)
    
    ''' Scale To voltage '''
    analogVoltage = analogData * binToVoltageFactor + binToVoltageConstant
    
    return analogVoltage, markerData

def getPhysicalData(fileRef, channel, block = 1):
    channel_group           = fileRef[getChannelGroupName(channel)]
    dataset_name            = getDataSetName(channel,block)

    ''' Get Scaling Parameters '''
    binToVoltageFactor      = channel_group.attrs['binToVoltFactor']
    binToVoltageConstant    = channel_group.attrs['binToVoltConstant']
    VoltToPhysicalFactor    = channel_group.attrs['voltToPhysicalFactor']
    VoltToPhysicalConstant  = channel_group.attrs['voltToPhysicalConstant']

    ''' Get Analog and Digital Mask for Data separation '''
    analogMask              = channel_group.attrs['analogMask']
    markerMask              = channel_group.attrs['markerMask']
    
    analogData              = fileRef[dataset_name] & analogMask
    
    ''' Scale To voltage '''
    voltageData = analogData * binToVoltageFactor + binToVoltageConstant
    return voltageData * VoltToPhysicalFactor + VoltToPhysicalConstant

def getChannelName(fileRef, channel):
    channel_group           = fileRef[getChannelGroupName(channel)]
    return channel_group.attrs['name']

def getPhysicalUnit(fileRef, channel):
    channel_group           = fileRef[getChannelGroupName(channel)]
    return  channel_group.attrs['physicalUnit']

def getSampleRate(fileRef, channel, block = 1):
    block_group             = fileRef[getBlockName(channel,block)]
    return block_group.attrs['sampleRateHertz']

def getTriggerSample(fileRef, channel, block = 1):
    block_group             = fileRef[getBlockName(channel,block)]
    return block_group.attrs['triggerSample']

def getTriggerTime(fileRef, channel, block = 1):
    block_group             = fileRef[getBlockName(channel,block)]
    return block_group.attrs['triggerTimeSeconds']

def getStartTime(fileRef, channel, block = 1):
    block_group             = fileRef[getBlockName(channel,block)]
    return block_group.attrs['startTime']