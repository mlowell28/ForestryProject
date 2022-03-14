# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:26:18 2022

@author: lowellm
"""

import os
import PySpin
import sys
import time

logfile = None

def configure_image_format(cam):
    
    result = True
    nodemap = cam.GetNodeMap()
    
    node_pixel_format_selector = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    if not PySpin.IsAvailable(node_pixel_format_selector) or not PySpin.IsWritable(node_pixel_format_selector):
        logfile.write('Unable to access PixelFormat node. Aborting...\n')
        return False
    
    node_pixel_format_RGB8 = node_pixel_format_selector.GetEntryByName('RGB8')
    if not PySpin.IsAvailable(node_pixel_format_RGB8) or not PySpin.IsReadable(node_pixel_format_RGB8):
        logfile.write('Unable to access RGB8 pixel format node(enum entry retrieval). Aborting...\n')
        return False
    
    node_pixel_format_selector.SetIntValue(node_pixel_format_RGB8.GetValue())
    return result

def configure_exposure(cam):
    
    result = True
    nodemap = cam.GetNodeMap()
    
    # set exposure mode to timed
    node_exposure_selector = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureMode'))
    if not PySpin.IsAvailable(node_exposure_selector) or not PySpin.IsWritable(node_exposure_selector):
        logfile.write('Unable to access Exposure_Selector node. Aborting...\n')
        return False
    node_timed_exposure_mode = node_exposure_selector.GetEntryByName("Timed")
    if not PySpin.IsAvailable(node_timed_exposure_mode) or not PySpin.IsReadable(node_timed_exposure_mode):
        logfile.write("Unable to access Time format node(enum entry retrival). Aborting...\n")
        
    node_exposure_selector.SetIntValue(node_timed_exposure_mode.GetValue())
    
    # set automatic exposure mode
    
    node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
    if not PySpin.IsAvailable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
        logfile.write('Unable to access Exposure_Auto node. Aborting...\n')
        return False
    
    node_exposure_auto_continuous_mode = node_exposure_auto.GetEntryByName("Continuous")
    if not PySpin.IsAvailable(node_exposure_auto_continuous_mode) or not PySpin.IsReadable(node_exposure_auto_continuous_mode):
        logfile.write("Unable to access exposure_auto_continuous node(enum entry retrival). Aborting...\n")
        
    node_exposure_auto.SetIntValue(node_exposure_auto_continuous_mode.GetValue())
    
    # set automatic exposure upper time limit
    
    node_auto_exposure_upper_time_limit = PySpin.CFloatPtr(nodemap.GetNode('AutoExposureTimeUpperLimit'))
    if not PySpin.IsAvailable( node_auto_exposure_upper_time_limit) or not PySpin.IsWritable(node_auto_exposure_upper_time_limit):
        logfile.write('Unable to access Auto_Exposure_Time_Upper_Limit node. Aborting...\n')
        return False
    
    # to 1000us
    node_auto_exposure_upper_time_limit.SetValue(7000)
    
    node_exposure_auto_continuous_mode = node_exposure_auto.GetEntryByName("Continuous")
    if not PySpin.IsAvailable(node_exposure_auto_continuous_mode) or not PySpin.IsReadable(node_exposure_auto_continuous_mode):
        logfile.write("Unable to access exposure_auto_continuous node(enum entry retrival). Aborting...\n")
        
    node_exposure_auto.SetIntValue(node_exposure_auto_continuous_mode.GetValue())
    
    return result
    
    
def configure_trigger(cam, triggerType="Hardware"):

    result = True

    logfile.write('*** CONFIGURING TRIGGER ***\n')
    logfile.write('Note that if the application / user software triggers faster than frame time, the trigger may be dropped / skipped by the camera.\n')
    logfile.write('If several frames are needed per trigger, a more reliable alternative for such case, is to use the multi-frame mode.\n\n')
    logfile.write('Software trigger chosen ...\n')


    try:
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        nodemap = cam.GetNodeMap()
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            logfile.write('Unable to disable trigger mode (node retrieval). Aborting...\n')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            logfile.write('Unable to disable trigger mode (enum entry retrieval). Aborting...\n')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        logfile.write('Trigger mode disabled...\n')
        
        # Set TriggerSelector to FrameStart
        # For this example, the trigger selector should be set to frame start.
        # This is the default for most cameras.
        
        node_trigger_selector= PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
        if not PySpin.IsAvailable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
            logfile.write('Unable to get trigger selector (node retrieval). Aborting...\n')
            return False

        node_trigger_selector_framestart = node_trigger_selector.GetEntryByName('FrameStart')
        if not PySpin.IsAvailable(node_trigger_selector_framestart) or not PySpin.IsReadable(
                node_trigger_selector_framestart):
            logfile.write('Unable to set trigger selector (enum entry retrieval). Aborting...\n')
            return False
        node_trigger_selector.SetIntValue(node_trigger_selector_framestart.GetValue())
        
        logfile.write('Trigger selector set to frame start...\n')

        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
        # mode is off.
        
        
        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
            logfile.write('Unable to get trigger source (node retrieval). Aborting...\n')
            return False

        if triggerType == "Software":
            node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
            if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                    node_trigger_source_software):
                logfile.write('Unable to set trigger source software (enum entry retrieval). Aborting...\n')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())
            logfile.write('Trigger source set to software...\n')
            
        if triggerType == "Hardware":
            node_trigger_source_software = node_trigger_source.GetEntryByName('Line0')
            if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                    node_trigger_source_software):
                logfile.write('Unable to set trigger source Hardware (enum entry retrieval). Aborting...\n')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())
            logfile.write('Trigger source set to Hardware...\n')

        
        
        # Turn trigger mode on
        # Once the appropriate trigger source has been set, turn trigger mode
        # on in order to retrieve images using the trigger.
        node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
        if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
            logfile.write('Unable to enable trigger mode (enum entry retrieval). Aborting...\n')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
        logfile.write('Trigger mode turned back on...\n')
           

    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s\n' % ex)
        return False

    return result


def configure_output_line(cam):
    
    result = True
        # configure output of cam to be 1 if triggered
    try:
        
        nodemap = cam.GetNodeMap()
        #set output to be triggered when external trigger is applied
        
        node_line_selector = PySpin.CEnumerationPtr(nodemap.GetNode('LineSelector'))
        if not PySpin.IsAvailable(node_line_selector) or not PySpin.IsWritable(node_line_selector):
            logfile.write('Unable to get line selector (node retrieval). Aborting...\n')
            return False
        
        node_line_selector_line1 = node_line_selector.GetEntryByName('Line1')
        if not PySpin.IsAvailable(node_line_selector_line1) or not PySpin.IsReadable(node_line_selector_line1):
            logfile.write('Unable to set line selector (enum entry retrieval). Aborting...\n')
            return False
        
        node_line_selector.SetIntValue(node_line_selector_line1.GetValue())
        logfile.write("output line selector set to Line1\n")
        
        node_line_source = PySpin.CEnumerationPtr(nodemap.GetNode('LineSource'))
        
        if not PySpin.IsAvailable(node_line_source) or not PySpin.IsWritable(node_line_source):
            logfile.write('Unable to get line source (node retrieval). Aborting...\n')
            return False
        
        node_line_source_external_trig = node_line_source.GetEntryByName("ExternalTriggerActive")
        
        if not PySpin.IsAvailable(node_line_source_external_trig ) or not PySpin.IsReadable(node_line_source_external_trig ):
            logfile.write('Unable to set line 1 source (enum entry retrieval). Aborting...\n')
            return False
        
        node_line_source.SetIntValue(node_line_source_external_trig.GetValue())
        logfile.write("output line 1 source set to external trigger\n")
        
    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s\n' % ex)
        return False

    return result
        
         
def Venable(cam, value):    
    
    result = True
    try:
        nodemap = cam.GetNodeMap()
        # set 3.3V output
                
        node_3V3_control = PySpin.CBooleanPtr(nodemap.GetNode('V3_3Enable'))
        if not PySpin.IsAvailable(node_3V3_control) or not PySpin.IsWritable(node_3V3_control):
            logfile.write('Unable to get 3.3V control (node retrieval). Aborting...\n')
            return False
        
        node_3V3_control.SetValue(value)
        
        if value == True:
            logfile.write("Set 3.3V output enabled\n")
        
        if value == False:
            logfile.write("Sev 3.3V output disabled\n")
                

    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s \n' % ex)
        return False

    return result
    

def grab_next_image_by_trigger(cam, user_input = False):
    
    nodemap = cam.GetNodeMap()
    
    try:
        result = True
        
        # Execute software trigger
        node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
        if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
            logfile.write('Unable to execute trigger. Aborting...\n')
            return False
        
        logfile.write("Camera Triggered, time: " + str(time.time_ns()) + "\n")
        node_softwaretrigger_cmd.Execute()

    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s \n' % ex)
        return False

    return result


def acquire_images_slave_camera(slave_cam, control_pipe, folder_name): # nodemap, nodemap_tldevice):
    

    slave_nodemap = slave_cam.GetNodeMap()
    
    logfile.write('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous
        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        slave_node_acquisition_mode = PySpin.CEnumerationPtr(slave_nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(slave_node_acquisition_mode) or not PySpin.IsWritable(slave_node_acquisition_mode):
            logfile.write('Unable to set Slave Camera acquisition mode to continuous (enum retrieval). Aborting...\n')
            return False

        # Retrieve entry node from enumeration node
        slave_node_acquisition_mode_continuous = slave_node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(slave_node_acquisition_mode_continuous) or not PySpin.IsReadable(
                slave_node_acquisition_mode_continuous):
            logfile.write('Unable to set Slave Camera acquisition mode to continuous (entry retrieval). Aborting...\n')
            return False

        # Retrieve integer value from entry node
        slave_acquisition_mode_continuous = slave_node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        slave_node_acquisition_mode.SetIntValue(slave_acquisition_mode_continuous)

        logfile.write('Slave camera acquisition mode set to continuous...\n')
        
        #  Begin acquiring images
        slave_cam.BeginAcquisition()
    
        logfile.write('Acquiring images...\n\n')
        
        slave_TLnodemap = slave_cam.GetTLDeviceNodeMap()
        
        slave_device_serial_number = ''
        slave_node_device_serial_number = PySpin.CStringPtr(slave_TLnodemap.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(slave_node_device_serial_number) and PySpin.IsReadable(slave_node_device_serial_number):
            slave_device_serial_number = slave_node_device_serial_number.GetValue()
            logfile.write('Slave cam serial number retrieved as %s...\n' % slave_device_serial_number)
            
        
        # image count 
        image_count = 0
        
        # starting state
        state = None
        
        # while state != exit, attempt to retrieve next image
        while state != "exit":
            
            command = control_pipe.recv()
            
            if command == "capture_image":
                logfile.write(str(time.time_ns()) +": Command received: capture_image\n")
                state = "capture_image"
            elif command == "exit":
                logfile.write(str(time.time_ns())+": Command received: exit, time\n")
                state = "exit" 
                
                    
            if command == "capture_image":
                    
                try:
                    start_time = time.time_ns() 
                    logfile.write(str(start_time) +": Retrieving frame:" + str(image_count) +"\n")
                    slave_image_result = slave_cam.GetNextImage(2000)
                    
                    #  Retrieve the next image 
                    image_retrived_time = time.time_ns()
                    logfile.write(str(image_retrived_time) + ": Image Retrieved, retrival duration:: " + str(image_retrived_time - start_time) + "\n")
                    
        
                    #  Ensure image completion
                    if slave_image_result.IsIncomplete():
                        logfile.write(str(time.time_ns()) + ': Slave Camera Image incomplete with image status %d ...\n' % slave_image_result.GetImageStatus())
                        
                    else:
        
                        width = slave_image_result.GetWidth()
                        height = slave_image_result.GetHeight()
            
                        logfile.write(str(time.time_ns())+': Grabbed Slave Camera Image %d, width = %d, height = %d\n' % (image_count, width, height))
                        
                        slave_image_converted = slave_image_result.Convert(PySpin.PixelFormat_RGB8)
                        
                        image_conversion_time = time.time_ns()
                        logfile.write(str(image_conversion_time) + ": Image Conversion duration: " + str(image_conversion_time - image_retrived_time) + "\n")
                        
                        # Create unique file ID
                        
                        if folder_name != None:
                            
                            if os.path.isdir(folder_name) == True:
                                slave_filename = folder_name + '%s.jpg' % (image_count)
                                
                            else:
                                logfile.write("ERROR: Invalid Folder Path \n")
                                return False
                            
                        else:     
                            slave_filename = '%s.jpg' % (image_count)
         
                        slave_image_converted.Save(slave_filename)

                        image_save_time = time.time_ns()
                        logfile.write(str(image_save_time) + ": Saved image at location : " + slave_filename + '\n') 
                        logfile.write(str(image_save_time) + ": Image save duration: " + str(image_save_time - image_conversion_time) + "\n")
                        slave_image_result.Release()
        
                        end_time = time.time_ns()
                        logfile.write(str(end_time) + ": Camera Frame: " + str(image_count) + " Loop End\n")
                        logfile.write(str(end_time) + ": Image Capture duration: " + str(end_time - start_time) + "\n \n")
                        
                except PySpin.SpinnakerException as ex:
                    logfile.write('Error: %s\n' % ex)
                    return False
        
                image_count += 1
                
                
        # End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        slave_cam.EndAcquisition()
    
    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s \n' % ex)
        return False

    return result


def reset_trigger(cam):
    
    nodemap = cam.GetNodeMap()
    try:
        result = True
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            logfile.write('Unable to disable trigger mode (node retrieval). Aborting...\n')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            logfile.write('Unable to disable trigger mode (enum entry retrieval). Aborting...\n')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        logfile.write('Trigger mode disabled...\n')

    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s \n' % ex)
        result = False

    return result


def print_device_info(cam):

    logfile.write('*** DEVICE INFORMATION ***\n')
    
    nodemap = cam.GetTLDeviceNodeMap()
    
    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                logfile.write('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            logfile.write('Device control information not available.\n')

    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s \n' % ex)
        return False

    return result

def run_slave_camera(serial_number, control_pipe, folder_name):
    
    global logfile
    try:
        logfile = open(folder_name +'SlaveCameraLog.txt', 'w')
    except IOError:
        print('Unable to write to directory. Please check permissions.')
        input('Press Enter to exit...')
        return False
    
    # new folder to hold images 
    
    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    logfile.write('Library version: %d.%d.%d.%d\n' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    logfile.write('Number of cameras detected: %d\n' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        logfile.write('No camera cameras!\n')
        return False
 
    slave_cam = cam_list.GetBySerial(serial_number)  
    
    try:
        result = True
        
        # get slave cam info
        logfile.write("TL LAYER SLAVE INFO\n")
        result &= print_device_info(slave_cam)
        
        # Initialize cameras
        slave_cam.Init()
        
        # configure cameras
        logfile.write("CONFIGURING HARDWARE TRIGGER ON SLAVE CAMERA\n")
        result &= configure_trigger(slave_cam, "Hardware")
        
        # verify configuration worked
        if result is False:
            return False
        
        #set image format
        logfile.write("\nSetting SLAVE CAMERA to RGB8 format\n")
        result &= configure_image_format(slave_cam)
         
        #set slave camera exposure
        logfile.write("\nSetting SLAVE CAMERA Exposure\n")
        result &= configure_exposure(slave_cam)
        
        #acquire images
        logfile.write("\nACQUIRING IMAGE\n")
        result&= acquire_images_slave_camera(slave_cam, control_pipe, folder_name)
        
        #reset trigger
        logfile.write("\nRESETTING TRIGGERS\n")
        result &= reset_trigger(slave_cam)
        
        # Deinitialize camera
        logfile.write("\nDEINITIALIZING CAMERAS\n")
        slave_cam.DeInit()

    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s\n' % ex)
        result = False
        
    finally:

        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del slave_cam
    
        # Clear camera list before releasing system
        cam_list.Clear()
    
        # Release system instance
        system.ReleaseInstance()
    
        logfile.write('exiting\n')
        logfile.close()
        return result
    
  