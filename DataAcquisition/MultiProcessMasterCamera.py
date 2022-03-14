# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:53:25 2022

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
    
    # to 10000us
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
    

def grab_next_image_by_trigger(cam):
    
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


def acquire_images_master_camera(master_cam, control_pipe, folder_name, slave_pipe=None): # nodemap, nodemap_tldevice):
    

    master_nodemap = master_cam.GetNodeMap()
    
    logfile.write('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous
        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        master_node_acquisition_mode = PySpin.CEnumerationPtr(master_nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(master_node_acquisition_mode) or not PySpin.IsWritable(master_node_acquisition_mode):
            logfile.write('Unable to set Master Camera acquisition mode to continuous (enum retrieval). Aborting...\n')
            return False

        # Retrieve entry node from enumeration node
        master_node_acquisition_mode_continuous = master_node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(master_node_acquisition_mode_continuous) or not PySpin.IsReadable(
                master_node_acquisition_mode_continuous):
            logfile.write('Unable to set Master Camera acquisition mode to continuous (entry retrieval). Aborting...\n')
            return False

        # Retrieve integer value from entry node
        master_acquisition_mode_continuous = master_node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        master_node_acquisition_mode.SetIntValue(master_acquisition_mode_continuous)

        logfile.write('Master camera acquisition mode set to continuous...\n')
        
        #  Begin acquiring images
        master_cam.BeginAcquisition()
    
        logfile.write('Acquiring images...\n\n')
        
        master_TLnodemap = master_cam.GetTLDeviceNodeMap()
        
        master_device_serial_number = ''
        master_node_device_serial_number = PySpin.CStringPtr(master_TLnodemap.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(master_node_device_serial_number) and PySpin.IsReadable(master_node_device_serial_number):
            master_device_serial_number = master_node_device_serial_number.GetValue()
            logfile.write('Master cam serial number retrieved as %s...\n' % master_device_serial_number)
            
        master_node_softwaretrigger_cmd = PySpin.CCommandPtr(master_nodemap.GetNode('TriggerSoftware'))
        if not PySpin.IsAvailable(master_node_softwaretrigger_cmd) or not PySpin.IsWritable(master_node_softwaretrigger_cmd):
            logfile.write('Unable to retrive software trigger node. Aborting...\n')
            
        
        # image count 
        image_count = 0
        
        # starting state
        state = None
        
        # state machine loop
        while state != "exit":
                
            # read current command
            command = control_pipe.recv()

            if command == "capture_image":
                logfile.write("Command received: capture_image, time: "+ str(time.time_ns()) + "\n")
                state = "capture_image"
            elif command == "exit":
                logfile.write("Command received: exit, time: "+ str(time.time_ns())+"\n")
                state = "exit" 
            
            # trigger capture
            if state == "capture_image":    
                try:
         
                    #  Retrieve the next image from the trigger
                    start_time = time.time_ns()
                    master_node_softwaretrigger_cmd.Execute()
                    slave_pipe.send("capture_image")
                    logfile.write(str(start_time) + ": Camera Frame: " + str(image_count) + " Triggered \n")
                    
                    trigger_end_time = time.time_ns()
                    logfile.write(str(trigger_end_time) + ": Trigger duration: " + str(trigger_end_time-start_time) + "\n")  
                                     
                    #  Retrieve next received image
                    master_image_result = master_cam.GetNextImage(2000)
                    
                    image_retrived_time = time.time_ns()
                    logfile.write(str(image_retrived_time) + ": Image Retrieved, retrival duration: " + str(image_retrived_time - trigger_end_time) + "\n")
    
                    #  Ensure image completion
                    if master_image_result.IsIncomplete():
                        logfile.write('Master Camera Image incomplete with image status %d ...\n' % master_image_result.GetImageStatus())
                        
                    else:
    
                        width = master_image_result.GetWidth()
                        height = master_image_result.GetHeight()
                        
                        logfile.write('Grabbed Master Camera Image %d, width = %d, height = %d\n' % (image_count, width, height))
                        
                        master_image_converted = master_image_result.Convert(PySpin.PixelFormat_RGB8)
                        
                        image_conversion_time = time.time_ns()
                        logfile.write(str(image_conversion_time) + ": Image Conversion duration: " + str(image_conversion_time - image_retrived_time) + "\n")
                        
                        # Create unique file ID
                        
                        if folder_name != None:
                            
                            if os.path.isdir(folder_name) == True:
                                master_filename = folder_name + '%s.jpg' % (image_count)
                                
                            else:
                                logfile.write("ERROR: Invalid Folder Path \n")
                                return False
                            
                        else:     
                            master_filename = '%s.jpg' % (image_count)
     
                        master_image_converted.Save(master_filename)
                        
                        image_save_time = time.time_ns()
                        
                        logfile.write(str(image_save_time) + ": Saved image at location : " + master_filename + '\n') 
                        logfile.write(str(image_save_time) + ": Image save duration: " + str(image_save_time - image_conversion_time) + "\n")
                        
                        master_image_result.Release()

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
        slave_pipe.send("exit")
        master_cam.EndAcquisition()
        time.sleep(1)
    
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

def run_master_camera(serial_number, control_pipe, folder_name, slave_pipe=None):
    
    global logfile
    try:
        logfile = open(folder_name +'MasterCameraLog.txt', 'w')
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
 
    master_cam = cam_list.GetBySerial(serial_number)  
    
    try:
        result = True
        
        # get master cam info
        logfile.write("TL LAYER MASTER INFO\n")
        result &= print_device_info(master_cam)
        
        # Initialize cameras
        master_cam.Init()
        
        # configure cameras
        logfile.write("CONFIGURING SOFTWARE TRIGGER ON MASTER CAMERA\n")
        result &= configure_trigger(master_cam, "Software")
        
        # verify configuration worked
        if result is False:
            return False
        
        #Configure output lin on master camera
        logfile.write("\nCONFIGURING OUTPUT LINE ON MASTER CAMERA\n")
        result &= configure_output_line(master_cam)
        
        #enable 3.3V output
        logfile.write("\nENABLING 3.3V OUTPUT ON MASTER CAMERA\n")
        result &= Venable(master_cam, True)
        
        #set image format
        logfile.write("\nSetting MASTER CAMERA to RGB8 format\n")
        result &= configure_image_format(master_cam)
         
        #set master camera exposure
        logfile.write("\nSetting MASTER CAMERA Exposure\n")
        result &= configure_exposure(master_cam)
        
        #acquire images
        logfile.write("\nACQUIRING IMAGE\n")
        result&= acquire_images_master_camera(master_cam, control_pipe, folder_name, slave_pipe)
        
        #reset trigger
        logfile.write("\nRESETTING TRIGGERS\n")
        result &= reset_trigger(master_cam)
        
        # turn off 3.3v output
        logfile.write("\nDISABLING 3.3V OUTPUT\n")
        Venable(master_cam, False)
        
        # Deinitialize camera
        logfile.write("\nDEINITIALIZING CAMERAS\n")
        master_cam.DeInit()

    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s\n' % ex)
        result = False
        
    finally:

        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del master_cam
    
        # Clear camera list before releasing system
        cam_list.Clear()
    
        # Release system instance
        system.ReleaseInstance()
    
        logfile.write('exiting\n')
        logfile.close()
        return result
    
   


    

