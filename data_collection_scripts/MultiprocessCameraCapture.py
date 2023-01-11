# -*- coding: utf-8 -*-
"""
Created on Fri May 13 21:28:36 2022

@author: lowellm
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:48:53 2022

@author: lowellm
"""


import os
import PySpin
import sys
import time
import multiprocessing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

logfile = None
    
def ImageSaveProcess(control_pipe, image_array, dimensions, folder_path, logfile_name):
 
    os.mkdir(folder_path)

    try:
        logfile = open(folder_path+logfile_name, 'w')
    except IOError:
        print('Unable to write to directory. Please check permissions.')
        input('Press Enter to exit...')
        return False
    
    process_start_time = time.time_ns()
    logfile.write(str(process_start_time) + ": Starting image saving process\n")
    
    buffer_image = np.zeros(dimensions, dtype=np.ubyte)
    
    shared_image = np.frombuffer(image_array.get_obj(), dtype=np.ubyte)
    shared_image = np.reshape(shared_image, dimensions)
    
    logfile.write("image dimensions is: " + str(shared_image.shape)+"\n")
    
    # copy image memmory to shared object 
    done = False
    while done == False:
        
        message = control_pipe.recv()
        parsed_message = message.split(" ")
        
        command = parsed_message[0]
        
        if command == "save_image":
           
            start_time = time.time_ns()
            logfile.write(str(start_time) + ": save_image command received\n")
            
            np.copyto(buffer_image, shared_image)
            
            image_name = parsed_message[1]
            swapped_color_buffer = cv.cvtColor(buffer_image, cv.COLOR_RGB2BGR)
            cv.imwrite(folder_path+image_name,swapped_color_buffer)
            
            end_time = time.time_ns()
            logfile.write(str(end_time) + ": image saved, write time: "+str(end_time-start_time) + "\n")
            
        if command == "exit":
            process_end_time = time.time_ns()
            logfile.write(str(process_end_time) + ": Ending image saving process\n")
            done = True
            
    logfile.close()
    return 1


def configure_acquisition_mode(cam):
    
    nodemap = cam.GetNodeMap()
    
    logfile.write('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous
        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            logfile.write('Unable to set Master Camera acquisition mode to continuous (enum retrieval). Aborting...\n')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            logfile.write('Unable to set Master Camera acquisition mode to continuous (entry retrieval). Aborting...\n')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        logfile.write('Master camera acquisition mode set to continuous...\n')
        
    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s \n' % ex)
        return False
    return result

def configure_buffer_mode(cam):
    
    try:
        result = True
        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()
        
        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
            logfile.write('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False
        
        handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(handling_mode_entry):
            logfile.write('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
            return False
        
        
        handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
        handling_mode.SetIntValue(handling_mode_entry.GetValue())
        logfile.write('\n\nBuffer Handling Mode has been set to %s\n' % handling_mode_entry.GetDisplayName())   
    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s \n' % ex)
        return False
    return result
    
    

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
    node_auto_exposure_upper_time_limit.SetValue(40000)
    
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
    logfile.write('Trigger type: ' + triggerType)


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
        
        logfile.write(str(time.time_ns())+": Camera Triggered\n")
        node_softwaretrigger_cmd.Execute()

    except PySpin.SpinnakerException as ex:
        logfile.write('Error: %s \n' % ex)
        return False

    return result
            
def acquire_images_master_camera(master_cam, slave_cam, folder, control_pipe=None): 
     
    result = True
        
    left_folder = folder + "\\left\\"
    right_folder = folder+ "\\right\\"
 
    
    master_nodemap = master_cam.GetNodeMap()
    slave_nodemap = slave_cam.GetNodeMap()
    
    logfile.write("Getting image size\n")
    
    # get image sizes
    master_node_iWidth_int = PySpin.CIntegerPtr(master_nodemap.GetNode("Width"))
    master_node_iHeight_int = PySpin.CIntegerPtr(master_nodemap.GetNode("Height"))
    
    master_image_width = master_node_iWidth_int.GetValue()
    master_image_height = master_node_iHeight_int.GetValue()
    
    master_dimensions = (master_image_height, master_image_width, 3)
    
    slave_node_iWidth_int = PySpin.CIntegerPtr(slave_nodemap.GetNode("Width"))
    slave_node_iHeight_int = PySpin.CIntegerPtr(slave_nodemap.GetNode("Height"))
    
    slave_image_width = slave_node_iWidth_int.GetValue()
    slave_image_height = slave_node_iHeight_int.GetValue()
    
    slave_dimensions = (slave_image_height, slave_image_width, 3)
    
    logfile.write("Master image size: "+ str(master_image_width) + " " + str(master_image_height) + "\n")
    logfile.write("Slave image size: "+ str(slave_image_width) + " " + str(slave_image_height) + "\n")
    
    # create shared array objects for saving processes
    logfile.write("Creating shared memmory\n")
    
    master_image_array = multiprocessing.Array('c', master_image_width*master_image_height*3) 
    slave_image_array = multiprocessing.Array('c', slave_image_width*slave_image_height*3)
    
    master_shared_image = np.frombuffer(master_image_array.get_obj(), dtype=np.ubyte)
    slave_shared_image = np.frombuffer(slave_image_array.get_obj(), dtype=np.ubyte)
    
    master_shared_image = master_shared_image.reshape((master_image_height, master_image_width,3))
    slave_shared_image = slave_shared_image.reshape((slave_image_height, slave_image_width,3))
    
    # create communication pipes to saving processes
    logfile.write("Creating control pipes\n")
    control_to_master_save, master_save_to_control = multiprocessing.Pipe()
    control_to_slave_save, slave_save_to_control = multiprocessing.Pipe()
        
    # create saving processes
    logfile.write("Creating image saving processes\n")
    master_save_process = multiprocessing.Process(target=ImageSaveProcess, args=(master_save_to_control, master_image_array, master_dimensions, left_folder, "master_saver_log.txt"))
    slave_save_process = multiprocessing.Process(target=ImageSaveProcess, args=(slave_save_to_control, slave_image_array, slave_dimensions, right_folder, "slave_saver_log.txt"))
    
    # start saving processes
    logfile.write("Starting save processes\n")
    master_save_process.start()
    slave_save_process.start()
    
    time.sleep(.1)
    
    logfile.write('*** IMAGE ACQUISITION ***\n')
    try:
        result = True
        
        master_TLnodemap = master_cam.GetTLDeviceNodeMap()
        
        master_device_serial_number = ''
        master_node_device_serial_number = PySpin.CStringPtr(master_TLnodemap.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(master_node_device_serial_number) and PySpin.IsReadable(master_node_device_serial_number):
            master_device_serial_number = master_node_device_serial_number.GetValue()
            logfile.write('Master cam serial number retrieved as %s...\n' % master_device_serial_number)
            
        master_node_softwaretrigger_cmd = PySpin.CCommandPtr(master_nodemap.GetNode('TriggerSoftware'))
        if not PySpin.IsAvailable(master_node_softwaretrigger_cmd) or not PySpin.IsWritable(master_node_softwaretrigger_cmd):
            logfile.write('Unable to retrive software trigger node. Aborting...\n')
                   
            
        master_cam.BeginAcquisition()
        slave_cam.BeginAcquisition()
        
        #clear buffer
        logfile.write("\n Clearing master camera buffer\n")
        loop_count = 0
        while loop_count < 10:
            buffer_clear = False
            while buffer_clear == False:
                try:
                    master_image_result = master_cam.GetNextImage(50)
                    logfile.write("master buffer contained image\n")
                except PySpin.SpinnakerException as ex:
                    logfile.write('Error: %s \n' % ex)
                    logfile.write("Image Buffer clear\n")
                    buffer_clear = True     
            loop_count +=1;
            
            
        logfile.write("\n Clearing slave camera buffer\n")
        loop_count = 0
        while loop_count < 10:
            buffer_clear = False
            while buffer_clear == False:
                try:
                    slave_image_result = slave_cam.GetNextImage(50)
                    logfile.write("slave buffer contained image\n")
                except PySpin.SpinnakerException as ex:
                    logfile.write('Error: %s \n' % ex)
                    logfile.write("Image Buffer clear\n")
                    buffer_clear = True     
            loop_count +=1;
        
        
        # image count 
        image_count = 0
        
        done = False
        
        logfile.write("\n"+str(time.time_ns())+": Acquiring images...\n\n')")
        
        # state machine loop
        while done == False:
                
            # read current command
            if control_pipe != None:
                command = control_pipe.recv()
            else:
                command = "capture_image"
            
            # trigger capture
            if command == "capture_image":
                
                logfile.write("\n" + str(time.time_ns())+": Command received: capture_image, time\n")
                
                try:
         
                    # execute trigger
                    start_time = time.time_ns()
                    master_node_softwaretrigger_cmd.Execute()
                    logfile.write(str(start_time) + ": Camera Frame: " + str(image_count) + " Triggered \n")
                    trigger_end_time = time.time_ns()
                    logfile.write(str(trigger_end_time) + ": Trigger duration: " + str(trigger_end_time-start_time) + "\n")  
                    
                    
                    #  Retrieve images
                    master_image_result = master_cam.GetNextImage(2000)
                    logfile.write(str(time.time_ns()) + ": Master camera image received\n")
                    slave_image_result = slave_cam.GetNextImage(2000)
                    logfile.write(str(time.time_ns()) + ": Slave camera image received\n")
                    
                    image_retrived_time = time.time_ns()
                    logfile.write(str(image_retrived_time) + ": Images Retrieved, retrival duration: " + str(image_retrived_time - trigger_end_time) + "\n")
    
                    #  Ensure image completion
                    if master_image_result.IsIncomplete() or slave_image_result.IsIncomplete():
                        if master_image_result.IsIncomplete():
                            logfile.write('Master Camera Image incomplete with image status %d ...\n' % master_image_result.GetImageStatus())
                        else:
                            logfile.write('Slave Camera Image incomplete with image status %d ...\n' % master_image_result.GetImageStatus())
                        
                    else:
    
                        master_width = master_image_result.GetWidth()
                        master_height = master_image_result.GetHeight()
                        
                        logfile.write(str(time.time_ns())+': Grabbed Master Camera Image %d, width = %d, height = %d\n' % (image_count, master_width, master_height))
                        master_image_converted = master_image_result.Convert(PySpin.PixelFormat_RGB8)
                        
                        slave_width = slave_image_result.GetWidth()
                        slave_height = master_image_result.GetHeight()
                        
                        logfile.write(str(time.time_ns())+': Grabbed Slave Camera Image %d, width = %d, height = %d\n' % (image_count, slave_width, slave_height))
                        
                        slave_image_converted = slave_image_result.Convert(PySpin.PixelFormat_RGB8)
                        
                        image_conversion_time = time.time_ns()
                        logfile.write(str(image_conversion_time) + ": Image Conversion duration: " + str(image_conversion_time - image_retrived_time) + "\n")
                        
                        # convert saved images to numpy array

                        npy_master_image = master_image_converted.GetNDArray()         
                        npy_slave_image = slave_image_converted.GetNDArray()
                        
                        # copy images to shared buffer 
                        
                        np.copyto(master_shared_image, npy_master_image)
                        np.copyto(slave_shared_image, npy_slave_image)
                        
                        image_to_buffer_time = time.time_ns()
                        
                        logfile.write(str(image_to_buffer_time) + ": Image To Shared Memmory Duration: " + str(image_to_buffer_time - image_conversion_time) + "\n")
                        
                        # send command so images are saved by saving processes
                        control_to_master_save.send("save_image " +str(image_count) + ".jpg")
                        control_to_slave_save.send("save_image " + str(image_count) + ".jpg")
                        
                        master_image_result.Release()
                        slave_image_result.Release()

                        end_time = time.time_ns()
                        
                        logfile.write(str(end_time) + ": Camera Frame: " + str(image_count) + " Loop End\n")
                        logfile.write(str(end_time) + ": Image Capture duration: " + str(end_time - start_time) + "\n")
    
                        
                except PySpin.SpinnakerException as ex:
                    logfile.write('Error: %s\n' % ex)
                    return False

                image_count += 1
                
                if control_pipe==None:
                    time.sleep(.1)
                
                if (image_count > 100) and control_pipe==None:
                    done = True
            
                    
            elif command == "exit":
                logfile.write(str(time.time_ns()) + ": Command received: exit\n")
                done = True            
                
        # End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
 
        logfile.write("\n"+str(time.time_ns())+": Ending acquisition\n")
        logfile.write(str(time.time_ns())+": Sending exit command to saving proccesses\n")
        
        control_to_master_save.send("exit")
        control_to_slave_save.send("exit")
        
        master_save_process.join()
        slave_save_process.join()
        
        logfile.write(str(time.time_ns())+": Saving processes closed \n")
        
        master_cam.EndAcquisition()
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

def run_cameras(master_serial_number, slave_serial_number, folder_name, control_pipe=None):
    
    global logfile
    try:
        logfile = open(folder_name +'\\CameraLog.txt', 'w')
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
 
    master_cam = cam_list.GetBySerial(master_serial_number)
    slave_cam = cam_list.GetBySerial(slave_serial_number)
    
    try:
        result = True
        
        # get master cam info
        logfile.write("TL LAYER MASTER INFO\n")
        result &= print_device_info(master_cam)
        
        # Initialize master camera
        master_cam.Init()
        
        #set master camera exposure
        logfile.write("\nSetting MASTER CAMERA Exposure\n")
        result &= configure_exposure(master_cam)
        
        #set image format
        logfile.write("\nSetting MASTER CAMERA to RGB8 format\n")
        result &= configure_image_format(master_cam)

        #Configure output lin on master camera
        logfile.write("\nCONFIGURING OUTPUT LINE ON MASTER CAMERA\n")
        result &= configure_output_line(master_cam)
        
        #enable 3.3V output
        logfile.write("\nENABLING 3.3V OUTPUT ON MASTER CAMERA\n")
        result &= Venable(master_cam, True)
        
        # configure trigger 
        logfile.write("CONFIGURING SOFTWARE TRIGGER ON MASTER CAMERA\n")
        result &= configure_trigger(master_cam, "Software")
        
        # configure acquisition mode
        logfile.write("CONFIGURING ACQUISITION MODE ON MASTER CAMERA\n")
        result &= configure_acquisition_mode(master_cam)
        
        # # # configure buffer handling
        # logfile.write("CONFIGURING BUFFER MODE ON MASTER CAMERA\n")
        # result &= configure_buffer_mode(master_cam)
        
        # get slave cam info
        logfile.write("TL LAYER SLAVE INFO\n")
        result &= print_device_info(slave_cam)
        
        # Initialize slave cameras
        slave_cam.Init()
        
        #set slave camera exposure
        logfile.write("\nSetting SLAVE CAMERA Exposure\n")
        result &= configure_exposure(slave_cam)
        
        #set slave image format
        logfile.write("\nSetting SLAVE CAMERA to RGB8 format\n")
        result &= configure_image_format(slave_cam)

        # configure hardware trigger on master cam
        logfile.write("CONFIGURING HARDWARE TRIGGER ON SLAVE CAMERA\n")
        result &= configure_trigger(slave_cam, "HARDWARE")
        
        # configure acquisition mode
        logfile.write("CONFIGURING ACQUISITION MODE ON SLAVE CAMERA\n")
        result &= configure_acquisition_mode(slave_cam)
        
        # # # configure buffer handling
        # logfile.write("CONFIGURING BUFFER MODE ON SLAVE CAMERA\n")
        # result &= configure_buffer_mode(slave_cam)
        
        #acquire images
        logfile.write("\nACQUIRING IMAGES\n")
        result&= acquire_images_master_camera(master_cam, slave_cam, folder_name, control_pipe)
        
        # reset slave trigger 
        logfile.write("\nRESETTING SLAVE TRIGGER\n")
        result &= reset_trigger(slave_cam)
        
        # Deinitialize camera
        logfile.write("\nDEINITIALIZING SLAVE CAMERA\n")
        slave_cam.DeInit()
        
        #reset trigger
        logfile.write("\nRESETTING MASTER TRIGGER\n")
        result &= reset_trigger(master_cam)
        
        # turn off 3.3v output
        logfile.write("\nDISABLING MASTER 3.3V OUTPUT\n")
        Venable(master_cam, False)
        
        # Deinitialize camera
        logfile.write("\nDEINITIALIZING MASTER CAMERA\n")
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
        del slave_cam
    
        # Clear camera list before releasing system
        cam_list.Clear()
    
        # Release system instance
        system.ReleaseInstance()
    
        logfile.write('exiting\n')
        logfile.close()
        return result
    

if __name__ == "__main__":
    
    folder_name="\\stand_alone_" + str(time.time_ns())+"\\" 
    
    path = os.getcwd()
    base_folder = path + folder_name
    camera_folder = path + folder_name + "\\camera\\"

    os.mkdir(base_folder)
    os.mkdir(camera_folder)
    
    run_cameras('20162694', '20162696', camera_folder)