# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:08:38 2021

@author: Michael
"""

import numpy as np
import math
import socket 
import sys
import threading
import time 
import open3d as o3d
import urllib.request
import urllib.parse
import os


#global logfile
logfile = None

# Recv Lidar Port number

IP_address = "192.168.1.2"
lidar_recv_port = 2368
pkt_length = 1248 - 42 # 42 byte UDP header 

points_per_packet = 32*12

firing_period = 55.296*10**(-6) # in seconds

Horizontile_FOV = 60 # set degrees for horizintile FOV

padding_packets = 4 # additional packets to include in saved scan must be even,

vert_angles = np.array([-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15], dtype=np.float32)*np.pi/180 # in radians
vert_correction = np.array([11.2, -.7, 9.7, -2.2, 8.1, -3.7, 6.6, -5.1, 5.1, -6.6, 3.7, -8.1, 2.2, -9.7, .7, -11.2])/1000 # in m



def run_lidar(folder_path, fps, control_pipe = None, cam_pipe=None):
    
    return_value = -1
    global logfile
    
    RPM =  int(60*fps)
    Azimuth_Resolution = RPM*1/60*360*firing_period
    pkt_ang_swp = Azimuth_Resolution*24

    pkt_count = math.ceil(Horizontile_FOV/pkt_ang_swp)+1+padding_packets # add 1 since capture starts ~1 pkt before ideal and four extra padding packets, 
    frm_len = pkt_length*pkt_count

    points = np.zeros((pkt_count*points_per_packet, 4), dtype=np.float32)

    # the lidar is mounted at 90 degrees so an azimuth angle of 90 is straight ahead

    start_angle = 90-Horizontile_FOV/2# add one packet to start and end angle 
    end_angle = 90 + Horizontile_FOV/2

    padded_start_angle = 90-Horizontile_FOV/2-pkt_ang_swp*(padding_packets/2) # add one packet to start and end angle 
    padded_end_angle = 90 + Horizontile_FOV/2+pkt_ang_swp*(padding_packets/2)
    

    logfile = open(folder_path + "\\LidarLog.txt", "w")
    logfile.write("Assumed RPM: " + str(RPM) + "\n")
    logfile.write("Target Horizontile_FOV: " + str(Horizontile_FOV) + "\n")
    logfile.write("FOV start angle: " + str(start_angle) + " FOV end angle: " + str(end_angle)+"\n")
    logfile.write("Packet padding: " + str(padding_packets) + "\n")
    logfile.write("Packet angle sweep: " + str(pkt_ang_swp) + "\n")
    logfile.write("Packets in Frame: " + str(pkt_count) +"\n")
    logfile.write("Points in Frame: "+str(pkt_count*points_per_packet)+"\n")
    logfile.write("XYZI is: "+str(pkt_count*points_per_packet) + "x4 numpy array of type numpy.uint\n")
    logfile.write("XYZ values are in mm")
    logfile.write("Initializing Lidar\n")
    
    
    # getting current parameters
    status_URL = "http://192.168.1.201/cgi/status.json"
    status = urllib.request.urlopen(status_URL)
    logfile.write("Current LIDAR status\n")
    logfile.write(str(status.read()))
    logfile.write("\n")
    
    # update RPM and return type
    rpm_and_return_param = {"rpm":str(RPM), "returns":"strongest"}
    logfile.write("Updating lidar RPM and return type\n")
    logfile.write(str(rpm_and_return_param))
    logfile.write("\n")
    
    post_URL = "http://192.168.1.201/cgi/setting"
    post_data = urllib.parse.urlencode(rpm_and_return_param).encode("ascii")
    post_response = urllib.request.urlopen(url=post_URL, data=post_data)
    
    #update FOV
    fov = {"start":str(int(padded_start_angle-pkt_ang_swp)), "end":str(int(padded_end_angle + pkt_ang_swp))}
    logfile.write("Updating lidar FOV\n")
    logfile.write(str(fov))
    logfile.write("\n")
    
    post_URL = "http://192.168.1.201/cgi/setting/fov"
    post_data = urllib.parse.urlencode(fov).encode("ascii")
    post_response = urllib.request.urlopen(url=post_URL, data=post_data)
    
    logfile.write("Pausing 5s to stablize settings\n")
    time.sleep(5)
    
    # get updated parameters
    status_URL = "http://192.168.1.201/cgi/status.json"
    status = urllib.request.urlopen(status_URL)
    logfile.write("Current LIDAR status\n")
    logfile.write(str(status.read()))
    logfile.write("\n")
    
    
    logfile.write("creating UDP port on IP: " + str(IP_address) + " port number " + str(lidar_recv_port) + "\n")
    
    try :# create UDP socket
        logfile.write("attempting to create socket\n")
        lidar_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logfile.write("socket created\n")
        logfile.write("attempting to bind IP and port to socket\n")
        lidar_socket.bind((IP_address, lidar_recv_port))
        logfile.write("socket bound\n")
    
        logfile.write("starting receive loop\n")
        logfile.write("padded target start angle: " + str(padded_start_angle) + " padded target end angle: " + str(padded_end_angle) + "\n")
        
        # pre allocate frame buffer
        frame = np.zeros(frm_len, dtype=np.ubyte)
        
        
        frame_count = 0
        
        # starting state
        if control_pipe == None:
            state = "continuous_capture"
            logfile.write("No control pipe, starting frame capture\n")

        else:
            state = "pause"
        
        while(state != "exit"):
            if control_pipe != None:
                if control_pipe.poll() == True:
                    command = control_pipe.recv()
                    
                    # capture a single frame
                    if command == "frame_capture":
                        state = "frame_capture"
                        logfile.write("command received, state changed to frame_capture\n")
                        
                    # capture frames continuously
                    elif command == "continuous_capture":
                        state = "continuous_capture"
                        logfile.write("command received, state changed to continuous_capture\n")
                                
                    # pause capture
                    elif command == "pause":
                        state = "pause"
                        logfile.write("command received, state changed to pause\n")
                    # exit process
                    elif command == "exit":
                        state = "exit"
                        logfile.write("command received, state changed to exit\n")
            
            if state == "frame_capture" or state == "continuous_capture":
                
                logfile.write("starting packet scan\n")
                
                pkt_ind = 0;
                
                # scan for start of frame,
                
                start_packet = False
                while start_packet == False:
                    (pkt, address) = lidar_socket.recvfrom(2000)
                    
                    npy_pkt = np.frombuffer(pkt, dtype=np.ubyte)
                
                    # retrieve start and end angle of packet
                
                    pkt_start_angle_bytes = npy_pkt[2:2+2].astype(dtype=np.uint32)
                    pkt_start_angle = ((pkt_start_angle_bytes[1]<<8) + pkt_start_angle_bytes[0])/100 #divide by 100 since angles are fixed point with .01 the basic unit,
                    
                    pkt_end_angle_bytes = npy_pkt[11*100+2:11*100+4].astype(dtype=np.uint32)
                    pkt_end_angle = ((pkt_end_angle_bytes[1]<<8) + pkt_end_angle_bytes[0])/100 
                 
                    if abs(pkt_start_angle - padded_start_angle) <= 1.5*pkt_ang_swp: 
                        start_packet = True
                
                # launch camera capture once frame recording starts.
                start_time = time.time_ns()
                
                if cam_pipe != None:
                    logfile.write("Sending capture_image command to camera\n")
                    cam_pipe.send("capture_image")
                
                logfile.write("Frame: " + str(frame_count) + " start time: " + str(start_time) + "\n")
                logfile.write("packet: " + str(0)) 
                logfile.write(" packet length is: " + str(len(npy_pkt)))
                logfile.write(" start angle: " + str(pkt_start_angle) + " end angle: " + str(pkt_end_angle) + "\n")
                       
                # save points 
                
                points_index = 0
                for i in range(12):
                    
                    angle_bytes = npy_pkt[i*100+2:i*100+2+2].astype(dtype=np.uint32)
                    
                    horz_angle = np.copy(((angle_bytes[1]<<8) + angle_bytes[0])/100)*np.pi/180 # in radians, 
                    
                    dist_inten_buffer = np.copy(npy_pkt[i*100+4:(i+1)*100].astype(dtype=np.uint32))
                    dist_inten_buffer = np.reshape(dist_inten_buffer, (32,3))
                    
                    depth_values = np.copy((dist_inten_buffer[:,1]<<8) + dist_inten_buffer[:,0])*2/1000
                    intensity_values = np.copy(dist_inten_buffer[:,2])
                    
                    horz_angle_vec = np.repeat(horz_angle, len(vert_angles))

                    points[points_index:points_index+16, 0] = depth_values[0:16]*np.cos(vert_angles)*np.sin(horz_angle_vec)
                    points[points_index:points_index+16, 1] = depth_values[0:16]*np.cos(vert_angles)*np.cos(horz_angle_vec)
                    points[points_index:points_index+16, 2] = depth_values[0:16]*np.sin(vert_angles) + vert_correction
                    points[points_index:points_index+16, 3] = intensity_values[0:16]
                    
                    horz_angle = horz_angle + Azimuth_Resolution*np.pi/180
                    horz_angle_vec = np.repeat(horz_angle, len(vert_angles))
                    
                    points[points_index+16:points_index+32, 0] = depth_values[16:32]*np.cos(vert_angles)*np.sin(horz_angle_vec)
                    points[points_index+16:points_index+32, 1] = depth_values[16:32]*np.cos(vert_angles)*np.cos(horz_angle_vec)
                    points[points_index+16:points_index+32, 2] = depth_values[16:32]*np.sin(vert_angles) + vert_correction
                    points[points_index+16:points_index+32, 3] = intensity_values[16:32]
                    
                    points_index +=32
                    
                
                frame[pkt_length*pkt_ind:pkt_length*(pkt_ind+1)] = npy_pkt
                frame_start_angle = pkt_start_angle
                pkt_ind += 1
                
                
                for i in range(1,pkt_count):
                    
                    (packet, address) = lidar_socket.recvfrom(2000)
                    
                    npy_pkt = np.frombuffer(packet, dtype=np.ubyte)
            
                    # retrieve start and end angle of packet
            
                    pkt_start_angle_bytes = npy_pkt[2:2+2].astype(dtype=np.uint32)
                    pkt_start_angle = ((pkt_start_angle_bytes[1]<<8) + pkt_start_angle_bytes[0])/100
                    
                    pkt_end_angle_bytes = npy_pkt[100*11+2:100*11+2+2].astype(dtype=np.uint32)
                    pkt_end_angle = ((pkt_end_angle_bytes[1]<<8) + pkt_end_angle_bytes[0])/100
                    
                    logfile.write("packet: " + str(i)) 
                    logfile.write(" packet length is: " + str(len(npy_pkt)))
                    logfile.write(" padded start angle: " + str(pkt_start_angle) + " padded end angle: " + str(pkt_end_angle) + "\n")
                    
                    for i in range(12):
                        
                        angle_bytes = npy_pkt[i*100+2:i*100+2+2].astype(dtype=np.uint32)
                        
                        horz_angle = np.copy(((angle_bytes[1]<<8) + angle_bytes[0])/100)*np.pi/180 
                        
                        dist_inten_buffer = npy_pkt[i*100+4:(i+1)*100].astype(dtype=np.uint32)
                        dist_inten_buffer = np.reshape(dist_inten_buffer, (32,3))
                        
                        depth_values = np.copy((dist_inten_buffer[:,1]<<8) + dist_inten_buffer[:,0])*2/1000
                        intensity_values = np.copy(dist_inten_buffer[:,2])
                        
                        horz_angle_vec = np.repeat(horz_angle, len(vert_angles))

                        points[points_index:points_index+16, 0] = depth_values[0:16]*np.cos(vert_angles)*np.sin(horz_angle_vec)
                        points[points_index:points_index+16, 1] = depth_values[0:16]*np.cos(vert_angles)*np.cos(horz_angle_vec)
                        points[points_index:points_index+16, 2] = (depth_values[0:16]*np.sin(vert_angles) + vert_correction)
                        points[points_index:points_index+16, 3] = intensity_values[0:16]
                        
                        horz_angle = horz_angle + Azimuth_Resolution*np.pi/180 
                        horz_angle_vec = np.repeat(horz_angle, len(vert_angles))
                        
                        points[points_index+16:points_index+32, 0] = depth_values[16:32]*np.cos(vert_angles)*np.sin(horz_angle_vec)
                        points[points_index+16:points_index+32, 1] = depth_values[16:32]*np.cos(vert_angles)*np.cos(horz_angle_vec)
                        points[points_index+16:points_index+32, 2] = (depth_values[16:32]*np.sin(vert_angles) + vert_correction) 
                        points[points_index+16:points_index+32, 3] = intensity_values[16:32]
                        
                        points_index +=32
                    
                    frame[pkt_length*pkt_ind:pkt_length*(pkt_ind+1)] = npy_pkt
                    pkt_ind += 1
                
                frame_end_angle = pkt_end_angle
                end_time = time.time_ns()
                logfile.write("Frame end time: " + str(end_time) + "\n")
                logfile.write("Frame start angle: " + str(frame_start_angle) + " frame end angle: " + str(frame_end_angle) + "\n")
                logfile.write("Frame capture time: " + str(end_time-start_time) + "\n")
                logfile.write("Saving packet frame as lidar_packet_frame_" + str(frame_count)+".npy"+"\n\n")
                logfile.write("Saving XYZI frame as lidar_XYZI_frame_" + str(frame_count)+".npy"+"\n\n")
                logfile.write("Saving pcd frame as lidar_PCD_frame_" + str(frame_count)+".pcd"+"\n\n")
                
                np.save(folder_path + "\\lidar_packet_frame_"+str(frame_count), frame)
                np.save(folder_path + "\\lidar_XYZI_frame_" + str(frame_count), points)
                
                # write pcd file
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:,0:3])
                o3d.io.write_point_cloud(folder_path + "\\"+str(frame_count)+".pcd", pcd)
                  
                frame_count +=1     
                 
                # after single frame put back in pause state 
                if state == "frame_capture":
                    state = "pause"
            
            elif state == "pause":
                #clears UDP buffer when capture is paused
                (pkt, address) = lidar_socket.recvfrom(2000)
                continue
            
        logfile.write("captured frames: " + str(frame_count) + "\n")
        return_value = frame_count

    except socket.error as msg:
        logfile.write("socket error: " + str(msg[0]) + "\n")
        return_value = -1
        
    finally:
        if cam_pipe != None:
            logfile.write("closing camera capture process\n")
            cam_pipe.send("exit")    
        
        logfile.write("closing port and exiting\n")
        lidar_socket.close()
        logfile.close()
        
        return return_value       


if __name__ == "__main__":
    
    time.time_ns()
    lidar_test_folder = os.getcwd() + "\\lidar_test"+ str(time.time_ns()) + "\\"
    os.mkdir(lidar_test_folder)
    
    run_lidar(lidar_test_folder)
    
    
    
        





        
