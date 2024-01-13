#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import cv2
import jetson.inference
import jetson.utils
from jetson_inference import depthNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log,cudaFromNumpy
import numpy as np
from depthnet_utils import depthBuffers



from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaFromNumpy,cudaToNumpy

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 


# parse the command line
parser1 = argparse.ArgumentParser(description="Mono depth estimation on a video/image stream using depthNet DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=depthNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser1.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser1.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser1.add_argument("--network", type=str, default="fcn-mobilenet", help="pre-trained model to load, see below for options")
parser1.add_argument("--visualize", type=str, default="input,depth", help="visualization options (can be 'input' 'depth' 'input,depth'")
parser1.add_argument("--depth-size", type=float, default=1.0, help="scales the size of the depth map visualization, as a percentage of the input size (default is 1.0)")
parser1.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser1.add_argument("--colormap", type=str, default="inferno", help="colormap to use for visualization (default is 'viridis-inverted')",
                                  choices=["inferno", "inferno-inverted", "magma", "magma-inverted", "parula", "parula-inverted", 
                                           "plasma", "plasma-inverted", "turbo", "turbo-inverted", "viridis", "viridis-inverted"])


try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

################################
try:
	args1 = parser1.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
# load the segmentation network
net1 = depthNet(args1.network, sys.argv)
depth_field = net1.GetDepthField()
# create buffer manager
buffers = depthBuffers(args1)

###############################
# create video sources and outputs
input = cv2.VideoCapture("udp://127.0.0.1:10000")
output = videoOutput("output.mp4")
	
# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = detectNet(model="model/ssd-mobilenet.onnx", labels="model/labels.txt", 
#                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
#                 threshold=args.threshold)

# process frames until EOS or the user exits

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 15
fontColor              = (255,0,0)
thickness              = 9
lineType               = 4

while True:
    # capture the next image
    ret,frame=input.read()

    img = cudaFromNumpy(frame)

    if img is None: # timeout
        continue  
        
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
    	
    	img_input = cudaFromNumpy(frame)
    	if img_input is None:
    		continue
    	
    	# allocate buffers for this size image
    	buffers.Alloc(img_input.shape, img_input.format)

    	# process the mono depth and visualize
    	net1.Process(img_input, buffers.depth, args1.colormap, args1.filter_mode)

    	# composite the images
    	if buffers.use_input:
        	cudaOverlay(img_input, buffers.composite, 0, 0)
        
    	if buffers.use_depth:
        	cudaOverlay(buffers.depth, buffers.composite, img_input.width if buffers.use_input else 0, 0)
	
    	# render the output image
    	#output.Render(buffers.composite)

    	# update the title bar
    	output.SetStatus("{:s} | {:s} | Network {:.0f} FPS".format(args1.network, net1.GetNetworkName(), net1.GetNetworkFPS()))

    	# print out performance info
    	cudaDeviceSynchronize()
    	net1.PrintProfilerTimes()
    
    	#added
    	#lets say we are trying to get the depth values for X - 500 and y 640, we would do it like this
    	scale_x = float(net1.GetDepthFieldWidth()) / float(img_input.width)
    	scale_y = float(net1.GetDepthFieldHeight()) / float(img_input.height)

    	u=detection.Center[0]
    	v=detection.Center[1]
    	depth_x = int(scale_x * u)
    	depth_y = int(scale_y * v)
    	d= depth_field[depth_x ,depth_y]
    	dn= (d*10)/1.46
    	

    	print("-----")
    	print("ID is", detection.ClassID)
    	xyz= net.GetClassLabel(detection.ClassID)
    	print(xyz)
    	print("U is ",u)
    	print("V is ",v)
    	print("Depth: ",dn, "cm")
    	print(detection)
    	print(detection.Center)
    	print("===============")
    	print("W ",float(img_input.width))
    	print("H ",float(img_input.height))
    	
          
	###########################

    # render the image
    output.Render(img)
    #cv2.imshow("s",cudaToNumpy(img))

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()
    
    cv2.putText(cudaToNumpy(img),'HEYY', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    cv2.imshow('Final video',cudaToNumpy(img))
    cv2.waitKey(1)

    # exit on input/output EOS
    if not output.IsStreaming():
        break
  
 
