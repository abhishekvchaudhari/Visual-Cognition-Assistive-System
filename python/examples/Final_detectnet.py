#!/usr/bin/env python3

import sys
import argparse
import cv2
import jetson.inference
import jetson.utils
from jetson_inference import depthNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log,cudaFromNumpy
import numpy as np
from gtts import gTTS
from playsound import playsound
from depthnet_utils import depthBuffers
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaFromNumpy,cudaToNumpy
import threading,queue
import uuid

class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

  # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # break
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
    def release(self):
        self.cap.release()
        return
        
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





def generateDirection(final_list_of_objects):

   dir=""
   oppDir=""
   pos=getOrientation(final_list_of_objects[0][1],final_list_of_objects[0][2])
   if pos==0:
        dir="Right"
        oppDir="Left"
   if pos==1:
        dir="Left"
        oppDir="Right"
   sentence=final_list_of_objects[0][0]+" " + "on your " + oppDir+ " at " + str(final_list_of_objects[0][3]) +" cm. "+" Move to the " + dir
   print(sentence)
   language="en"
   obj= gTTS(text=sentence, lang=language, slow=False)
   unique_filename = str(uuid.uuid4())
   obj.save("MainObject"+unique_filename+".mp3")
   playsound("MainObject"+unique_filename+".mp3")
   
   if(len(final_list_of_objects[1:])):
        sentence=" and other objects like "
        for i in final_list_of_objects[1:]:
            sentence=sentence+i[0]+" at " + str(i[3])+" cm. "
 	
        print(sentence)
        obj= gTTS(text=sentence, lang=language, slow=False)
        obj.save("OtherObjectAudio"+unique_filename+".mp3")
        playsound("OtherObjectAudio"+unique_filename+".mp3")
	#return sentence
	#audio.say(sentence)

 	
def getOrientation(left, right):
 	x1=left
 	x2=640-right
 	
 	if x2>x1:#obj at lhs
 		return 0
 	else:#obj at rhs
 		return 1
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
# load the segmentation networksentence
net1 = depthNet(args1.network, sys.argv)
depth_field = net1.GetDepthField()
# create buffer manager
buffers = depthBuffers(args1)

###############################
# create video sources and outputs
cap = VideoCapture("udp://127.0.0.1:10000")
	
# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)



while True:
    # capture the next image
    frame=cap.read()

    img = cudaFromNumpy(frame)

    if img is None: # timeout
        continue  
        
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    list_of_objects=[]

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
	

    	# print out performance info
    	#cudaDeviceSynchronize()
    	#net1.PrintProfilerTimes()
    
    	#added
    	#lets say we are trying to get the depth values for X - 500 and y 640, we would do it like this
    	scale_x = float(net1.GetDepthFieldWidth()) / float(img_input.width)
    	scale_y = float(net1.GetDepthFieldHeight()) / float(img_input.height)

    	u=detection.Center[0]
    	v=detection.Center[1]
    	depth_x = int(scale_x * u)
    	depth_y = int(scale_y * v)
    	d= depth_field[depth_x ,depth_y]
    	dn= int((d*10)*2.2)
    	

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
    	

    	list_of_objects.append([xyz,detection.Left, detection.Right,dn])

    final_list_of_objects=sorted(list_of_objects,key=lambda x: x[-1])


    try:
        cv2.imshow('Detection video',cudaToNumpy(img))
        cv2.imshow('DepthMap video',cudaToNumpy(buffers.composite))
    except:
        pass

    if(len(final_list_of_objects)>0): 	
    	generateDirection(final_list_of_objects)
    key=cv2.waitKey(0)	
    if key==ord('n'):
       continue
    elif key==ord('q'):
       break

cap.release()
cv2.destroyAllWindows() 

  

 
