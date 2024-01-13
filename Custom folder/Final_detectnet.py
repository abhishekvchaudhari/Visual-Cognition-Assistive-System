#!/usr/bin/env python3

import sys
import argparse
import cv2
from jetson_inference import depthNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, Log,cudaFromNumpy
import numpy as np
from gtts import gTTS
from playsound import playsound
from depthnet_utils import depthBuffers
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaFromNumpy,cudaToNumpy
import threading,queue
import uuid

#this is required as we have to discard the older frames and consider only the new frames from the go pro camera
#need to run keep alive code to start streaming the frames over the udp
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

#required to control the behavior of algorithm both detectnet and depthnet
# parse the command line
def generate_args():
    detect_parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    detect_parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    detect_parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    detect_parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
    detect_parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
    detect_parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 


    # parse the command line
    depth_parser = argparse.ArgumentParser(description="Mono depth estimation on a video/image stream using depthNet DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=depthNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    depth_parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    depth_parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    depth_parser.add_argument("--network", type=str, default="fcn-mobilenet", help="pre-trained model to load, see below for options")
    depth_parser.add_argument("--visualize", type=str, default="input,depth", help="visualization options (can be 'input' 'depth' 'input,depth')")
    depth_parser.add_argument("--depth-size", type=float, default=1.0, help="scales the size of the depth map visualization, as a percentage of the input size (default is 1.0)")
    depth_parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
    depth_parser.add_argument("--colormap", type=str, default="inferno", help="colormap to use for visualization (default is 'viridis-inverted')",
                                  choices=["inferno", "inferno-inverted", "magma", "magma-inverted", "parula", "parula-inverted", 
                                           "plasma", "plasma-inverted", "turbo", "turbo-inverted", "viridis", "viridis-inverted"])
                                               
    return detect_parser,depth_parser



"""
Takes list of objects and generates the direction
final_list of_objects has fllowing components in order
TODO: Efficient way is to use dictionary
{
    ObjectName
    Left
    RIght
    Depth
}
"""
def generateDirection(final_list_of_objects):

   dir=""
   oppDir=""
   #get the orientation of the final object takes left and right pixel of the detected object
   pos=getOrientation(final_list_of_objects[0][1],final_list_of_objects[0][2])
   dir, oppDir = getTheActualDirection(pos)
   sentence = buildTheSenetenceForMainObject(final_list_of_objects, dir, oppDir)
   print(sentence)

   #setup the gtts for audio generatopm
   language, unique_filename = saveAndPlayAudio_MainObject(sentence)
   
   #do this only when more than one object is recognized
   if(len(final_list_of_objects[1:])):
        sentence = buildOtherObjectSentence(final_list_of_objects)
        saveAndPlayAudio_OtherObject(sentence, language, unique_filename)



def buildOtherObjectSentence(final_list_of_objects):
    sentence=" and other objects like "
    for i in final_list_of_objects[1:]:
        sentence=sentence+i[0]+" at " + str(i[3])+" cm. "
    return sentence



def saveAndPlayAudio_OtherObject(sentence, language, unique_filename):
    print(sentence)
    obj= gTTS(text=sentence, lang=language, slow=False)
    obj.save("OtherObjectAudio"+unique_filename+".mp3")
    playsound("OtherObjectAudio"+unique_filename+".mp3")




def saveAndPlayAudio_MainObject(sentence):
    language="en"
    obj= gTTS(text=sentence, lang=language, slow=False)
    unique_filename = str(uuid.uuid4())
    obj.save("MainObject"+unique_filename+".mp3")
    playsound("MainObject"+unique_filename+".mp3")
    return language,unique_filename




def buildTheSenetenceForMainObject(final_list_of_objects, dir, oppDir):
    sentence=final_list_of_objects[0][0]+" " + "on your " + oppDir+ " at " + str(final_list_of_objects[0][3]) +" cm. "+" Move to the " + dir
    return sentence



def getTheActualDirection(pos):
    if pos==0:
         dir="Right"
         oppDir="Left"
    if pos==1:
         dir="Left"
         oppDir="Right"
    return dir,oppDir



def getOrientation(left, right):
    x1=left
    x2=640-right

    if x2>x1:#obj at lhs
        return 0
    else:#obj at rhs
        return 1


def parse_arguments(detect_parser, depth_parser):
    try:
     detect_args = detect_parser.parse_known_args()[0]
    except:
     detect_parser.print_help()
     sys.exit(0)

    try:
     depth_args = depth_parser.parse_known_args()[0]
    except:
     depth_parser.print_help()
     sys.exit(0)
    return detect_args,depth_args



# load the segmentation networksentence
def init_depth_params(depth_args):
    depth_net_object = depthNet(depth_args.network, sys.argv)
    depth_field = depth_net_object.GetDepthField()
    # create buffer manager
    buffers = depthBuffers(depth_args)
    return depth_net_object,depth_field,buffers



# load the object detection network
def init_detect_params(detect_args):
    detect_net_object = detectNet(detect_args.network, sys.argv, detect_args.threshold)
    return detect_net_object




def run_depth(depth_args, depth_net_object, buffers, img_input):
    buffers.Alloc(img_input.shape, img_input.format)

    	# process the mono depth and visualize
    depth_net_object.Process(img_input, buffers.depth, depth_args.colormap, depth_args.filter_mode)

    	# composite the images
    if buffers.use_input:
       cudaOverlay(img_input, buffers.composite, 0, 0)
        
    if buffers.use_depth:
       cudaOverlay(buffers.depth, buffers.composite, img_input.width if buffers.use_input else 0, 0)



def get_the_actual_depth(depth_net_object, depth_field, detection, img_input):
    scale_x = float(depth_net_object.GetDepthFieldWidth()) / float(img_input.width)
    scale_y = float(depth_net_object.GetDepthFieldHeight()) / float(img_input.height)
    u=detection.Center[0]
    v=detection.Center[1]
    depth_x = int(scale_x * u)
    depth_y = int(scale_y * v)
    d= depth_field[depth_x ,depth_y]
        #image caliberation with multiple detection
    dn= int((d*10)*2.2)
    return u,v,dn



def display(depth_net_object, detection, img_input, u, v, dn):
    print("-----")
    print("ID is", detection.ClassID)
    xyz= depth_net_object.GetClassLabel(detection.ClassID)
    print(xyz)
    print("U is ",u)
    print("V is ",v)
    print("Depth: ",dn, "cm")
    print(detection)
    print(detection.Center)
    print("===============")
    print("W ",float(img_input.width))
    print("H ",float(img_input.height))
    return xyz



def show_modified_image(buffers, img):
    try:
        #show the detection window on the screen
        cv2.imshow('Detection video',cudaToNumpy(img))
        #show the depth map window on the screen
        cv2.imshow('DepthMap video',cudaToNumpy(buffers.composite))
    except:
        #currently no exception is caught
        pass




def iterate_frames(detect_args, depth_args, depth_net_object, depth_field, buffers, cap, detect_net_object):
    while True:
        # capture the next image
        frame=cap.read()

        #converting images to readable format
        img = cudaFromNumpy(frame)

        if img is None: # timeout
            continue  
        
        # detect objects in the image (with overlay)
        detections = detect_net_object.Detect(img, overlay=detect_args.overlay)

        # print the detections
        print("detected {:d} objects in image".format(len(detections)))

        list_of_objects=[]

        for detection in detections:
            #reread the original image not the modified img
            img_input = cudaFromNumpy(frame)
            if img_input is None:
                continue
    	
    	    # allocate buffers for this size image
            run_depth(depth_args, depth_net_object, buffers, img_input)
	

    	    #added
    	    #lets say we are trying to get the depth values for X - 500 and y 640
            u, v, depth = get_the_actual_depth(depth_net_object, depth_field, detection, img_input)


            name_of_detected_object = display(depth_net_object, detection, img_input, u, v, depth)

            list_of_objects.append([name_of_detected_object,detection.Left, detection.Right,depth])

        #sort the all the detected objects based on the depth. The nearest object should be in the first.
        final_list_of_objects=sorted(list_of_objects,key=lambda x: x[-1])

        #do this in try except block as the object show will fail if the cuda memory exceeds and the image formation fails
        show_modified_image(buffers, img)

        if(len(final_list_of_objects)>0): 	
            generateDirection(final_list_of_objects)
    
        #wait for the n key before processing next image or the frame as we cannot process data unless blind person commands it
        key=cv2.waitKey(0)	
        if key==ord('n'):
           continue
        #key to stop the whole code
        elif key==ord('q'):
           break

######################################################################### Main code ###################################################################################################################
  
if __name__=="__main__":
    detect_parser, depth_parser = generate_args()
    detect_args, depth_args = parse_arguments(detect_parser, depth_parser)
    depth_net_object, depth_field, buffers = init_depth_params(depth_args)
    detect_net_object = init_detect_params(detect_args)
    # create video sources and outputs
    cap = VideoCapture("udp://127.0.0.1:10000")
    iterate_frames(detect_args, depth_args, depth_net_object, depth_field, buffers, cap, detect_net_object)
    #destroy all image window on exit and release the capture
    cap.release()
    cv2.destroyAllWindows() 






 
