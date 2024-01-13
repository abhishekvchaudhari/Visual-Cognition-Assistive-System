import sys
import argparse
import jetson.inference
import jetson.utils
from jetson_inference import depthNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log,cudaFromNumpy
import numpy as np
from depthnet_utils import depthBuffers
import cv2

# parse the command line
parser = argparse.ArgumentParser(description="Mono depth estimation on a video/image stream using depthNet DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=depthNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-mobilenet", help="pre-trained model to load, see below for options")
parser.add_argument("--visualize", type=str, default="input,depth", help="visualization options (can be 'input' 'depth' 'input,depth'")
parser.add_argument("--depth-size", type=float, default=1.0, help="scales the size of the depth map visualization, as a percentage of the input size (default is 1.0)")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--colormap", type=str, default="inferno", help="colormap to use for visualization (default is 'viridis-inverted')",
                                  choices=["inferno", "inferno-inverted", "magma", "magma-inverted", "parula", "parula-inverted", 
                                           "plasma", "plasma-inverted", "turbo", "turbo-inverted", "viridis", "viridis-inverted"])

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the segmentation network
net = depthNet(args.network, sys.argv)
depth_field = net.GetDepthField()
# create buffer manager
buffers = depthBuffers(args)

# create video sources & outputs
input = cv2.VideoCapture("udp://127.0.0.1:10000")
output = videoOutput("output_depth_12_01_2023.mp4")


depth_numpy = jetson.utils.cudaToNumpy(depth_field)
# process frames until EOS or the user exits
while True:
    # capture the next image
      # capture the next image
    ret,frame=input.read()

    img_input = cudaFromNumpy(frame)

   # img_input = input.Capture()
    img = cv2.imread(args.input)

    if img_input is None: # timeout
        continue
        
    # allocate buffers for this size image
    buffers.Alloc(img_input.shape, img_input.format)

    # process the mono depth and visualize
    net.Process(img_input, buffers.depth, args.colormap, args.filter_mode)

    # composite the images
    if buffers.use_input:
        cudaOverlay(img_input, buffers.composite, 0, 0)
        
    if buffers.use_depth:
        cudaOverlay(buffers.depth, buffers.composite, img_input.width if buffers.use_input else 0, 0)

    # render the output image
    output.Render(buffers.composite)

    # update the title bar
    #output.SetStatus("{:s} | {:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    cudaDeviceSynchronize()
    net.PrintProfilerTimes()

    #lets say we are trying to get the depth values for X - 500 and y 640, we would do it like this
    scale_x = float(net.GetDepthFieldWidth()) / float(img_input.width)
    scale_y = float(net.GetDepthFieldHeight()) / float(img_input.height)
    
    print("Scale X:")
    print(scale_x)
    
    print("Scale Y:")
    print(scale_y)
    
    
    print("width and height")
    print(img_input.width)
    print(img_input.height)
    depth_list = []
    height, width, _ = img_input.shape
    
    print("Test")
    
    depth_x = int(scale_x * 320)
    depth_y = int(scale_y * 240)
    d= depth_field[depth_x ,depth_y]
    dn= (d*10)/1.46
    print("Depth: ",dn, "cm")
    #print("Depth y",depth_y )
    cv2.circle(img, (320, 240), 50, (0, 255, 0), thickness=1)
    #cv2.imshow('img',img)
    #cv2.waitkey(0)
    #cv2.destroyAllWindows()
    
    print(type(img_input.shape))
    for x in range(width):
        for y in range(height):
            new_scale_x = int(scale_x * x)
            new_scale_y = int(scale_y * y)
            depth_list.append(depth_field[new_scale_x ,new_scale_y])
            #print(depth_list)

    percent = max(depth_list) * 0.4
    for x in range(width):
        for y in range(height):
            new_scale_x = int(scale_x * x)
            new_scale_y = int(scale_y * y)
            if depth_numpy[new_scale_x ,new_scale_y] < percent: 
                cv2.circle(img, (x, y), 50, (0, 255, 0), thickness=1)

    #if not input.IsStreaming() or not output.IsStreaming():
      #  break
