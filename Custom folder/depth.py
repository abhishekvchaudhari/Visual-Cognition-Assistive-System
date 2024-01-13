import jetson.inference
import jetson.utils
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log

import numpy as np

# load mono depth network
net = jetson.inference.depthNet()

# depthNet re-uses the same memory for the depth field,
# so you only need to do this once (not every frame)
depth_field = net.GetDepthField()

# cudaToNumpy() will map the depth field cudaImage to numpy
# this mapping is persistent, so you only need to do it once
depth_numpy = jetson.utils.cudaToNumpy(depth_field)

#print(f"depth field resolution is {depth_field.width}x{depth_field.height}, format={depth_field.format})


img = 'couch.jpg'	# assumes you have created an input videoSource stream
input = videoSource(img)
net.Process(input.Capture())
jetson.utils.cudaDeviceSynchronize() # wait for GPU to finish processing, so we can use the results on CPU

# find the min/max values with numpy
min_depth = np.amin(depth_numpy)
max_depth = np.amax(depth_numpy)
print(min_depth)
print(max_depth)
