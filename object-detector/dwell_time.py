import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
import glob
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import argparse
import cv2
import time
import imutils
import pickle

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

sys.path.append('../../../models/research/object_detection')  # point to the tensorflow dir
# sys.path.append('~/Documents/Tensorflow/models/slim')

from utils import label_map_util
from utils import visualization_utils as vis_util

# Model name
MODEL_NAME = '../trained-inference-graphs/output_inference_graph_v1.pb'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '../annotations/label_map.pbtxt'

# Load the frozen Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading the label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Detection for a single image
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


# Read the original video
def read_vid(from_path):
    return cv2.VideoCapture(path_from)


def identify_position(w,h, center):
    #NW
    if (center[0] < (w*2/5)) and (center[1]<h/2):
        return 1
    #SW
    elif (center[0] < (w*2/5)) and (center[1]>h/2):
        return 2
    #NE
    elif (center[0]>(w*3/5)) and (center[1]<h/2):
        return 4
    #SE
    elif (center[0]>(w*3/5)) and (center[1]>h/2):
        return 5
    #main
    else:
        return 3

def identify_transit(previous, current):
    maze_part = [1, 2, 4 ,5]
    if (previous!=current) and (current in maze_part):
        return current
    return 0

# Track the fish in the video
def tracking(vid, name, path_to):
    dwell = []
    passes = []
    previous = 0
    timestamp = 0

    #positions are 1:NW 2:SW 3:main 4:NE 5:SE
    possible_positions = [1,2,3,4,5]

    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Find the fish every 60 frames
    count = 60

    while (vid.isOpened()):
        # Read a single frame
        ret, frame = vid.read()

        if ret:
            # Convert colors to RGB
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(color_frame, axis=0)

            if count % 60 == 0:

                # Actual detection for the single frame
                output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
                if output_dict['detection_scores'][0] > 0.9:
                    # Determine the center of bordering box
                    ymin, xmin, ymax, xmax = output_dict['detection_boxes'][0]
                    center = (int((xmax + xmin) * width / 2), int((ymax + ymin) * height / 2))

                    count += 1
                    timestamp_ = vid.get(cv2.CAP_PROP_POS_MSEC)
                    delta_time = timestamp_ - timestamp
                    position = identify_position(width, height, center)
                    if previous == 0:
                        dwell.append((position, delta_time))
                    else:
                        dwell.append((previous, delta_time))

                    pass_ = identify_transit(previous, position)
                    if (pass_ in possible_positions) and (previous!=0):
                        passes.append((pass_, timestamp_))
                    print(position, pass_, timestamp_)
                    timestamp = timestamp_
                    previous = position
                else:
                    count += 1

            else:
                count += 1
                #timestamp = vid.get(cv2.CAP_PROP_POS_MSEC)
        else:
            break
    vid.release()

    #cv2.destroyWindow('frame')

    return dwell, passes

def dump_file(path, name, file):
    with open(path+name[:-4], 'wb') as pathto:
        pickle.dump(file, pathto)

if __name__ == '__main__':
    # construct the needed arguments
    ap = argparse.ArgumentParser(description="Converting video(s) to images")
    ap.add_argument("-v", "--video", help="path to a single video file")
    ap.add_argument("-d", "--directory", help="path to a directory including video(s)")
    ap.add_argument("-s", "--save", help="path to a directory to save the output images")
    ap.add_argument("--start", help="Start point to trim the video")
    ap.add_argument("--end", help="End point to trim the video")
    args = vars(ap.parse_args())

    # handle wrong arguments
    if (args.get("video", True)) and (args.get("directory", True)):
        raise ValueError("Use either --video or --directory, not both of them.")
    elif not args.get("save", True):
        raise ValueError("Use --save flag to specify a directory to save the output images")
    elif args.get("video", True):
        arg_type = "video"
        path_from = args["video"]
        path_to = args["save"]
    elif args.get("directory", True):
        arg_type = "directory"
        path_from = args["directory"]
        path_to = args["save"]
    else:
        raise ValueError("use --video or --directory flag with a following valid path.")

    # place a '/' at the end of the path_to if it doesn't have it
    if not path_to[-1] == "/":
        path_to += "/"
    if not path_to[-1] == "/":
        path_to += "/"

    try:
        if arg_type == "video":
            name = path_from.split("/")[-1]
            vid = read_vid(path_from)
            dwell, passes = tracking(vid, name, path_to)
            dump_file(path_to, "dwell/"+name, dwell)
            dump_file(path_to, "passes/"+name, passes)
        elif arg_type == "directory":
            videos = glob.glob(path_from + "*")
            for video in videos:
                name = video.split("/")[-1]
                vid = cv2.VideoCapture(video)
                dwell, passes = tracking(vid, name, path_to)
                dump_file(path_to,"dwell/"+name, dwell)
                dump_file(path_to, "passes/"+name, passes)
    except Exception as e:
        print(e)



