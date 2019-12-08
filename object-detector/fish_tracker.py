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

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')



sys.path.append('../../../models/research/object_detection') # point to the tensorflow dir
#sys.path.append('~/Documents/Tensorflow/models/slim')

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

# cut the video and save the trimmed video
"""
def cut_vid(from_path, start, end):
  from_path_cutted = from_path[0:-4]+"-cutted.mkv"
  ffmpeg_extract_subclip(from_path, float(start), float(end), targetname=from_path_cutted)
  return from_path_cutted
"""

# Track the fish in the video
def tracking(vid, name,  path_to):

  frame_width = int(vid.get(3))
  frame_height = int(vid.get(4))

  out = cv2.VideoWriter(path_to + name, cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

  while(vid.isOpened()):
    # Read a single frame
    ret, frame = vid.read()

    # Convert colors to RGB
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(color_frame, axis=0)

    # Actual detection for the single frame
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

    vis_util.visualize_boxes_and_labels_on_image_array(
      color_frame,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)

    cv2.imshow('frame', color_frame)
    output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

    # Save the video
    out.write(output_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  out.release()
  vid.release()

  cv2.destroyAllWindows()


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

    # check the start time for cutting the video
    if args.get("start", True):
      start = args["start"]
    else:
      start = 0

    # check the end time for trimming the video
    vid = read_vid(path_from)
    time.sleep(2.0)
    if args.get("end", True):
      end = args["end"]*1000
    else:
      end = vid.get(cv2.CAP_PROP_POS_MSEC)

    """
    if (start!=0) or (end!=vid.get(cv2.CAP_PROP_POS_MSEC)):
      path_from = cut_vid(path_from, start, end)
      vid = read_vid(path_from)
    """

    name = path_from.split("/")[-1]

    tracking(vid, name, path_to)



