"""
Capture needed images for training/testing the model from video(s)
"""

import glob
import cv2
import argparse

# capture an image from a frame
def get_image_from_frame(vid, vid_name, path_to, count, sec):
    vid.set(cv2.CAP_PROP_POS_MSEC, (sec*1000))
    hasFrame, image = vid.read()
    if hasFrame:
        print(path_to + vid_name + str(count) + ".jpg")
        cv2.imwrite(path_to + vid_name + str(count) + ".jpg", image)
    return hasFrame

# capture images from a video
def get_images_from_video(path_from, path_to):
    vid = cv2.VideoCapture(path_from)
    vid.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    end = vid.get(cv2.CAP_PROP_POS_MSEC)
    sec = 50
    frame_rate = 590
    count = 1
    name = path_from.split("/")[-1]
    hasFrame = get_image_from_frame(vid, name, path_to, count, sec)
    while (hasFrame) and (sec+frame_rate<=(end/1000)):
        count += 1
        sec += frame_rate
        hasFrame = get_image_from_frame(vid, name, path_to, count, sec)

if __name__ == '__main__':
    # construct the needed arguments
    ap = argparse.ArgumentParser(description="Converting video(s) to images")
    ap.add_argument("-v", "--video", help="path to a single video file")
    ap.add_argument("-d", "--directory", help="path to a directory including video(s)")
    ap.add_argument("-s", "--save", help="path to a directory to save the output images")
    args = vars(ap.parse_args())

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

    if not path_to[-1] == "/":
        path_to += "/"
    if not path_to[-1] == "/":
        path_to += "/"

    # Save the images
    try:
        if arg_type == "video":
            get_images_from_video(path_from, path_to)
        elif arg_type == "directory":
            videos = glob.glob(path_from + "*")
            for video in videos:
                get_images_from_video(video, path_to)
    except Exception as e:
        print(e)