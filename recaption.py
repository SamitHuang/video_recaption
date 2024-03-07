import cv2
import argparse
import os
from PIL import Image
import torch
import open_clip


def extract_mid_frame(video_path, output_image=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the mid-frame index
    mid_frame_index = total_frames // 2

    # Set the current frame position to the mid-frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)

    # Read the mid-frame
    ret, frame = cap.read()
    if ret:
        middle_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        middle_frame = Image.fromarray(middle_frame)

        if output_image is not None:
        # Save the frame as an image
            cv2.imwrite(output_image, frame)
    else:
        print("Error reading the mid frame")

    # Release the video capture object
    cap.release()

    return middle_frame


class VideoCaptioner():
    def __init__(self, model_name: str='coca', model_path: str="models/coca/open_clip_pytorch_model.bin"):
        if model_name == 'coca': 
            if not os.path.exists(model_path):
                pretrained = "mscoco_finetuned_laion2B-s13B-b90k"
                print(f"WARNING: {model_path} not exist. Model will be donwloeded from HF (network issue may incur!).")
            else:
                pretrained = model_path
            self.model, _, self.transform = open_clip.create_model_and_transforms(
	      model_name="coca_ViT-L-14",
              pretrained=pretrained,
	    )
        else:
            raise NotImplementedError

    def __call__(self, video_path: str, repetition_penalty: float=1.0, seq_len: int=30):
        middle_frame = extract_mid_frame(video_path)

        img = self.transform(middle_frame).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
              generated = self.model.generate(img)

        caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

        return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", default="moscow.mp4", type=str, help="data path")
    args = parser.parse_args()

    vc = VideoCaptioner('coca')
    video_path = args.video_path
    caption = vc(video_path)

    print(caption)
