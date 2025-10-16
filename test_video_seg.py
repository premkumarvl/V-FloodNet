import numpy as np
from tqdm import tqdm, trange
import os
import argparse
from glob import glob
import torch
from torch import utils
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from video_module.dataset import Video_DS
from video_module.model import AFB_URR, FeatureBank
from test_image_seg import test_waterseg
import myutils
import cv2
from PIL import Image
from torchvision import transforms as T

torch.set_grad_enabled(False)


def get_args():
    parser = argparse.ArgumentParser(description='V-FloodNet: Water Video Segmentation')
    # ... (other arguments remain the same) ...
    parser.add_argument('--gpu', type=int, default=0, help='GPU card id.')
    parser.add_argument('--budget', type=int, default='250000', help='Max number of features that feature bank can store.')
    parser.add_argument('--viz', action='store_true', default=True, help='Visualize data.')
    parser.add_argument('--model-path', type=str, default='records/video_seg_checkpoint_20200212-001734.pth', help='Path to the checkpoint')
    parser.add_argument('--update-rate', type=float, default=0.1, help='Update Rate.')
    parser.add_argument('--merge-thres', type=float, default=0.95, help='Merging Rate.')
    
    ### MODIFIED: Changed the argument to expect a file path, not a directory
    parser.add_argument('--test-path', type=str, required=True,
                        help='Path to the input video file (e.g., video.mp4)')
                        
    parser.add_argument('--test-name', type=str, required=True,
                        help='A name for this video run (used for output folder)')
    return parser.parse_args()


def main(args, device):
    model = AFB_URR(device, update_bank=True, load_imagenet_params=False)
    model = model.to(device)
    model.eval()

    downsample_size = 480

    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(myutils.gct(), f'Loaded checkpoint {args.model_path}.')
    else:
        print(myutils.gct(), f'No checkpoint found at {args.model_path}')
        raise IOError

    ### MODIFIED: Replaced file-based data loading with OpenCV video capture
    cap = cv2.VideoCapture(args.test_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {args.test_path}")

    # --- First Frame Processing ---
    ret, first_frame_cv2 = cap.read()
    if not ret:
        raise ValueError("Video file is empty or corrupted.")

    # Convert from OpenCV BGR format to PIL RGB format
    first_frame = Image.fromarray(cv2.cvtColor(first_frame_cv2, cv2.COLOR_BGR2RGB))
    first_name = f"frame_{0:05d}"

    # Setup output directories
    out_dir = './output/segs'
    mask_dir = os.path.join(out_dir, args.test_name, 'mask')
    mask_path = os.path.join(mask_dir, first_name + '.png')

    # For the initial segmentation, we need an image file. We'll save a temp file.
    if not os.path.exists(mask_path):
        # temp_first_frame_path = 'temp_first_frame.png'
        first_frame.save(mask_path)
        image_model_path = './records/link_efficientb4_model.pth'
        test_waterseg(image_model_path, mask_path, args.test_name, out_dir, device)
        # os.remove(temp_first_frame_path) # Clean up the temporary file
    print()

    first_mask = myutils.load_image_in_PIL(mask_path, 'P')
    
    # Define a transform to convert PIL images to PyTorch tensors
    transform = T.Compose([T.ToTensor()])
    
    # Manually create tensors for the first frame, which was previously done by the DataLoader
    ori_first_frame = transform(first_frame).unsqueeze(0).to(device)
    first_mask_tensor = myutils.mask_to_onehot(first_mask, myutils.color_palette)
    ori_first_mask = transform(first_mask_tensor).unsqueeze(0).to(device)

    # --- Initialize Feature Bank and Directories ---
    seg_dir = os.path.join(out_dir, args.test_name, 'mask')
    os.makedirs(seg_dir, exist_ok=True)
    if args.viz:
        overlay_dir = os.path.join(out_dir, args.test_name, 'overlay')
        os.makedirs(overlay_dir, exist_ok=True)

    obj_n = first_mask_tensor.shape[2]
    fb = FeatureBank(obj_n, args.budget, device, update_rate=args.update_rate, thres_close=args.merge_thres)
    
    first_frame_resized = TF.resize(ori_first_frame, downsample_size, InterpolationMode.BICUBIC)
    first_mask_resized = TF.resize(ori_first_mask, downsample_size, InterpolationMode.NEAREST)

    with torch.no_grad():
        k4_list, v4_list = model.memorize(first_frame_resized, first_mask_resized)
        fb.init_bank(k4_list, v4_list)

    print(myutils.gct(), "Feature bank initialized. Starting video processing...")
    
    ### MODIFIED: Replaced DataLoader loop with a `while` loop for video frames
    frame_idx = 1 # Start from the second frame
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    
    while True:
        ret, frame_cv2 = cap.read()
        if not ret:
            break # End of video
            
        frame_pil = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        ori_frame = transform(frame_pil).unsqueeze(0).to(device)
        frame_name = f"frame_{frame_idx:05d}"
        
        # --- This is the original processing logic from your for-loop ---
        with torch.no_grad():
            ori_size = ori_frame.shape[-2:]
            frame = TF.resize(ori_frame, downsample_size, InterpolationMode.BICUBIC)
            score, _ = model.segment(frame, fb)
            pred_mask = F.softmax(score, dim=1)

            k4_list, v4_list = model.memorize(frame, pred_mask)
            fb.update(k4_list, v4_list, frame_idx)

            pred = TF.resize(pred_mask, ori_size, InterpolationMode.BICUBIC)
            pred = torch.argmax(pred[0], dim=0).cpu().numpy().astype(np.uint8)
            pred = myutils.postprocessing_pred(pred)
            
            seg_path = os.path.join(seg_dir, f'{frame_name}.png')
            myutils.save_seg_mask(pred, seg_path, myutils.color_palette)
            
            if args.viz:
                overlay_path = os.path.join(overlay_dir, f'{frame_name}.png')
                myutils.save_overlay(ori_frame[0], pred, overlay_path, myutils.color_palette)
        
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    fb.print_peak_mem()


if __name__ == '__main__':
    args = get_args()
    print(myutils.gct(), 'Args =', args)

    # --- Automatic Device Detection (CUDA > MPS > CPU) ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
        print(f"✅ Using NVIDIA GPU (CUDA): cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple Silicon GPU (MPS)")
    else:
        raise ValueError('CUDA or MPS is required')

    # Check if the video file exists
    assert os.path.isfile(args.test_path), f"Video file not found at: {args.test_path}"

    # Run the main function with the detected device
    main(args, device)

    print(myutils.gct(), 'Test video segmentation done.')