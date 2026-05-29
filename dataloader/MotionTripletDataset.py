import torch
import os
import decord
import numpy as np
import json
from natsort import natsorted
import glob
import re
import random
from einops import rearrange
from functools import lru_cache

decord.bridge.set_bridge('torch')

@lru_cache(maxsize=1024)
def load_json_cached(path):
    with open(path, 'r') as f:
        return json.load(f)

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class MotionTripletDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832):

        self.base_path = base_path
        self.path = natsorted(glob.glob(os.path.join(base_path, 'train', "f*", "scene*", "videos", "*.mp4")))
        self.max_num_frames = max_num_frames
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.height = height
        self.width = width
        

        self.mean_latent = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
        self.std_latent = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)

        self.mean_embed = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std_embed = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def process_frames(self, frames, height, width, mean, std):
        T, H, W, C = frames.shape
        frames = frames.permute(0, 3, 1, 2)
        
        orig_h, orig_w = frames[0].shape[1], frames[0].shape[2]
        scale = max(width / orig_w, height / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        processed_frames = []
        first_frame_array = None
        
        for i in range(T):
            frame = frames[i]
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
            ).squeeze(0)
            
            start_y, start_x = (new_h - height) // 2, (new_w - width) // 2
            frame_cropped = frame[:, start_y:start_y+height, start_x:start_x+width]
            
            if i == 0:
                if frame_cropped.max() <= 1.0:
                    first_frame_array = (frame_cropped * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                else:
                    first_frame_array = frame_cropped.clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            if frame_cropped.max() > 1.0:
                frame_cropped = frame_cropped / 255.0
            frame_cropped = (frame_cropped - mean) / std
            processed_frames.append(frame_cropped)
        
        return torch.stack(processed_frames, dim=0), first_frame_array

    def load_video(self, file_path, interval, num_frames, height, width, mean, std):
        vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=4)
        total_frames = len(vr)
        
        frame_indices = [min(idx * interval, total_frames - 1) for idx in range(num_frames)]
        frames = vr.get_batch(frame_indices)
        
        frames, first_frame_array = self.process_frames(frames, height, width, mean, std)
        
        return frames, first_frame_array

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        
        cam_to_origin = 0
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        return np.array(ret_poses, dtype=np.float32)

    def __len__(self):
        return len(self.path) 

    def __getitem__(self, data_id):

        video_path = self.path[data_id]

        base_path = video_path.rsplit('/', 2)[0]
        caption_data = load_json_cached(os.path.join(base_path, 'merged_conditions.json'))
        tgt_idx = int(re.search(r'cam(\d+)', video_path).group(1))
        cam_data = caption_data[f"cam{tgt_idx:02d}"]

        src_idx = random.randint(1, 10)
        while src_idx == tgt_idx:
            src_idx = random.randint(1, 10)
        src_video_path = re.sub(r'cam(\d+)', f'cam{src_idx:02}', video_path)

        video, first_frame = self.load_video(video_path, self.frame_interval, self.num_frames, self.height, self.width, self.mean_latent, self.std_latent)
        src_video, _ = self.load_video(src_video_path, self.frame_interval, self.num_frames, self.height, self.width, self.mean_latent, self.std_latent)

        reference_type = random.choice(['video', 'pose', 'text'])
        if reference_type == 'video':
            reference_latents = torch.load(video_path.replace(".mp4", "_emb.pt").replace("videos", "latent"))['video_frame']  
        elif reference_type == 'pose':
            reference_latents = torch.load(video_path.replace(".mp4", "_emb.pt").replace("videos", "latent"))['pose_frame']
        elif reference_type == 'text':
            if random.random() < 0.5:
                reference_latents = torch.load(video_path.replace(".mp4", "_emb.pt").replace("videos", "latent"))['text_long_frame']
            else:
                reference_latents = torch.load(video_path.replace(".mp4", "_emb.pt").replace("videos", "latent"))['text_short_frame']

        reference_global = torch.load(video_path.replace(".mp4", "_emb.pt").replace("videos", "latent"))['video_global']

        video = rearrange(video, "T C H W -> C T H W")
        src_video = rearrange(src_video, "T C H W -> C T H W")
        text = cam_data['captions']['text']

        return {"video": video, "reference_latents": reference_latents, "reference_global": reference_global, "src_video": src_video, "text": text, "first_frame": first_frame}


