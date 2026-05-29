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

decord.bridge.set_bridge('torch') 

class Embedding_Space(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=4, num_frames=21, height=224, width=448):
        self.path = natsorted(p for p in glob.glob(os.path.join(base_path, 'train', "f*", "scene*", "videos", "*.mp4")))

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def process_frames(self, frames):
        T, H, W, C = frames.shape
        
        frames = frames.permute(0, 3, 1, 2) 
        orig_h, orig_w = frames[0].shape[1], frames[0].shape[2]
        scale = max(self.width / orig_w, self.height / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        processed_frames = []
        
        for i in range(T):
            frame = frames[i]  
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0),  
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0) 
            
            start_y = (new_h - self.height) // 2
            start_x = (new_w - self.width) // 2
            frame_cropped = frame[:, start_y:start_y+self.height, start_x:start_x+self.width]
            
            if frame_cropped.max() > 1.0:
                frame_cropped = frame_cropped / 255.0
            frame_cropped = (frame_cropped - self.mean) / self.std
            
            processed_frames.append(frame_cropped)
        
        frames = torch.stack(processed_frames, dim=0)
        
        return frames

    def load_video(self, file_path):
        vr = decord.VideoReader(file_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        
        frame_indices = list(range(self.num_frames))
        frame_indices = [min(idx * self.frame_interval, total_frames - 1) for idx in frame_indices]
        
        frames = vr.get_batch(frame_indices) 
        
        frames = self.process_frames(frames)
        
        return frames

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
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses
    
    def __len__(self):
        return len(self.path)

    def __getitem__(self, data_id):
        
        video_path = self.path[data_id]
        base_path = video_path.rsplit('/', 2)[0]
        caption_path = os.path.join(base_path, 'merged_conditions.json')

        match = re.search(r'cam(\d+)', video_path)
        idx = int(match.group(1))

        video = self.load_video(video_path)

        with open(caption_path, 'r') as file:
            caption_data = json.load(file)[f"cam{idx:02d}"]

        camera_extrinsic = caption_data['extrinsics']
        cam_caption_long = caption_data['captions']['long']
        cam_caption_short = caption_data['captions']['short']

        cam_idx = list(range(self.max_num_frames))[::self.frame_interval]
        traj = [self.parse_matrix(camera_extrinsic[f"frame{idx}"]) for idx in cam_idx]
        traj = np.stack(traj).transpose(0, 2, 1)
        c2ws = []
        for c2w in traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 100
            c2ws.append(c2w)

        cam_params = [Camera(cam_param) for cam_param in c2ws]
        poses = []
        for i in range(len(cam_params)):
            relative_pose = self.get_relative_pose([cam_params[0], cam_params[i]])
            poses.append(torch.as_tensor(relative_pose)[1])

        pose_embedding = torch.stack(poses, dim=0) 

        if random.random() < 0.5:
            cam_caption = cam_caption_long
        else:
            cam_caption = cam_caption_short

        pose_embedding = pose_embedding[:, :3, :] 
        pose_embedding = rearrange(pose_embedding, 't c d -> t (c d)') 

        data = {"video": video, "idx":data_id}
        data['cam_caption'] = cam_caption
        data['pose_embedding'] = pose_embedding
        return data

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

