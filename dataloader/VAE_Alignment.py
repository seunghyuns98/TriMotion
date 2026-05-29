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
from packaging import version as pver
from functools import lru_cache

decord.bridge.set_bridge('torch')

class VAE_Alignment(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832):

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
        
        for i in range(T):
            frame = frames[i]
            
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
            ).squeeze(0)
            
            start_y, start_x = (new_h - height) // 2, (new_w - width) // 2
            frame_cropped = frame[:, start_y:start_y+height, start_x:start_x+width]
                       
            if frame_cropped.max() > 1.0:
                frame_cropped = frame_cropped / 255.0
            frame_cropped = (frame_cropped - mean) / std
            processed_frames.append(frame_cropped)
        
        return torch.stack(processed_frames, dim=0)

    def load_video(self, file_path, interval, num_frames, height, width, mean, std):
        vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=4)
        total_frames = len(vr)
        
        frame_indices = [min(idx * interval, total_frames - 1) for idx in range(num_frames)]
        frames = vr.get_batch(frame_indices)
        
        frames= self.process_frames(frames, height, width, mean, std)
        
        return frames

    def __len__(self):
        return len(self.path) 

    def __getitem__(self, data_id):

        video_path = self.path[data_id]
        vae_input_video = self.load_video(video_path, self.frame_interval, self.num_frames, self.height, self.width, self.mean_latent, self.std_latent)
        emb_input_video = self.load_video(video_path, 4, 21, 224, 448, self.mean_embed, self.std_embed)

        vae_input_video = rearrange(vae_input_video, "T C H W -> C T H W")
        return {"vae_input": vae_input_video, "emb_input": emb_input_video}
