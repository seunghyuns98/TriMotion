import torch
import decord
from natsort import natsorted
import glob
from functools import lru_cache
import lightning as pl
from model import Cam_Encoder
import numpy as np
import json
import re
from einops import rearrange
import os
import argparse

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True  
torch.set_float32_matmul_precision('high') 

decord.bridge.set_bridge('torch')

EMB_KEYS = (
    "video_global", "video_frame", "pose_global", "pose_frame",
    "text_long_global", "text_long_frame", "text_short_global", "text_short_frame",
)


def get_pt_paths(mp4_path):
    base = mp4_path.replace(".mp4", "_emb.pt").replace("videos", "latent")
    return {"embeds": base}


def run_verify(dataset_path):
    pattern = os.path.join(dataset_path, "train", "f*", "scene*", "videos", "*.mp4")
    mp4_paths = natsorted(glob.glob(pattern))
    to_process = []
    ok_count = 0
    for mp4 in mp4_paths:
        paths = get_pt_paths(mp4)
        embeds_path = paths["embeds"]
        if not os.path.isfile(embeds_path):
            to_process.append(mp4)
            continue
        try:
            data = torch.load(embeds_path, map_location="cpu", weights_only=True)
            if not isinstance(data, dict):
                print(f"Error: {embeds_path} is not a dictionary.")
                to_process.append(mp4)
                continue
            for key in EMB_KEYS:
                if key not in data:
                    print(f"Error: {embeds_path} does not contain {key}.")
                    to_process.append(mp4)
                    break
                t = data[key]
                if not hasattr(t, "shape") or t.numel() == 0:
                    print(f"Error: {embeds_path} contains {key} with shape {t.shape}.")
                    to_process.append(mp4)
                    break
            else:
                ok_count += 1
        except Exception as e:
            print(f"Error: {embeds_path} is not a valid tensor.")
            to_process.append(mp4)
            continue
        ok_count += 1
    print(f"Videos total: {len(mp4_paths)}, OK: {ok_count}, To (re)process: {len(to_process)}")
    return to_process


@lru_cache(maxsize=1024)
def load_json_cached(path):
    with open(path, 'r') as f:
        return json.load(f)

class MotionTripletDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=4, num_frames=21, height=224, width=448, to_process=None):

        if to_process is not None:
            self.path = natsorted(to_process)
            print(f"Using {len(self.path)} videos to (re)process.")
        else:
            self.path = natsorted(glob.glob(os.path.join(base_path, 'train', "f*", "scene*", "videos", "*.mp4")))
        self.max_num_frames = max_num_frames
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.height = height
        self.width = width
        
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
        
        frames = self.process_frames(frames, height, width, mean, std)
        
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
        ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        return np.array(ret_poses, dtype=np.float32)
    

    def __len__(self):
        return len(self.path)

    def __getitem__(self, data_id):

        video_path = self.path[data_id]
        video = self.load_video(video_path, self.frame_interval, self.num_frames, self.height, self.width, self.mean_embed, self.std_embed)
        base_path = video_path.rsplit('/', 2)[0]
        caption_data = load_json_cached(os.path.join(base_path, 'merged_conditions.json'))
        
        cam_idx = int(re.search(r'cam(\d+)', video_path).group(1))
        cam_data = caption_data[f"cam{cam_idx:02d}"]

        cam_idx = list(range(self.max_num_frames))[::self.frame_interval]
        traj = np.stack([
            self.parse_matrix(cam_data['extrinsics'][f"frame{idx}"])
            for idx in cam_idx
        ]).transpose(0, 2, 1)
        
        c2ws = []
        for c2w in traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 100
            c2ws.append(c2w)
        
        cam_params = [Camera(cam_param) for cam_param in c2ws]
        poses = [
            torch.as_tensor(self.get_relative_pose([cam_params[0], cam_params[i]]))[1]
            for i in range(len(cam_params))
        ]
        pose_embedding = torch.stack(poses, dim=0)  
        pose_embedding = rearrange(pose_embedding[:, :3, :], 't c d -> t (c d)')

        camera_caption_long = cam_data['captions']['long'] 
        camera_caption_short = cam_data['captions']['short']

        return {"video": video, "pose_embedding": pose_embedding, "camera_caption_long": camera_caption_long, "camera_caption_short": camera_caption_short, "video_path": video_path}

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)




class LightningModelForLatentPreprocess(pl.LightningModule):
    def __init__(
        self,
        cam_encoder_ckpt_path=None,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        output_dim=768,
        text_max_length=512,
        num_frames=21,
        vggt_ckpt_path=None,
    ):
        super().__init__()

        self.cam_encoder = Cam_Encoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            output_dim=output_dim,
            text_max_length=text_max_length,
            num_frames=num_frames,
            vggt_ckpt_path=vggt_ckpt_path)

        assert cam_encoder_ckpt_path is not None, "CAM encoder checkpoint path is required"
        state_dict = torch.load(cam_encoder_ckpt_path, map_location="cpu", weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        if any(key.startswith('cam_encoder.') for key in state_dict.keys()):
            state_dict = {k[len('cam_encoder.'):]: v for k, v in state_dict.items() if k.startswith('cam_encoder.')}
        
        self.cam_encoder.load_state_dict(state_dict, strict=True)
        print(f"Loaded Shared Embedding Space from {cam_encoder_ckpt_path}")

        self.freeze_parameters()

        
    def freeze_parameters(self):
        self.cam_encoder.requires_grad_(False)
        self.cam_encoder.eval()

    def test_step(self, batch, batch_idx):
        video_path = batch["video_path"]
        video = batch["video"].to(self.device)
        pose_embedding = batch["pose_embedding"].to(self.device)
        camera_caption_long = batch["camera_caption_long"]
        camera_caption_short = batch["camera_caption_short"]

        with torch.no_grad():
            self.cam_encoder.to(self.device)
            results = self.cam_encoder(videos=video, poses=pose_embedding, texts=camera_caption_long)
            results_short = self.cam_encoder(texts=camera_caption_short)
            video_embed_global = results['video_global_embeds']
            video_embed_frame = results['video_frame_embeds']
            pose_embed_global = results['pose_global_embeds']
            pose_embed_frame = results['pose_frame_embeds']
            text_embed_global = results['text_global_embeds']
            text_embed_frame = results['text_frame_embeds']
            text_embed_short_global = results_short['text_global_embeds']
            text_embed_short_frame = results_short['text_frame_embeds']
            
            for idx in range(len(video_path)):
                output_path = video_path[idx].replace(".mp4", "_emb.pt").replace("videos", "latent")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                embeds = {
                    "video_global": video_embed_global[idx].cpu(),
                    "video_frame": video_embed_frame[idx].cpu(),
                    "pose_global": pose_embed_global[idx].cpu(),
                    "pose_frame": pose_embed_frame[idx].cpu(),
                    "text_long_global": text_embed_global[idx].cpu(),
                    "text_long_frame": text_embed_frame[idx].cpu(),
                    "text_short_global": text_embed_short_global[idx].cpu(),
                    "text_short_frame": text_embed_short_frame[idx].cpu(),
                }
                torch.save(embeds, output_path)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent Preprocess")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./MotionTriplet-Dataset",
        help="The path of the Dataset.",
    )

    parser.add_argument(
        "--cam_encoder_ckpt_path",
        type=str,
        default="./checkpoint/trimotion/embedding_space.ckpt",
        required=True,
        help="Path of Shared Embedding Space.",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=81,
        help="Maximum number of frames.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=4,
        help="Frame interval.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=21,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=448,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Prefetch factor.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="latent",
        help="Output path.",
    )
    parser.add_argument(
        "--check_load",
        action="store_true",
        help="Check if the embeds are already processed.",
    )

    args = parser.parse_args()
    return args
    
    
def train(args, to_process=None):
    dataset = MotionTripletDataset(
        args.dataset_path,
        max_num_frames=args.max_num_frames,
        frame_interval=args.frame_interval,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        to_process=to_process,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    model = LightningModelForLatentPreprocess(
        cam_encoder_ckpt_path=args.cam_encoder_ckpt_path,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)

if __name__ == '__main__':
    args = parse_args()
    print("Processing Motion Triplet Dataset...")
    if args.check_load:
        print("Checking load...")
        to_process = run_verify(args.dataset_path, check_load=args.check_load)
        if not to_process:
            print("All OK: nothing to (re)process. Exiting.")
            exit(0)
        print(f"Running preprocess for {len(to_process)} videos.")
        train(args, to_process=to_process)
    else:
        train(args)
