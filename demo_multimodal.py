import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from einops import rearrange
from PIL import Image
import decord
import json

from diffsynth import ModelManager, WanVideoMultimodalPipeline, save_video
from diffsynth.models.wan_video_dit import MLP, RMSNorm
from model import Cam_Encoder

from demo import (
    load_video_for_dit,
    load_ref_video_for_cam,
    load_ref_pose,
    load_first_frame_image,
    build_pipeline,
    build_cam_encoder,
    denormalize_tensor,
)

decord.bridge.set_bridge('torch')

def parse_args():
    p = argparse.ArgumentParser(
        description="Multimodal inference — combine any subset of (video / text / pose) references via embedding averaging."
    )

    # === Input ===
    p.add_argument("--content_video", type=str, required=True, help="Source(content) video path.")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt describing the scene. Accepts a raw string or a path to a .txt file.")
    p.add_argument("--ref_video", type=str, default=None, help="Reference video path.")
    p.add_argument("--ref_text", type=str, default=None, help="Reference text (raw string or .txt file path).")
    p.add_argument("--ref_pose", type=str, default=None, help="Reference camera pose file (.json / .npy / .pt).")
    p.add_argument("--first_frame", type=str, default=None, help="(Optional) Image path for i2v first frame.")

    # === Modality weights (optional) ===
    p.add_argument("--type", type=str, default="interpolation", choices=["interpolation", "sequential"], help="How to combine the two reference embeddings.")
    p.add_argument("--scale", type=float, default=None, help="[interpolation only] Weight for first embedding: target = scale*e0 + (1-scale)*e1.")
    p.add_argument("--order", type=str, default=None, choices=["video", "text", "pose"], help="[sequential only] Which modality comes first. The remaining provided modality goes second.")

    # === Checkpoints ===
    p.add_argument("--text_encoder_path", type=str, default="checkpoint/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    p.add_argument("--image_encoder_path", type=str, default="checkpoint/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    p.add_argument("--vae_path", type=str, default="checkpoint/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    p.add_argument("--dit_path", type=str, default="checkpoint/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    p.add_argument("--trimotion_path", type=str, default="checkpoint/trimotion/trimotion.ckpt")
    p.add_argument("--embedding_space_path", type=str, default="checkpoint/trimotion/embedding_space.ckpt")

    # === Output/Inference Settings ===
    p.add_argument("--output_dir", type=str, default="./results/multimodal")
    p.add_argument("--output_name", type=str, default="generated.mp4")
    p.add_argument("--dit_num_frames", type=int, default=81)
    p.add_argument("--dit_height", type=int, default=384)
    p.add_argument("--dit_width", type=int, default=672)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="v2v", choices=["i2v", "v2v"])
    return p.parse_args()

def main():
    args = parse_args()

    if args.prompt.endswith(".txt") and os.path.isfile(args.prompt):
        with open(args.prompt, "r") as f:
            args.prompt = f.read().strip()

    provided = {name for name, val in [("video", args.ref_video), ("text", args.ref_text), ("pose", args.ref_pose)] if val is not None}
    assert len(provided) == 2, (
        f"demo_multimodal.py requires EXACTLY 2 reference modalities, got {len(provided)}. "
        "Provide exactly two of: --ref_video, --ref_text, --ref_pose."
    )

    if args.type == "interpolation":
        assert args.scale is not None, "--scale is required when --type interpolation."
        assert 0.0 <= args.scale <= 1.0, f"--scale must be in [0, 1], got {args.scale}."
    elif args.type == "sequential":
        assert args.order is not None, "--order is required when --type sequential (choose which modality comes first)."
        assert args.order in provided, (
            f"--order={args.order!r} is not among the provided references {sorted(provided)}."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # === Load inputs ===
    content_video, content_first_frame = load_video_for_dit(
        args.content_video, args.dit_num_frames, args.dit_height, args.dit_width
    )
    content_video = content_video.unsqueeze(0)  # B C T H W

    ref_video = None
    ref_text = None
    ref_pose = None
    ref_tags = []

    if args.ref_video is not None:
        ref_video = load_ref_video_for_cam(args.ref_video).unsqueeze(0)  # B T C H W
        ref_tags.append("video")
    if args.ref_text is not None:
        if args.ref_text.endswith(".txt") and os.path.isfile(args.ref_text):
            with open(args.ref_text, "r") as f:
                ref_text = f.read().strip()
        else:
            ref_text = args.ref_text
        ref_tags.append("text")
    if args.ref_pose is not None:
        ref_pose = load_ref_pose(args.ref_pose).unsqueeze(0)  # (1, T, 12)
        ref_tags.append("pose")

    stem, ext = os.path.splitext(args.output_name)
    if args.type == "interpolation":
        mix_tag = f"interp{args.scale:g}_{'-'.join(ref_tags)}"
    else:  # sequential
        second = next(t for t in ref_tags if t != args.order)
        mix_tag = f"seq_{args.order}-{second}"
    output_name = f"{stem}_{mix_tag}{ext}"

    if args.first_frame is not None:
        first_frame_np = load_first_frame_image(args.first_frame, args.dit_height, args.dit_width)
    else:
        first_frame_np = content_first_frame

    pipe = build_pipeline(args, device)
    cam_encoder = build_cam_encoder(args.embedding_space_path, device)


    with torch.no_grad():
        embeds = {}
        if ref_video is not None:
            ref_video = ref_video.to(device)
            embeds["video"] = cam_encoder(videos=ref_video, texts=None, poses=None)['video_frame_embeds']
        if ref_text is not None:
            embeds["text"] = cam_encoder(videos=None, texts=ref_text, poses=None)['text_frame_embeds']
        if ref_pose is not None:
            ref_pose = ref_pose.to(device)
            embeds["pose"] = cam_encoder(videos=None, texts=None, poses=ref_pose)['pose_frame_embeds']

        # Cast to pipeline dtype/device
        embeds = {k: v.to(dtype=pipe.torch_dtype, device=pipe.device) for k, v in embeds.items()}

        if args.type == "interpolation":
            e0, e1 = list(embeds.values())  # 2 embeddings, order by insertion (video, text, pose)
            target_camera = torch.nn.functional.normalize(
                args.scale * e0 + (1 - args.scale) * e1, dim=-1
            )
        elif args.type == "sequential":
            # --order picks which modality comes first; the other goes second.
            first_key = args.order
            second_key = next(k for k in embeds if k != first_key)
            front = embeds[first_key][:, ::2]                  # (B, T1, D)
            back = embeds[second_key][:, 1::2]                 # (B, T2, D)
            back = back + front[:, -1:, :]                     # shift back by last feature of front
            target_camera = torch.cat([front, back], dim=1)    # (B, T1+T2, D)
        else:
            raise ValueError(f"Unknown --type: {args.type}")

        source_video = content_video.to(dtype=pipe.torch_dtype, device=device)
        _, _, num_frames, height, width = source_video.shape
        first_frames_pil = [Image.fromarray(first_frame_np)]

        # === Inference ===
        generated_video = pipe(
            prompt=args.prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_video=source_video,
            target_camera=target_camera,
            input_image=first_frames_pil,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            mode=args.mode,
        )

    # === Save ===
    if isinstance(generated_video, list):
        generated_frames = generated_video
    elif isinstance(generated_video, np.ndarray):
        if generated_video.dtype != np.uint8:
            generated_video = (np.clip(generated_video, 0, 1) * 255).astype(np.uint8)
        generated_frames = [Image.fromarray(f) for f in generated_video]
    else:
        generated_frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in generated_video]

    out_path = os.path.join(args.output_dir, output_name)
    save_video(generated_frames, out_path, fps=30, quality=5)
    print(f"[done] saved to {out_path}")

    src_path = os.path.join(args.output_dir, "source.mp4")
    save_video(denormalize_tensor(source_video[0]), src_path, fps=30, quality=5)
    print(f"[done] saved source to {src_path}")

if __name__ == '__main__':
    main()
