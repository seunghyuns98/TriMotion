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

decord.bridge.set_bridge('torch')


def crop_and_resize(image, target_width, target_height):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    return torchvision.transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )


def load_video_for_dit(file_path, num_frames, height, width):
    frame_process = v2.Compose([
        v2.CenterCrop(size=(height, width)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_process = v2.CenterCrop(size=(height, width))

    vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=4)
    total = len(vr)
    indices = [min(i, total - 1) for i in range(num_frames)]
    raw_frames = vr.get_batch(indices).cpu().numpy()  # T H W C (uint8)

    frames, first_frame = [], None
    for f in raw_frames:
        frame = Image.fromarray(f)
        frame = crop_and_resize(frame, width, height)
        if first_frame is None:
            first_frame = np.array(image_process(frame))
        frames.append(frame_process(frame))

    frames = torch.stack(frames, dim=0)  # T C H W
    frames = rearrange(frames, "T C H W -> C T H W") # C T H W
    return frames, first_frame


def load_ref_video_for_cam(file_path, num_frames=21, interval=4, height=224, width=448):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=4)
    total = len(vr)
    indices = [min(i * interval, total - 1) for i in range(num_frames)]
    frames = vr.get_batch(indices)  # T H W C (torch)
    frames = frames.permute(0, 3, 1, 2).float()  # T C H W

    orig_h, orig_w = frames.shape[2], frames.shape[3]
    scale = max(width / orig_w, height / orig_h)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    processed = []
    for i in range(frames.shape[0]):
        f = F.interpolate(frames[i:i+1], size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
        sy, sx = (new_h - height) // 2, (new_w - width) // 2
        f = f[:, sy:sy + height, sx:sx + width]
        if f.max() > 1.0:
            f = f / 255.0
        f = (f - mean) / std
        processed.append(f)
    return torch.stack(processed, dim=0)  # T C H W


class _Camera:
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def _parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def _get_relative_pose(cam_params):
    abs_w2cs = [c.w2c_mat for c in cam_params]
    abs_c2ws = [c.c2w_mat for c in cam_params]
    target_cam_c2w = np.eye(4)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)


def load_ref_pose(file_path, max_num_frames=81, frame_interval=4):
    if file_path.endswith(".npy"):
        arr = np.load(file_path)
        t = torch.from_numpy(arr).float()
    elif file_path.endswith(".pt"):
        t = torch.load(file_path, map_location="cpu").float()
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
        if "extrinsics" in data:
            extrinsics = data["extrinsics"]
        else:
            extrinsics = data
        cam_idx = list(range(max_num_frames))[::frame_interval]
        traj = [_parse_matrix(extrinsics[f"frame{idx}"]) for idx in cam_idx]
        traj = np.stack(traj).transpose(0, 2, 1)

        c2ws = []
        for c2w in traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 100
            c2ws.append(c2w)

        cam_params = [_Camera(c) for c in c2ws]
        poses = []
        for i in range(len(cam_params)):
            rel = _get_relative_pose([cam_params[0], cam_params[i]])
            poses.append(torch.as_tensor(rel)[1])
        t = torch.stack(poses, dim=0)  # (T, 4, 4)
    else:
        raise ValueError(f"Unsupported pose file: {file_path}")

    if t.ndim == 3 and t.shape[-2:] == (4, 4):
        t = t[:, :3, :]
    if t.ndim == 3 and t.shape[-2:] == (3, 4):
        t = rearrange(t, "t c d -> t (c d)")
    return t  # (T, 12)


def load_first_frame_image(file_path, height, width):
    frame = Image.open(file_path).convert("RGB")
    frame = crop_and_resize(frame, width, height)
    frame = v2.CenterCrop(size=(height, width))(frame)
    return np.array(frame)


def parse_args():
    p = argparse.ArgumentParser(description="Single-example multimodal inference")

    # === Input ===
    p.add_argument("--content_video", type=str, required=True, help="Source(content) video path.")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt describing the scene. Accepts a raw string or a path to a .txt file.")
    p.add_argument("--ref_video", type=str, default=None, help="Reference video path (for CAM encoder).")
    p.add_argument("--ref_text", type=str, default=None, help="Text describing target camera motion. Accepts a raw string or a path to a .txt file.")
    p.add_argument("--ref_pose", type=str, default=None, help="Path to camera pose file (.json / .npy / .pt).")
    p.add_argument("--first_frame", type=str, default=None, help="(Optional) Image path for i2v first frame. If not given, first frame of content_video is used.")

    # === Checkpoints ===
    p.add_argument("--text_encoder_path", type=str, default="checkpoint/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    p.add_argument("--image_encoder_path", type=str, default="checkpoint/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    p.add_argument("--vae_path", type=str, default="checkpoint/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    p.add_argument("--dit_path", type=str, default="checkpoint/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    p.add_argument("--trimotion_path", type=str, default="checkpoint/trimotion/trimotion.ckpt")
    p.add_argument("--embedding_space_path", type=str, default="checkpoint/trimotion/embedding_space.ckpt")

    # === Output/Inference Settings ===
    p.add_argument("--output_dir", type=str, default="./results/single_example")
    p.add_argument("--output_name", type=str, default="generated.mp4")
    p.add_argument("--dit_num_frames", type=int, default=81)
    p.add_argument("--dit_height", type=int, default=384)
    p.add_argument("--dit_width", type=int, default=672)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="v2v", choices=["i2v", "v2v"])
    return p.parse_args()


def build_pipeline(args, device):
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_paths = [args.text_encoder_path, args.vae_path, args.dit_path]
    if args.image_encoder_path is not None:
        model_paths.append(args.image_encoder_path)
    model_manager.load_models(model_paths)
    pipe = WanVideoMultimodalPipeline.from_model_manager(model_manager)

    with torch.no_grad():
        orig = pipe.dit.patch_embedding
        new_embed = nn.Conv3d(36, 1536, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        new_weights = torch.cat([orig.weight, orig.weight, orig.weight[:, :4]], dim=1)
        assert new_weights.shape[1] == 36
        new_embed.weight.copy_(new_weights)
        new_embed.bias.copy_(orig.bias)
    pipe.dit.patch_embedding = new_embed

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    pipe.dit.img_emb = MLP(1280, dim)
    for block in pipe.dit.blocks:
        block.cross_attn.k_img = nn.Linear(dim, dim)
        block.cross_attn.v_img = nn.Linear(dim, dim)
        block.cross_attn.norm_k_img = RMSNorm(dim, eps=1e-6)

    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(768, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    pipe.denoising_model().has_image_input = True

    state_dict = torch.load(args.trimotion_path, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)
    print(f"[pipe] loaded {args.trimotion_path}")

    pipe.to(device)
    pipe.to(dtype=torch.bfloat16)
    pipe.eval()
    pipe.device = device
    pipe.torch_dtype = torch.bfloat16
    return pipe


def build_cam_encoder(ckpt_path, device):
    cam_encoder = Cam_Encoder()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']
    if any(k.startswith('cam_encoder.') for k in state.keys()):
        state = {k[len('cam_encoder.'):] if k.startswith('cam_encoder.') else k: v
                 for k, v in state.items()}
    cam_encoder.load_state_dict(state, strict=True)
    cam_encoder.to(device).eval()
    print(f"[cam] loaded {ckpt_path}")
    return cam_encoder


def denormalize_tensor(tensor):
    """(C, T, H, W) normalized → list[PIL.Image]."""
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1, 1).to(tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1, 1).to(tensor.device)
    x = tensor.float() * std + mean
    x = torch.clamp(x, 0, 1)
    x = rearrange(x, "C T H W -> T H W C")
    frames = (x * 255).cpu().numpy().astype(np.uint8)
    return [Image.fromarray(f) for f in frames]


def main():
    args = parse_args()

    if args.prompt.endswith(".txt") and os.path.isfile(args.prompt):
        with open(args.prompt, "r") as f:
            args.prompt = f.read().strip()

    # Require at least one reference modality (video / text caption / pose).
    if args.ref_video is None and args.ref_text is None and args.ref_pose is None:
        raise ValueError("At least one of --ref_video, --ref_text, --ref_pose must be provided.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    content_video, content_first_frame = load_video_for_dit(args.content_video, args.dit_num_frames, args.dit_height, args.dit_width)
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
    if not ref_tags:
        raise ValueError("At least one of --ref_video, --ref_text, --ref_pose must be provided.")

    stem, ext = os.path.splitext(args.output_name)
    output_name = f"{stem}_{'_'.join(ref_tags)}{ext}"

    if args.first_frame is not None:
        first_frame_np = load_first_frame_image(args.first_frame, args.dit_height, args.dit_width)
    else:
        first_frame_np = content_first_frame

    # === Load models ===
    pipe = build_pipeline(args, device)
    cam_encoder = build_cam_encoder(args.embedding_space_path, device)

    # === Encode ===
    with torch.no_grad():
        if ref_video is not None:
            ref_video = ref_video.to(device)
            target_camera = cam_encoder(videos=ref_video, texts=None, poses=None)['video_frame_embeds']
        elif ref_text is not None:
            target_camera = cam_encoder(videos=None, texts=ref_text, poses=None)['text_frame_embeds']
        elif ref_pose is not None:
            ref_pose = ref_pose.to(device)
            target_camera = cam_encoder(videos=None, texts=None, poses=ref_pose)['pose_frame_embeds']
            
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


if __name__ == "__main__":
    main()