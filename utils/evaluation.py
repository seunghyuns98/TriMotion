from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Literal

_EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_EVAL_ROOT, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from natsort import natsorted


def _load_video_frames(
    video_path: str,
    num_frames: int = 16,
    sample: str = "start",
    target_size: tuple[int, int] | None = None,
) -> Tensor:
    try:
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge("native")
        vr = VideoReader(video_path, ctx=cpu(0))
        vlen = len(vr)
        if sample == "start":
            indices = list(range(min(num_frames, vlen)))
        elif sample == "middle":
            start = max(0, vlen // 2 - num_frames // 2)
            indices = list(range(start, min(start + num_frames, vlen)))
        elif sample == "uniform":
            if num_frames >= vlen:
                indices = list(range(vlen))
            else:
                indices = np.linspace(0, vlen - 1, num_frames, dtype=np.int64).tolist()
        else:
            step = max(1, (vlen - 1) // max(1, num_frames - 1))
            indices = list(range(0, vlen, step))[:num_frames]
        frames = vr.get_batch(indices)
        frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2)
    except Exception:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "frame_%04d.png")
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vframes", str(num_frames), "-y", out],
                capture_output=True,
                check=False,
            )
            from PIL import Image
            paths = sorted(glob.glob(os.path.join(tmp, "frame_*.png")))
            if not paths:
                raise FileNotFoundError(f"Cannot load video: {video_path}")
            frames = []
            for p in paths[:num_frames]:
                img = Image.open(p).convert("RGB")
                frames.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
            frames = torch.stack(frames)
    if target_size:
        frames = F.interpolate(
            frames.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).to(torch.uint8)
    if frames.shape[0] < num_frames:
        from utils.video_frame_interpolation import interpolate_frames_to_length
        frames = interpolate_frames_to_length(frames, num_frames)
    return frames


def _collect_frames_from_paths(
    paths: list[str],
    num_frames: int = 16,
    sample: str = "uniform",
    target_size: tuple[int, int] | None = None,
    batch_size: int = 8,
) -> Tensor:
    all_frames = []
    for path in tqdm(paths, desc="Loading videos"):
        frames = _load_video_frames(path, num_frames=num_frames, sample=sample, target_size=target_size)
        all_frames.append(frames)
    return torch.cat(all_frames, dim=0)


def _ensure_fvdcal(root_path: str | None = None):
    if root_path:
        fvd_path = os.path.join(os.path.abspath(root_path), "utils", "FVD")
        print(fvd_path)
        if os.path.isdir(fvd_path) and fvd_path not in sys.path:
            sys.path.insert(0, fvd_path)
    try:
        from fvdcal import FVDCalculation
        from fvdcal.video_preprocess import load_video
        return FVDCalculation, load_video
    except ImportError as e:
        raise ImportError(
            "CamI2V fvdcal is required for FVD computation. "
            "Pass cami2v_root or add CamI2V/evaluation/FVD to PYTHONPATH."
        ) from e


def compute_fvd(
    gt_folder: str,
    sample_folder: str,
    model_path: str = "FVD/model",
    method: str = "videogpt",
    root_path: str | None = None,
    num_frames: int = 16,
    gt_num_frames: int = 81,
    video_batch_size: int = 32,
    cache_dir: str | None = None,
    max_samples: int | None = None,
    cache_name: str = "gt",
) -> dict[str, float]:
    FVDCalculation, _ = _ensure_fvdcal(root_path)

    gt_paths = natsorted(glob.glob(os.path.join(gt_folder, "*.mp4")))
    sample_paths = natsorted(glob.glob(os.path.join(sample_folder, "*.mp4")))
    if max_samples is not None:
        gt_paths = gt_paths[:max_samples]
        sample_paths = sample_paths[:max_samples]
    if len(gt_paths) != len(sample_paths):
        raise ValueError(f"GT video count ({len(gt_paths)}) and generated video count ({len(sample_paths)}) differ.")

    n = len(gt_paths)
    print(f"GT/generated video count: {n}, video_batch_size={video_batch_size}, cache_dir={cache_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calc = FVDCalculation(method=method)

    def _cache_path(name: str) -> str | None:
        if not cache_dir:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"fvd_feats_{method}_{name}.pt")

    gt_cache = _cache_path(cache_name)
    sample_name = os.path.basename(sample_folder)
    sample_cache = _cache_path(sample_name)

    if gt_cache and os.path.isfile(gt_cache):
        print("Loading GT features from cache...")
        gt_feats = torch.load(gt_cache, map_location="cpu", weights_only=False)
    else:
        gt_feats = calc.extract_features_from_paths(
            gt_paths, model_path, device,
            video_batch_size=video_batch_size,
            num_frames=gt_num_frames,
            sample="start",
        )
        if gt_cache:
            torch.save(gt_feats, gt_cache)
    if sample_cache and os.path.isfile(sample_cache):
        print("Loading generated features from cache...")
        sample_feats = torch.load(sample_cache, map_location="cpu", weights_only=False)
    else:
        print("Extracting I3D features (batch-wise)...")
        sample_feats = calc.extract_features_from_paths(
            sample_paths, model_path, device,
            video_batch_size=video_batch_size,
            num_frames=num_frames,
            sample="start",
        )
        if sample_cache:
            torch.save(sample_feats, sample_cache)
    print(gt_feats.shape, sample_feats.shape)
    print("Computing FVD...")
    fvd_val = calc.compute_fvd_from_features(gt_feats, sample_feats, device)
    return {"FVD_videogpt" if method == "videogpt" else "FVD_stylegan": float(fvd_val.detach().cpu().numpy())}


def _get_inception_feature_extractor(device: torch.device):
    from torchvision import models
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.aux_logits = False
    model.eval()
    return model.to(device)


def _inception_features(frames: Tensor, model: torch.nn.Module, device: torch.device, batch_size: int = 32) -> Tensor:
    N = frames.shape[0]
    frames = frames.float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    frames = F.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)
    feats = []
    for i in range(0, N, batch_size):
        batch = frames[i: i + batch_size].to(device)
        with torch.no_grad():
            out = model(batch)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    try:
        from scipy import linalg
    except ImportError as e:
        raise ImportError("scipy is required for FID. Install with: pip install scipy") from e
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _load_index_names_from_csv(csv_path: str) -> set[str]:
    names = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "Name" not in reader.fieldnames:
            raise ValueError("index_csv must have a 'Name' column")
        for row in reader:
            name_val = row.get("Name", "").strip()
            base = os.path.splitext(name_val)[0] if name_val else ""
            if base:
                names.add(base)
    return names


def compute_fid(
    gt_folder: str,
    sample_folder: str,
    gt_num_frames: int = 16,
    num_frames: int = 16,
    frame_sample: str = "uniform",
    batch_size: int = 32,
    cache_dir: str | None = None,
    max_samples: int | None = None,
    cache_name: str = "gt",
) -> dict[str, float]:
    gt_paths = natsorted(glob.glob(os.path.join(gt_folder, "*.mp4")))
    sample_paths = natsorted(glob.glob(os.path.join(sample_folder, "*.mp4")))
    if max_samples is not None:
        gt_paths = gt_paths[:max_samples]
        sample_paths = sample_paths[:max_samples]
    if len(gt_paths) != len(sample_paths):
        raise ValueError(f"GT video count ({len(gt_paths)}) and generated count ({len(sample_paths)}) differ.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fid_cache_path(name: str) -> str | None:
        if not cache_dir:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"fid_feats_{name}.pt")

    sample_key = os.path.basename(sample_folder.rstrip(os.sep)) or "sample"
    gt_cache = _fid_cache_path(cache_name)
    sample_cache = _fid_cache_path(sample_key)

    if gt_cache and os.path.isfile(gt_cache):
        print("Loading GT FID features from cache...")
        gt_feats = torch.load(gt_cache, map_location="cpu", weights_only=False)
        gt_feats = gt_feats.numpy() if isinstance(gt_feats, torch.Tensor) else gt_feats
    else:
        model = _get_inception_feature_extractor(device)
        print("Extracting GT frame features for FID...")
        gt_frames = _collect_frames_from_paths(gt_paths, num_frames=num_frames, sample=frame_sample, target_size=(299, 299))
        gt_feats = _inception_features(gt_frames, model, device, batch_size=batch_size).numpy()
        if gt_cache:
            torch.save(torch.from_numpy(gt_feats), gt_cache)

    if sample_cache and os.path.isfile(sample_cache):
        print("Loading generated FID features from cache...")
        sample_feats = torch.load(sample_cache, map_location="cpu", weights_only=False)
        sample_feats = sample_feats.numpy() if isinstance(sample_feats, torch.Tensor) else sample_feats
    else:
        model = _get_inception_feature_extractor(device)
        print("Extracting generated frame features for FID...")
        sample_frames = _collect_frames_from_paths(sample_paths, num_frames=num_frames, sample=frame_sample, target_size=(299, 299))
        sample_feats = _inception_features(sample_frames, model, device, batch_size=batch_size).numpy()
        if sample_cache:
            torch.save(torch.from_numpy(sample_feats), sample_cache)

    print(gt_feats.shape, sample_feats.shape)

    mu_gt = np.mean(gt_feats, axis=0)
    sigma_gt = np.cov(gt_feats, rowvar=False)
    mu_sample = np.mean(sample_feats, axis=0)
    sigma_sample = np.cov(sample_feats, rowvar=False)
    fid_val = _frechet_distance(mu_gt, sigma_gt, mu_sample, sigma_sample)
    return {"FID": float(fid_val)}


def _get_clip_model(device: torch.device, model_name: str = "ViT-B-32", pretrained: str = "openai"):
    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "open_clip is required for CLIP-T and CLIP-V. Install with: pip install open-clip-torch"
        ) from e
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def compute_clip_t(
    sample_folder: str,
    text_folder: str,
    num_frames: int = 8,
    frame_sample: str = "middle",
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    batch_size: int = 16,
    max_samples: int | None = None,
    index_csv: str | None = None,
) -> dict[str, float]:
    sample_paths = natsorted(glob.glob(os.path.join(sample_folder, "*.mp4")))
    if index_csv and os.path.isfile(index_csv):
        names_in_csv = _load_index_names_from_csv(index_csv)
        sample_paths = [p for p in sample_paths if os.path.splitext(os.path.basename(p))[0] in names_in_csv]
        print(f"CLIP-T: index_csv filter -> {len(sample_paths)} videos")
    if max_samples is not None:
        sample_paths = sample_paths[:max_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = _get_clip_model(device, clip_model, clip_pretrained)
    from PIL import Image
    scores = []
    for path in tqdm(sample_paths, desc="CLIP-T"):
        stem = os.path.splitext(os.path.basename(path))[0]
        text_path = os.path.join(text_folder, stem + ".txt")
        if not os.path.isfile(text_path):
            raise FileNotFoundError(f"Text file not found for video {path}: {text_path}")
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise ValueError(f"Empty text in {text_path}")

        frames = _load_video_frames(path, num_frames=num_frames, sample=frame_sample)
        frame_embeds = []
        for t in range(frames.shape[0]):
            img = frames[t].permute(1, 2, 0).numpy()
            img_pil = Image.fromarray(img)
            img_t = preprocess(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img_t)
            frame_embeds.append(emb)
        video_emb = torch.cat(frame_embeds, dim=0).mean(dim=0, keepdim=True)
        video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)

        txt_t = tokenizer([text]).to(device)
        with torch.no_grad():
