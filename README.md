<div align="center">

# TriMotion: Modality-Agnostic Camera Control for Video Generation


*A unified framework for camera-controlled video generation that accepts **video**, **pose**, or **text**, all describing the same camera trajectory, and maps them into a shared motion embedding space.*

</div>

<p align="center">
  <img src="assets/teaser.png" width="90%">
</p>

---

## 📢 News

- **[2026-06]** Our paper is accepted at **ECCV2026**.
- **[2026-04]** Code, checkpoints, and the **Motion Triplet Dataset** are released.

---

## 🔍 Overview

Existing camera-control methods are typically restricted to a single input modality — pose-conditioned methods require precise geometric trajectories, reference-video methods lack explicit control, and text-based methods struggle with temporal consistency. **TriMotion** addresses all three limitations in one framework.

**Key contributions**

1. **Unified Motion Embedding Space** — aligns video, pose, and text in a shared representation via contrastive learning, temporal synchronization, and geometric fidelity regularization.
2. **Motion Triplet Dataset** — 136K synchronized *(video, pose, text)* triplets built on top of the MultiCamVideo Dataset with LLM-generated, geometry-grounded captions.
3. **Latent Motion Consistency** — a Motion Embedding Predictor that enforces trajectory fidelity directly in latent space, avoiding costly pixel-space decoding.

Built on top of [Wan2.1](https://github.com/Wan-Video/Wan2.1) and supports both **I2V** and **V2V** camera-controlled generation.

---

## ⚙️ Installation

```bash
git clone https://github.com/seunghyuns98/TriMotion.git
cd TriMotion
conda create -n TriMotion python=3.10
conda activate TriMotion
pip install -r requirements.txt

#Install Flash Attention
pip install flash_attn==2.8.3 --no-build-isolation
or
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

---

## 📦 Download Pretrained Weights

We use [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) as the diffusion backbone, along with an additional CLIP checkpoint required for the I2V branch.

```bash
# 1) Wan2.1-T2V-1.3B (T5 text encoder, VAE, DiT)
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir checkpoint/Wan2.1-T2V-1.3B

# 2) CLIP image encoder (open-clip-xlm-roberta-large-vit-huge-14)
hf download DeepBeepMeep/Wan2.1 \
    models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --revision 8bee6e003d1d9d31ecb2c75b643e57fa74fb2ad5 \
    --local-dir ./checkpoint/Wan2.1-T2V-1.3B
```

After downloading, the `checkpoint/` directory should look like:

```
checkpoint/
└── Wan2.1-T2V-1.3B/
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── Wan2.1_VAE.pth
    ├── diffusion_pytorch_model.safetensors
    └── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
```

You also need the TriMotion-specific checkpoints, available from our [Google Drive folder](https://drive.google.com/drive/folders/1tQznlZwoSTFRzDhgmikVCGAbDiO6YAVs?usp=sharing). This folder also includes the I2V model from [CamCloneMaster-Wan2.1](https://huggingface.co/KwaiVGI/CamCloneMaster-Wan2.1), which we use as the initialization for our DiT fine-tuning. Download the entire folder with [`gdown`](https://github.com/wkentaro/gdown):

```bash
pip install gdown

# Download the whole TriMotion checkpoint folder into ./checkpoint/trimotion/
gdown --folder https://drive.google.com/drive/folders/1tQznlZwoSTFRzDhgmikVCGAbDiO6YAVs \
      -O ./checkpoint/trimotion
```

> 💡 If `gdown` hits a quota error for large files, re-run the same command — partially downloaded files will be resumed. For very large files you may need `gdown --fuzzy <file-url>` on individual items.

| File | Description |
|---|---|
| `embedding_space.ckpt` | Unified Motion Embedding Space — video / text / pose encoders |
| `vae_projection.ckpt` | Motion Embedding Predictor — latent → motion embedding |
| `trimotion.ckpt` | Wan2.1 fine-tuned DiT — camera-controlled I2V / V2V |
| `i2v_baseline.ckpt` | I2V initialization from [CamCloneMaster-Wan2.1](https://huggingface.co/KwaiVGI/CamCloneMaster-Wan2.1), used as the starting point for DiT fine-tuning |
| `aggregator.ckpt` | VGGT Aggregator weights extracted from [VGGT](https://github.com/facebookresearch/vggt) — initializes the video motion encoder (required for both training and inference) |

---

## 🚀 Inference

### 1️⃣ Single Modality

`demo.py` runs a single-example inference. You must provide a source video (`--content_video`), a scene prompt (`--prompt`), and **at least one** camera reference among `--ref_video` / `--ref_text` / `--ref_pose`.

#### Reference modalities

| Flag | Accepts | Example |
|---|---|---|
| `--ref_video` | `.mp4` / any decord-readable video | `--ref_video path/to/ref.mp4` |
| `--ref_text`  | raw string **or** `.txt` file path | `--ref_text path/to/ref.txt` |
| `--ref_pose`  | `.json` (per-cam extrinsics) / `.npy` / `.pt` | `--ref_pose path/to/ref.json` |

- `--prompt` also accepts either a raw string or a `.txt` file path.
- If you are using examples/src_videos ref_video and prompt should have same number without ext.

#### Other options

- `--mode` — `i2v` or `v2v` (default `v2v`).
- `--first_frame` — optional image for I2V first frame. If omitted, the first frame of `--content_video` is used.
- `--dit_num_frames`, `--dit_height`, `--dit_width` — output shape (default 81 / 384 / 672).
- `--cfg_scale`, `--num_inference_steps`, `--seed` — standard diffusion controls.
- `--output_name` — output filename stem (default `generated.mp4`). The ref-modality tag is appended automatically, e.g. `generated_video.mp4`, `generated_text.mp4`.

#### Examples

Reference video only:
```bash
python demo.py \
    --content_video examples/src_videos/1.mp4 \
    --prompt        examples/prompt/1.txt \
    --ref_video     examples/ref_videos/1.mp4
```

Reference text only:
```bash
python demo.py \
    --content_video examples/src_videos/1.mp4 \
    --prompt        examples/prompt/1.txt \
    --ref_text      examples/ref_texts/cam04.txt 
```

Reference pose only (JSON extrinsics):
```bash
python demo.py \
    --content_video examples/src_videos/1.mp4 \
    --prompt        examples/prompt/1.txt \
    --ref_pose      examples/ref_poses/cam02.json
```

### 🔀 Multi Modality

`demo_multimodal.py` combines **exactly two** reference modalities from `--ref_video` / `--ref_text` / `--ref_pose` by fusing their motion embeddings. Two fusion modes are supported:

- `--type interpolation` — linearly blends the two embeddings: `target = scale · e₀ + (1 − scale) · e₁`. Set the blend with `--scale` (0.0–1.0).
- `--type sequential` — concatenates the two motion sequences in time to form a compound trajectory. Use `--order {video,text,pose}` to pick which provided modality goes first; the other one goes second.

```bash
python demo_multimodal.py \
    --content_video examples/src_videos/2.mp4 \
    --prompt        examples/prompt/2.txt \
    --ref_pose     examples/ref_poses/cam01.json \
    --ref_text     examples/ref_texts/cam05.txt \
    --type          sequential \
    --order         pose \
    --output_dir    ./results/multimodal \
    --mode          v2v
```

#### Examples

Interpolate between a reference video and a reference text (50/50 blend):
```bash
python demo_multimodal.py \
    --content_video examples/src_videos/2.mp4 \
    --prompt        examples/prompt/2.txt \
    --ref_pose     examples/ref_poses/cam01.json \
    --ref_text     examples/ref_texts/cam05.txt \
    --type          interpolation \
    --scale         0.5 \
    --output_dir    ./results/multimodal
```

Sequential composition (text motion first, then pose motion):
```bash
python demo_multimodal.py \
    --content_video examples/src_videos/2.mp4 \
    --prompt        examples/prompt/2.txt \
    --ref_pose     examples/ref_poses/cam01.json \
    --ref_text     examples/ref_texts/cam05.txt \
    --type          sequential \
    --order         pose \
    --output_dir    ./results/multimodal
```

All other flags (`--mode`, `--first_frame`, `--dit_num_frames/height/width`, `--cfg_scale`, `--num_inference_steps`, `--seed`, `--output_name`) behave the same as in `demo.py`.

---

## 🎬 Motion Triplet Dataset

We release the **Motion Triplet Dataset**, built upon the [MultiCamVideo Dataset](https://github.com/KlingAIResearch/ReCamMaster) (136K videos, 13.6K scenes, 40 Unreal Engine 5 environments) by adding geometry-grounded motion descriptions.

**Preparation**

1. Download the **MultiCamVideo Dataset** into the `MotionTriplet-Dataset/` directory.
2. Download our **Motion Descriptions** from [Google Drive](https://drive.google.com/file/d/1k4c7M6ttohMEXq1EVn7PARg0f8M7fLBA/view?usp=sharing).
3. Run the preparation script:

```bash
python preparing_dataset.py
```

**Directory structure**

```
MotionTriplet-Dataset/
├── train/
│   └── f00/
│       └── scene1/
│           ├── cameras/
│           │   ├── camera_extrinsics.json
│           │   ├── text_description_long.json
│           │   └── text_description_short.json
│           ├── videos/
│           │   ├── cam01.mp4
│           │   ├── ...
│           │   └── cam10.mp4
│           ├── text/
│               └── text_description.json   
│           └── merged_conditions.json
```



---

## Training

### Unified Motion Embedding Space

Trains motion encoders for all three modalities with a composite loss: global InfoNCE alignment, temporal synchronization, and geometric fidelity regularization.

```bash
python train_embedding_space.py \
    --output_path ./checkpoint/embedding_space
```

> 💡 Common knobs: `--batch_size` (default `24`), `--learning_rate` (default `1e-4`), `--max_epochs` (default `100`), `--training_strategy` (`deepspeed_stage_1|2|3`), and `--resume_ckpt_path` to continue from a checkpoint.

### Motion Embedding Predictor

Trains the predictor (3D convolutions + temporal Transformer) to estimate motion embeddings from VAE latents, using a dual-granularity cosine similarity loss (global + frame-wise).

```bash
python train_motion_embedding_projector.py \
    --cam_ckpt_path PATH TO YOUR CAM ENCODER WEIGHT \
    --output_path ./checkpoint/motion_embedding_projector
```

> 💡 Common knobs: `--batch_size` (default `8`), `--learning_rate` (default `1e-4`), `--max_epochs` (default `10`), `--training_strategy`, `--resume_ckpt_path`.

### Preprocess Embeddings

Precompute and cache embeddings before training:

```bash
python latent_preprocess.py \
    --cam_encoder_ckpt_path PATH TO YOUR CAM ENCODER WEIGHT
```

> 💡 You may also tune `--num_frames` (default `21`), `--height` / `--width` (default `224` / `448`), `--batch_size` (default `32`), and `--dataloader_num_workers` (default `8`) to match your hardware.


### Diffusion Model Fine-tuning

Fine-tunes WAN-Video with motion embedding conditioning via block-specific projection MLPs. Jointly trains I2V and V2V with equal probability per iteration.

```bash
train_trimotion.py \
    --vae_projector_ckpt_path PATH TO YOUR VAE PROJECTOR WEIGHT \
    --output_path ./checkpoint/tri_motion
```

> 💡 Common knobs: `--batch_size` (default `4`), `--accumulate_grad_batches` (default `4`), `--learning_rate` (default `1e-4`), `--max_epochs` (default `10`), `--num_frames` / `--height` / `--width` (default `81` / `384` / `672`), `--training_strategy` (default `deepspeed_stage_2`), `--resume_ckpt_path`.

---

## Camera Caption Generation

Geometry-grounded camera motion captions are produced with [Qwen3](https://github.com/QwenLM/Qwen3). Shared logic lives in `caption_generator.py`; use the scripts below for each workflow.

| File | Purpose |
|---|---|
| `caption_generator.py` | Pose parsing, motion analysis, prompts, Qwen inference (imported by the scripts) |
| `caption_motion_triplet.py` | Batch captions for the **Motion Triplet Dataset** |
| `caption_pose.py` | Long captions for individual pose files (e.g., **RealEstate10K**) |

Install the extra dependency (not required for inference/training above):

```bash
pip install scipy
```

### Motion Triplet Dataset (`caption_motion_triplet.py`)

After placing [MultiCamVideo](https://github.com/KlingAIResearch/ReCamMaster) under `MotionTriplet-Dataset/`, run caption generation. Outputs are written next to each scene as `cameras/text_description_long.json` and `cameras/text_description_short.json`. Both `camera_extrinsics.json` and merged `merged_conditions.json` are supported.

```bash
# Long + short captions for the train split (default dataset path: ./MotionTriplet-Dataset)
python caption_motion_triplet.py --mode train

# Validation split, long captions only
python caption_motion_triplet.py --mode val --caption-type long
```

| Flag | Description |
|---|---|
| `--dataset-path` | `MotionTriplet-Dataset` root (default: `./MotionTriplet-Dataset`) |
| `--mode` | `train` or `val` |
| `--caption-type` | `long`, `short`, or `both` (default `both`) |
| `--model-name` | Hugging Face Qwen3 id (default `Qwen/Qwen3-4B-Instruct-2507`) |

### RealEstate10K / custom poses (`caption_pose.py`)

Pass one or two pose files directly. Generates **long** captions only. Supports `.txt` trajectories (RealEstate10K-style extrinsics) and `.json` files in the same format as `examples/ref_poses/`.

```bash
# One RealEstate10K pose file
python caption_pose.py \
    --pose /path/to/realestate10k/poses/00001.txt \
    --output-dir ./cam_captions

# Try two example poses from this repo
python caption_pose.py \
    --pose examples/ref_poses/cam01.json examples/ref_poses/cam02.json \
    --output-dir ./cam_captions
```

| Flag | Description |
|---|---|
| `--pose` | One or more pose paths (`.txt` or `.json`) |
| `--output-dir` | Output directory (default `./cam_captions`; files named `{pose}_long.txt`) |
| `--frame-interval` / `--num-frames` | Sampling for `.txt` poses (default 4 / 21) |

---

## Acknowledgements

- [WAN-Video](https://github.com/Wan-Video/Wan2.1) — diffusion backbone
- [VGGT](https://github.com/facebookresearch/vggt) — video motion encoder
- [ReCamMaster](https://github.com/jianhongbai/ReCamMaster) — Multi-Cam Video Dataset
- [CamCloneMaster](https://huggingface.co/KwaiVGI/CamCloneMaster-Wan2.1) — I2V initialization for DiT fine-tuning
- [Qwen3](https://github.com/QwenLM/Qwen3) — geometry-grounded caption generation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — T5 text encoder

---

## 📝 Citation

If you find TriMotion useful in your research, please cite our paper:

```bibtex
@inproceedings{shin2026trimotion,
  title={TriMotion: Modality-Agnostic Camera Control for Video Generation},
  author={Shin, Seunghyun and Song, Jifei and Jeon, Wooseok and Jeon, Hae-Gon and Deng, Jiankang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
