import torch, os, argparse
import lightning as pl
import torch.nn.functional as F
from dataloader import VAE_Alignment
from diffsynth import WanVideoMultimodalPipeline, ModelManager

from model import Cam_Encoder, VAE_Projector

# CUDA 최적화 설정
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        vae_path,
        learning_rate=1e-4,
        cam_ckpt_path=None,
        vggt_ckpt_path=None,
        embed_dim=1024,
        output_dim=768,
        patch_size=14,
        text_max_length=512,
        img_size=518,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        weight_decay=0.2, 
        num_frames=21,
        latent_channels=16,
        hidden_dim=512,
        frame_align_weight=0.5,
        noise_timestep_min_ratio=0.7,
    ):
        super().__init__()
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([vae_path])
        self.pipe = WanVideoMultimodalPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.cam_encoder = Cam_Encoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            output_dim=output_dim,
            text_max_length=text_max_length,
            num_frames=num_frames,
            vggt_ckpt_path=vggt_ckpt_path,
        )
        
        assert cam_ckpt_path is not None, "CAM encoder checkpoint path is required"
        state_dict = torch.load(cam_ckpt_path, map_location="cpu", weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        if any(key.startswith('cam_encoder.') for key in state_dict.keys()):
            state_dict = {k[len('cam_encoder.'):]: v for k, v in state_dict.items() if k.startswith('cam_encoder.')}
        
        self.cam_encoder.load_state_dict(state_dict, strict=True)
        print(f"Loaded Shared Embedding Space from {cam_ckpt_path}")
        
        self.vae_projector = VAE_Projector(
            latent_channels=latent_channels,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
        )
        
        self.freeze_encoders()
        
        print("\n=== Trainable Parameters ===")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.numel():,} params")
        print("=" * 40 + "\n")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = total_params - trainable_params
        print(f"Total trainable parameters: {trainable_params:,}")
        print(f"Total frozen parameters: {frozen_params:,}")
        print(f"Total parameters: {total_params:,}")
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.frame_align_weight = frame_align_weight
        self.noise_timestep_min_ratio = noise_timestep_min_ratio

    
    def freeze_encoders(self):
        self.cam_encoder.requires_grad_(False)
        self.cam_encoder.eval()

        self.pipe.requires_grad_(False)
        self.pipe.eval()

        self.vae_projector.requires_grad_(True)
        self.vae_projector.train()
    

    def training_step(self, batch, batch_idx):
        vae_input = batch["vae_input"]
        emb_input = batch["emb_input"]

        if self.pipe.device != self.device:
            self.pipe.device = self.device
        vae_input = vae_input.to(dtype=self.pipe.torch_dtype, device=self.device)
        emb_input = emb_input.to(dtype=self.pipe.torch_dtype, device=self.device)

        with torch.no_grad():
            video_latents = self.pipe.encode_video(vae_input, **self.tiler_kwargs)
            emb_out = self.cam_encoder(videos=emb_input)
            emb_global = emb_out["video_global_embeds"]
            emb_frames = emb_out["video_frame_embeds"]

        video_latents = video_latents.to(dtype=self.pipe.torch_dtype, device=self.device)
        if self.noise_timestep_min_ratio < 1.0:
            N = self.pipe.scheduler.num_train_timesteps
            low = int(N * self.noise_timestep_min_ratio)
            noise = torch.randn_like(video_latents, device=video_latents.device, dtype=video_latents.dtype)
            timestep_id = torch.randint(low, N, (1,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=video_latents.dtype, device=video_latents.device)
            video_latents = self.pipe.scheduler.add_noise(video_latents, noise, timestep)
        proj_out = self.vae_projector(video_latents)
        vae_global = proj_out["video_global_embeds"]
        vae_frames = proj_out["video_frame_embeds"]

        cosine_sim_global = F.cosine_similarity(vae_global, emb_global, dim=-1)
        loss_global = (1 - cosine_sim_global).mean()
        cosine_sim_frame = F.cosine_similarity(vae_frames, emb_frames, dim=-1)
        loss_frame = (1 - cosine_sim_frame).mean()

        loss = loss_global + self.frame_align_weight * loss_frame
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train/loss_global", loss_global, prog_bar=True)
        self.log("train/loss_frame", loss_frame, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_params = list(self.vae_projector.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        print(f"Optimizer configured:")
        print(f"  - Trainable params (VAE Projector): {sum(p.numel() for p in trainable_params):,} params, lr={self.learning_rate:.2e}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def on_save_checkpoint(self, checkpoint):
        if self.trainer.is_global_zero:
            checkpoint_dir = self.trainer.checkpoint_callback.dirpath
            current_step = self.global_step
            print(f"Saving checkpoint at step {current_step} to {checkpoint_dir}")
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
            
            state_dict = self.state_dict()
            torch.save(state_dict, os.path.join(checkpoint_dir, f"state_dict_step{current_step}.ckpt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE Projector")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./MotionTriplet-Dataset",
        help="The path of the Motion Triplet Dataset.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoint/motion_embedding_projector",
        help="Path to save the model.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help="Number of subprocesses to use for data loading.",
    )

    parser.add_argument(
        "--vae_path",
        type=str,
        default="checkpoint/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Number of epochs.",
    )

    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )

    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="Path to resume training checkpoint.",
    )

    parser.add_argument(
        "--cam_ckpt_path",
        type=str,
        default="checkpoint/trimotion/embedding_space.ckpt",
        required=True,
        help="Path to CAM encoder checkpoint.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches prefetched by each worker.",
    )

    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients before optimizer step.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer (reduced for better convergence).",
    )


    parser.add_argument(
        "--frame_align_weight",
        type=float,
        default=0.5,
        help="Weight for frame-wise alignment loss.",
    )
    parser.add_argument(
        "--noise_timestep_min_ratio",
        type=float,
        default=0.8,
        help="Scheduler noise: sample t in [ratio*N, N) so only mild sigma (e.g. 0.7 -> sigma~[0,0.3]). 1.0 = no noise.",
    )

    args = parser.parse_args()
    
    return args
    
    
def train(args):

    model = LightningModelForTrain(
        vae_path=args.vae_path,
        learning_rate=args.learning_rate,
        cam_ckpt_path=args.cam_ckpt_path,
        weight_decay=args.weight_decay,
        frame_align_weight=args.frame_align_weight,
        noise_timestep_min_ratio=args.noise_timestep_min_ratio,
    )

    dataset = VAE_Alignment(args.dataset_path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True if args.dataloader_num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.dataloader_num_workers > 0 else None,
        drop_last=True
    )

    logger = None

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            pl.pytorch.callbacks.ModelCheckpoint(
                save_top_k=-1,
                save_last=True,
                every_n_epochs=1,
                filename="{epoch}-{step}",
            )
        ],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
        sync_batchnorm=False,
        gradient_clip_val=1.0,
        enable_checkpointing=True
    )
    trainer.fit(model, dataloader, ckpt_path=args.resume_ckpt_path)


if __name__ == '__main__':
    args = parse_args()
    train(args)
