import os
import torch
import argparse
import lightning as pl
from diffsynth import WanVideoMultimodalPipeline, ModelManager
from diffsynth.models.wan_video_dit import MLP, RMSNorm
from dataloader import MotionTripletDataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import warnings
from model import VAE_Projector

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path,
        vae_path,
        dit_path,
        image_encoder_path,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        learning_rate=1e-4,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        i2v_ckpt_path=None,
        resume_ckpt_path=None,
        vae_projector_ckpt_path=None,   
        align_loss_weight=0.1,
        frame_align_weight=0.05,
    ):
        super().__init__()

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_path = [text_encoder_path, vae_path, dit_path, image_encoder_path]
        model_manager.load_models(model_path)
        
        self.pipe = WanVideoMultimodalPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        with torch.no_grad():
            patch_embedding_ori = self.pipe.dit.patch_embedding
            patch_embedding_new = nn.Conv3d(36, 1536, kernel_size=(1, 2, 2), stride=(1, 2, 2))

            new_weights = torch.cat([
                patch_embedding_ori.weight,     
                patch_embedding_ori.weight,
                patch_embedding_ori.weight[:, :4]
            ], dim=1)

            assert new_weights.shape[1] == 36
            patch_embedding_new.weight.copy_(new_weights)
            patch_embedding_new.bias.copy_(patch_embedding_ori.bias)

        self.pipe.dit.patch_embedding = patch_embedding_new

        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        self.pipe.dit.img_emb = MLP(1280, dim)
        for block in self.pipe.dit.blocks:
            block.cross_attn.k_img = nn.Linear(dim, dim)
            block.cross_attn.v_img = nn.Linear(dim, dim)
            block.cross_attn.norm_k_img = RMSNorm(dim, eps=1e-6)

        if i2v_ckpt_path is not None:
            state_dict = torch.load(i2v_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
            print(f"load i2v ckpt from {i2v_ckpt_path}!")

        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(768, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))

        self.pipe.denoising_model().has_image_input = True

        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
            print(f"load ckpt from {resume_ckpt_path}!")
        self.freeze_parameters()

        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
                for param in module.parameters():
                    param.requires_grad = True

        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        print(f"Total number of trainable parameters: {trainable_params}")

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.align_loss_weight = align_loss_weight
        self.frame_align_weight = frame_align_weight

        self.vae_projector = VAE_Projector()
        
        if vae_projector_ckpt_path is not None:
            state_dict = torch.load(vae_projector_ckpt_path, map_location="cpu")
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if any(key.startswith('vae_projector.') for key in state_dict.keys()):
                state_dict = {k[len('vae_projector.'):]: v for k, v in state_dict.items() if k.startswith('vae_projector.')}
            self.vae_projector.load_state_dict(state_dict, strict=True)
            print(f"Loaded VAE projector from {vae_projector_ckpt_path}")
        
        self.vae_projector.requires_grad_(False)
        self.vae_projector.eval()
        
        
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()


    
    def training_step(self, batch, batch_idx):
        video = batch["video"]
        reference_latents = batch["reference_latents"]
        reference_global = batch["reference_global"]
        src_video = batch["src_video"]  
        text = batch["text"]  
        first_frames = batch["first_frame"] 
        batch_size = len(text)

        if self.pipe.device != self.device:
            self.pipe.device = self.device
        
        video = video.to(dtype=self.pipe.torch_dtype, device=self.device)
        src_video = src_video.to(dtype=self.pipe.torch_dtype, device=self.device)
        reference_latents = reference_latents.to(dtype=self.pipe.torch_dtype, device=self.device)
        reference_global = reference_global.to(dtype=self.pipe.torch_dtype, device=self.device)

        video_input = torch.cat([video, src_video], dim=0)
        _, _, num_frames, height, width = video.shape
        first_frames = [Image.fromarray(frame.cpu().numpy()) for frame in first_frames]

        
        with torch.no_grad():
            prompt_emb = self.pipe.encode_prompt(text) 
            latents_all = self.pipe.encode_video(video_input, **self.tiler_kwargs)
            latents = latents_all[:batch_size]
            src_latents = latents_all[batch_size:]
            src_latents = F.pad(src_latents, (0, 0, 0, 0, 0, 0, 0, 20))

            image_emb = self.pipe.encode_image(first_frames, num_frames, height, width)

        cam_emb = reference_latents 
        
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        tgt_latent_len = latents.shape[2] 
        
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            content_latents = src_latents, has_image_input=True
        )
        
        diffusion_loss = torch.nn.functional.mse_loss(noise_pred[:, :, :tgt_latent_len, ...].float(), training_target[:, :, :tgt_latent_len, ...].float())
        diffusion_loss = diffusion_loss * self.pipe.scheduler.training_weight(timestep)
        
        pred_x0 = self.pipe.scheduler.pred_x0(
            noisy_latents[:, :, :tgt_latent_len, ...],
            noise_pred[:, :, :tgt_latent_len, ...],
            timestep,
        )
        generated_video_emb = self.vae_projector(pred_x0)
        generated_video_global = generated_video_emb["video_global_embeds"]
        generated_video_frame = generated_video_emb["video_frame_embeds"]
        cosine_sim = F.cosine_similarity(generated_video_global, reference_global, dim=-1)
        align_loss = (1 - cosine_sim).mean()
        frame_cosine_sim = F.cosine_similarity(generated_video_frame, reference_latents, dim=-1)
        frame_align_loss = (1 - frame_cosine_sim).mean()

        total_loss = diffusion_loss + self.align_loss_weight * align_loss + self.frame_align_weight * frame_align_loss
        
        self.log("train_loss", total_loss, prog_bar=True, batch_size=batch_size)
        self.log("train/diffusion_loss", diffusion_loss, prog_bar=True, batch_size=batch_size)
        self.log("train/align_loss", align_loss, prog_bar=True, batch_size=batch_size)
        self.log("train/frame_align_loss", frame_align_loss, prog_bar=True, batch_size=batch_size)
        self.log("timestep_id", float(timestep_id.item()), prog_bar=True, batch_size=batch_size)
        return total_loss
    
    
    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(
            trainable_modules, 
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            fused=True,  
        )
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint.clear()
        state_dict = self.pipe.denoising_model().state_dict()
        ckpt_path = os.path.join(checkpoint_dir, f"step{current_step}.ckpt")
        torch.save(state_dict, ckpt_path)



def parse_args():
    parser = argparse.ArgumentParser(description="Train TriMotion")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./MotionTriplet-Dataset",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoint/tri_motion",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="checkpoint/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="checkpoint/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="checkpoint/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="checkpoint/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=384,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=672,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=6,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=4,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Prefetch factor.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_2",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=True,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--i2v_ckpt_path",
        type=str,
        default="checkpoint/trimotion/i2v_baseline.ckpt",
        help="Path to I2V encoder checkpoint.",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="Path of resume checkpoint. If None, the model will start from scratch.",
    )

    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=400,
        help="Number of training steps between validation checks. Set to 0 to disable validation during training.",
    )
    parser.add_argument(
        "--vae_projector_ckpt_path",
        type=str,
        default="checkpoint/trimotion/vae_projection.ckpt",
        required=True,
        help="Path to VAE projector checkpoint.",
    )
    parser.add_argument(
        "--align_loss_weight",
        type=float,
        default=0.1,
        help="Weight for alignment loss between predicted pose embedding and video embedding.",
    )
    parser.add_argument(
        "--frame_align_weight",
        type=float,
        default=0.05,
        help="Weight for frame-level alignment losses.",
    )
    args = parser.parse_args()
    return args
    
    
def train(args):

    train_dataset = MotionTripletDataset(
        args.dataset_path,
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    
    model = LightningModelForTrain(
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        dit_path=args.dit_path,
        image_encoder_path=args.image_encoder_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        i2v_ckpt_path=args.i2v_ckpt_path,
        resume_ckpt_path=args.resume_ckpt_path,
        vae_projector_ckpt_path=args.vae_projector_ckpt_path,
        align_loss_weight=args.align_loss_weight,
        frame_align_weight=args.frame_align_weight,
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
                filename="{epoch}-{step}",
                save_on_train_epoch_end=False,
            )
        ],
        logger=logger,
    )
    trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    train(args)