import torch, os, argparse
import lightning as pl
import torch.nn.functional as F
from dataloader import Embedding_Space
from utils.losses import InfoNCE_Loss
from model import Cam_Encoder

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        vggt_ckpt_path=None,
        embed_dim=1024,
        output_dim=768,
        patch_size=14,
        text_max_length=512,
        img_size=518,
        weight_decay=0.2,
        loss_temperature=0.07,
        pose_prediction_weight=1.0,
        num_frames=21,
        frame_align_weight=0.1,
    ):
        super().__init__()
        
        print(f"Loading Encoder with img_size={img_size}, patch_size={patch_size}, embed_dim={embed_dim}, output_dim={output_dim}")
        
        self.cam_encoder = Cam_Encoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            output_dim=output_dim,
            text_max_length=text_max_length,
            num_frames=num_frames,
            vggt_ckpt_path=vggt_ckpt_path,
        )

        self.num_frames = num_frames
        self.pose_prediction_weight = pose_prediction_weight
        self.frame_align_weight = frame_align_weight
        print(f"Pose prediction weight: {self.pose_prediction_weight}")
        print(f"Frame align weight: {self.frame_align_weight}")

        self.freeze_encoders()

        self.criterion = InfoNCE_Loss(temperature=loss_temperature)

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

    def freeze_encoders(self):
        self.cam_encoder.requires_grad_(True)
        self.cam_encoder.train()
        
        self.cam_encoder.aggregator.requires_grad_(False)
        self.cam_encoder.aggregator.eval()
        
        self.cam_encoder.text_encoder_t5.requires_grad_(False)
        self.cam_encoder.text_encoder_t5.eval()
        
        print("Encoders configured: aggregator and T5 are frozen.")
    
    def training_step(self, batch, batch_idx):
        video = batch['video']
        texts = batch['cam_caption']
        pose = batch['pose_embedding']

        encoder_outputs = self.cam_encoder(
            videos=video,
            texts=texts,
            poses=pose,
        )

        video_global = encoder_outputs['video_global_embeds']
        text_global = encoder_outputs['text_global_embeds']
        pose_global = encoder_outputs['pose_global_embeds']

        video_predicted_pose = encoder_outputs['video_predicted_pose']
        text_predicted_pose = encoder_outputs['text_predicted_pose']
        pose_predicted_pose = encoder_outputs['pose_predicted_pose']

        video_frame_embeds = encoder_outputs['video_frame_embeds']
        text_frame_embeds = encoder_outputs['text_frame_embeds']
        pose_frame_embeds = encoder_outputs['pose_frame_embeds']

        loss_v2t = self.criterion(video_global, text_global)
        loss_v2p = self.criterion(video_global, pose_global)
        loss_t2p = self.criterion(text_global, pose_global)
        contrastive_loss = loss_v2t + loss_t2p + loss_v2p

        def _frame_cos_loss(a, b):
            return (1 - F.cosine_similarity(a, b, dim=-1)).mean()
        frame_align_loss = (
            _frame_cos_loss(video_frame_embeds, text_frame_embeds)
            + _frame_cos_loss(video_frame_embeds, pose_frame_embeds)
            + _frame_cos_loss(text_frame_embeds, pose_frame_embeds)
        )

        loss_video_pose_pred = F.l1_loss(video_predicted_pose, pose)
        loss_text_pose_pred = F.l1_loss(text_predicted_pose, pose)
        loss_pose_pose_pred = F.l1_loss(pose_predicted_pose, pose)
        prediction_loss = loss_video_pose_pred + loss_text_pose_pred + loss_pose_pose_pred

        total_loss = (
            contrastive_loss
            + self.frame_align_weight * frame_align_loss
            + self.pose_prediction_weight * prediction_loss
        )

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train/c_loss", contrastive_loss, prog_bar=True)
        self.log("train/f_loss", frame_align_loss, prog_bar=True)
        self.log("train/p_loss", prediction_loss, prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )

        print(f"Optimizer configured:")
        print(f"  - Trainable params: {sum(p.numel() for p in trainable_params):,} params, lr={self.learning_rate:.2e}")

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
            
def parse_args():
    parser = argparse.ArgumentParser(description="Train Embedding Space")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./MotionTriplet-Dataset",
        help="The path of the Motion Triplet Dataset.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoint/embedding_space",
        help="Path to save the Embedding Space model.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help="Number of subprocesses to use for data loading.",
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
        default=100,
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
        "--vggt_ckpt_path",
        type=str,
        default="checkpoint/trimotion/aggregator.ckpt",
        help="Path to pretrained VGGT checkpoint (optional, loads aggregator from HuggingFace by default).",
    )

    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="Path to resume training checkpoint.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
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
        default=0.2,
        help="Weight decay for optimizer.",
    )

    parser.add_argument(
        "--loss_temperature",
        type=float,
        default=0.07,
        help="Temperature parameter for contrastive loss.",
    )

    parser.add_argument(
        "--pose_prediction_weight",
        type=float,
        default=1.0,
        help="Weight for pose prediction losses.",
    )
    parser.add_argument(
        "--frame_align_weight",
        type=float,
        default=0.1,
        help="Weight for frame-level alignment losses.",
    )

    args = parser.parse_args()

    return args
    
    
def train(args):

    model = LightningModelForTrain(
        learning_rate=args.learning_rate,
        vggt_ckpt_path=args.vggt_ckpt_path,
        weight_decay=args.weight_decay,
        loss_temperature=args.loss_temperature,
        pose_prediction_weight=args.pose_prediction_weight,
        frame_align_weight=args.frame_align_weight,
    )

    dataset = Embedding_Space(args.dataset_path)

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
        enable_checkpointing=True,
    )
    trainer.fit(model, dataloader, ckpt_path=args.resume_ckpt_path)


if __name__ == '__main__':
    args = parse_args()
    train(args)
