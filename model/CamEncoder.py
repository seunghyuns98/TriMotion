import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from huggingface_hub import PyTorchModelHubMixin 
from model.aggregator import Aggregator
from transformers import T5EncoderModel, T5Tokenizer


class Cam_Encoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, output_dim=768, text_max_length=512, 
                 num_frames=21, vggt_ckpt_path=None):
        super().__init__()

        print(f"Loading Aggregator with img_size={img_size}, patch_size={patch_size}, embed_dim={embed_dim}, output_dim={output_dim}")

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        if vggt_ckpt_path is not None:
            print(f"Loading aggregator checkpoint from {vggt_ckpt_path}")
            aggregator_state_dict = torch.load(vggt_ckpt_path, map_location="cpu")

            if isinstance(aggregator_state_dict, dict):
                if 'aggregator' in aggregator_state_dict:
                    aggregator_state_dict = aggregator_state_dict['aggregator']
                elif 'state_dict' in aggregator_state_dict:
                    aggregator_state_dict = {
                        k[len('aggregator.'):]: v 
                        for k, v in aggregator_state_dict['state_dict'].items() 
                        if k.startswith('aggregator.')
                    }
                
                self.aggregator.load_state_dict(aggregator_state_dict, strict=True)
            print(f"Loaded aggregator checkpoint from {vggt_ckpt_path}!")
        else:   
            print(f"No aggregator checkpoint provided, starting from scratch")
        print(f"Aggregator initialized")

        input_dim = 2 * embed_dim

        self.video_cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        nn.init.normal_(self.video_cls_token, std=0.02)

        self.video_post_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=input_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )

        self.video_norm = nn.LayerNorm(input_dim)
        self.video_proj = nn.Linear(input_dim, output_dim)

        print(f"Loading T5 encoder components (tokenizer, encoder, post_encoder) for text encoding")
        
        self.text_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.text_encoder_t5 = T5EncoderModel.from_pretrained("t5-base")
        t5_max_length = getattr(self.text_encoder_t5.config, 'n_positions', 512)
        self.text_max_length = min(text_max_length, t5_max_length)
        
        for param in self.text_encoder_t5.parameters():
            param.requires_grad = False
        
        text_embed_dim = self.text_encoder_t5.config.d_model

        self.text_norm = nn.LayerNorm(text_embed_dim)
        self.text_global_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_embed_dim,
                nhead=8,
                dim_feedforward=text_embed_dim * 4,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.text_pooling_queries = nn.Parameter(torch.randn(1, num_frames, text_embed_dim) * 0.02)
        self.text_pooling_attn = nn.MultiheadAttention(
            embed_dim=text_embed_dim,
            num_heads=8,
            batch_first=True,
        )
        self.text_pooling_norm = nn.LayerNorm(text_embed_dim)

        self.text_cls_token = nn.Parameter(torch.randn(1, 1, text_embed_dim))
        nn.init.normal_(self.text_cls_token, std=0.02)

        self.text_temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_embed_dim,
                nhead=8,
                dim_feedforward=text_embed_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )

        self.text_out_norm = nn.LayerNorm(text_embed_dim)
        self.text_proj = nn.Linear(text_embed_dim, output_dim)

        print(f"Loading PoseEncoder frame_encoder with pose_dim=12, output_dim={output_dim}")
        pose_hidden_dim = 256
        pose_num_layers = 2
        
        pose_layers = []
        pose_layers.append(nn.Linear(12, pose_hidden_dim))
        pose_layers.append(nn.LayerNorm(pose_hidden_dim))
        pose_layers.append(nn.GELU())
        
        for _ in range(pose_num_layers - 1):
            pose_layers.append(nn.Linear(pose_hidden_dim, pose_hidden_dim))
            pose_layers.append(nn.LayerNorm(pose_hidden_dim))
            pose_layers.append(nn.GELU())
        
        pose_layers.append(nn.Linear(pose_hidden_dim, output_dim))
        self.pose_frame_encoder = nn.Sequential(*pose_layers)
        self.pose_cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
        nn.init.normal_(self.pose_cls_token, std=0.02)

        self.pose_temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=8,
                dim_feedforward=output_dim * 4,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.pose_out_norm = nn.LayerNorm(output_dim)
        self.pose_proj = nn.Linear(output_dim, output_dim)

        self.num_frames = num_frames
        self.output_dim = output_dim       

        print(f"Initializing shared pose prediction head")
        self.pose_predictor = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 12)
        )
       
    def _encode_video_frames(self, videos):
        B, T = videos.shape[0], videos.shape[1]
        
        with torch.no_grad():
            aggregated_tokens_list, _ = self.aggregator(videos)
        
        camera_tokens = aggregated_tokens_list[-1][:, :, 0]
        
        B, S, C = camera_tokens.shape
        cls_tokens = self.video_cls_token.expand(B, -1, -1)
        sequence_with_cls = torch.cat([cls_tokens, camera_tokens], dim=1)

        camera_tokens_processed = self.video_post_encoder(sequence_with_cls)
        camera_tokens_processed = self.video_norm(camera_tokens_processed)
        frame_embeds = self.video_proj(camera_tokens_processed)
        return frame_embeds
    
    def _encode_text_frames(self, texts):

        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, list):
            encoded = self.text_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.text_max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(next(self.text_encoder_t5.parameters()).device)
            attention_mask = encoded["attention_mask"].to(next(self.text_encoder_t5.parameters()).device)
        else:
            input_ids = texts
            attention_mask = None
        
        with torch.no_grad():
            outputs = self.text_encoder_t5(input_ids=input_ids, attention_mask=attention_mask)
            text_token_embeds = outputs.last_hidden_state
        
        B, S, C = text_token_embeds.shape

        text_token_embeds_processed = self.text_norm(text_token_embeds)

        key_padding = (attention_mask == 0) if attention_mask is not None else None
        contextualized = self.text_global_encoder(
            text_token_embeds_processed, src_key_padding_mask=key_padding
        )

        pooling_queries = self.text_pooling_queries.expand(B, -1, -1)
        key_padding = (attention_mask == 0) if attention_mask is not None else None
        text_frame_embeds, _ = self.text_pooling_attn(
            query=pooling_queries,
            key=contextualized,
            value=contextualized,
            key_padding_mask=key_padding,
        )
        text_frame_embeds = self.text_pooling_norm(text_frame_embeds)

        cls_tokens = self.text_cls_token.expand(B, -1, -1)
        sequence_with_cls = torch.cat([cls_tokens, text_frame_embeds], dim=1)

        text_frame_embeds = self.text_temporal_encoder(sequence_with_cls)

        text_frame_embeds = self.text_out_norm(text_frame_embeds)
        text_frame_embeds = self.text_proj(text_frame_embeds)
        
        return text_frame_embeds
    
    def _encode_pose_frames(self, poses):

        B, T = poses.shape[0], poses.shape[1]
        
        pose_embedding_flat = poses.view(B * T, -1)
        frame_embeds_flat = self.pose_frame_encoder(pose_embedding_flat)
        frame_embeds = frame_embeds_flat.view(B, T, -1)
        
        cls_tokens = self.pose_cls_token.expand(B, -1, -1)
        sequence_with_cls = torch.cat([cls_tokens, frame_embeds], dim=1)
        
        frame_embeds = self.pose_temporal_encoder(sequence_with_cls)
        
        frame_embeds = self.pose_out_norm(frame_embeds)
        frame_embeds = self.pose_proj(frame_embeds)
        
        return frame_embeds
   
    def forward(self, videos=None, texts=None, poses=None):
        results = {}
        B = None
        
        if videos is not None:
            video_frame_feats = self._encode_video_frames(videos)
            video_global_embeds = F.normalize(video_frame_feats[:, 0], dim=-1)
            video_frame_embeds = F.normalize(video_frame_feats[:, 1:], dim=-1)
            video_predicted_pose = self.pose_predictor(video_frame_embeds)

            results['video_global_embeds'] = video_global_embeds
            results['video_frame_embeds'] = video_frame_embeds
            results['video_predicted_pose'] = video_predicted_pose
        
        if texts is not None:
            text_frame_feats = self._encode_text_frames(texts)
            text_global_embeds = F.normalize(text_frame_feats[:, 0], dim=-1)
            text_frame_embeds = F.normalize(text_frame_feats[:, 1:], dim=-1)
            text_predicted_pose = self.pose_predictor(text_frame_embeds)

            results['text_global_embeds'] = text_global_embeds
            results['text_frame_embeds'] = text_frame_embeds
            results['text_predicted_pose'] = text_predicted_pose
        
        if poses is not None:
            pose_frame_feats = self._encode_pose_frames(poses)
            pose_global_embeds = F.normalize(pose_frame_feats[:, 0], dim=-1)
            pose_frame_embeds = F.normalize(pose_frame_feats[:, 1:], dim=-1)
            pose_predicted_pose = self.pose_predictor(pose_frame_embeds)
            
            results['pose_global_embeds'] = pose_global_embeds
            results['pose_frame_embeds'] = pose_frame_embeds
            results['pose_predicted_pose'] = pose_predicted_pose
        
        return results

