#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi input models."""
import torch
import inspect
import random
import heapq
from torch.nn.init import xavier_uniform_
import collections
from torch.distributions.categorical import Categorical
from functools import reduce
import torch.nn.functional as F
import math
import copy
from einops import rearrange
import torch.nn as nn

from functools import reduce
from operator import mul
from .head_helper import MultiTaskHead, MultiTaskMViTHead
from .video_model_builder import SlowFast, _POOL1, MViT
from .build import MODEL_REGISTRY
import clip
from einops import rearrange, repeat



@MODEL_REGISTRY.register()
class MultiTaskSlowFast(SlowFast):
    def _construct_network(self, cfg, with_head=False):
        super()._construct_network(cfg, with_head=with_head)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = MultiTaskHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )
        self.head_name = "head"
        self.add_module(self.head_name, head)

@MODEL_REGISTRY.register()
class RecognitionSlowFastRepeatLabels(MultiTaskSlowFast):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def forward(self, x, tgts=None):
        # keep only first input
        x = [xi[:, 0] for xi in x]
        x = super().forward(x)

        # duplicate predictions K times
        K = self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        x = [xi.unsqueeze(1).repeat(1, K, 1) for xi in x]
        return x

    def generate(self, x, k=1):
        x = self.forward(x)
        results = []
        for head_x in x:
            preds_dist = Categorical(logits=head_x)
            preds = [preds_dist.sample() for _ in range(k)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)
        return results


@MODEL_REGISTRY.register()
class MultiTaskMViT(MViT):

    def __init__(self, cfg):

        super().__init__(cfg, with_head =False)

        self.head = MultiTaskMViTHead(
            [768],
            cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class ConcatAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        x = torch.stack(x, dim=1) # (B, num_input_clips, D)
        x = x.view(x.shape[0], -1) # (B, num_input_clips * D)
        return x

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES * cfg.FORECASTING.NUM_INPUT_CLIPS

@MODEL_REGISTRY.register()
class MeanAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        x = torch.stack(x, dim=1) # (B, num_input_clips, D)
        x = x.mean(1)
        return x

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :, :]
        return self.dropout(x)

@MODEL_REGISTRY.register()
class TransformerAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_heads = cfg.MODEL.TRANSFORMER_ENCODER_HEADS
        num_layers = cfg.MODEL.TRANSFORMER_ENCODER_LAYERS
        dim_in = cfg.MODEL.MULTI_INPUT_FEATURES
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_in, num_heads),
            num_layers,
            norm=nn.LayerNorm(dim_in),
        )
        self.pos_encoder = PositionalEncoding(dim_in, dropout=0.2)

    def forward(self, x):
        x = torch.stack(x, dim=1) # (B, num_inputs, D)
        x = x.transpose(0, 1) # (num_inputs, B, D)
        x = self.pos_encoder(x)
        x = self.encoder(x) # (num_inputs, B, D)
        return x[-1] # (B, D) return last timestep's encoding

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES 

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class TransformerAggregator_clip(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_heads = cfg.MODEL.TRANSFORMER_ENCODER_HEADS
        num_layers = cfg.MODEL.TRANSFORMER_ENCODER_LAYERS
        dim_in = cfg.MODEL.MULTI_INPUT_FEATURES + 512
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_in, num_heads),
            num_layers,
            norm=nn.LayerNorm(dim_in),
        )
        self.pos_encoder = PositionalEncoding(dim_in, dropout=0.2)

    def forward(self, x, x_clip):
        x = torch.stack(x, dim=1) # (B, num_inputs, D)
        x_clip = torch.stack(x_clip, dim=1) # (B, num_inputs, 512)
        x = torch.cat((x, x_clip), 2)
        x = x.transpose(0, 1) # (num_inputs, B, D+512)
        x = self.pos_encoder(x)
        x = self.encoder(x) # (num_inputs, B, D+512)
        return x[-1] # (B, D+512) return last timestep's encoding

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES


@MODEL_REGISTRY.register()
class TransformerAggregator_clip_text(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_heads = cfg.MODEL.TRANSFORMER_ENCODER_HEADS
        num_layers = cfg.MODEL.TRANSFORMER_ENCODER_LAYERS
        # dim_in = cfg.MODEL.MULTI_INPUT_FEATURES + 512
        dim_in_text = 512
        dim_in = dim_in_text
        '''
        dim_in_text = 512
        self.encoder_text = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_in_text, num_heads),
            num_layers,
            norm=nn.LayerNorm(dim_in_text),
        )
        self.pos_encoder_text = PositionalEncoding(dim_in_text, dropout=0.2)
        self.attend_text = nn.Softmax(dim=-1)
        '''
        self.text_q = nn.Parameter(torch.randn(1, 1, dim_in_text))
        dropout = 0.2
        self.multihead_attn1 = nn.MultiheadAttention(dim_in_text, num_heads=4)
        self.to_out1 = nn.Sequential(nn.Linear(dim_in_text, dim_in_text), nn.Dropout(dropout))
        self.multihead_attn2 = nn.MultiheadAttention(dim_in_text, num_heads=4)
        self.to_out2 = nn.Sequential(nn.Linear(dim_in_text, dim_in_text), nn.Dropout(dropout))
        self.multihead_attn3 = nn.MultiheadAttention(dim_in_text, num_heads=4)
        self.to_out3 = nn.Sequential(nn.Linear(dim_in_text, dim_in_text), nn.Dropout(dropout))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_in, num_heads),
            num_layers,
            norm=nn.LayerNorm(dim_in),
        )
        self.pos_encoder = PositionalEncoding(dim_in, dropout=0.2)

    def forward(self, x, x_img, x_text):
        tau = 0.1
        # x = torch.stack(x, dim=1) # (B, num_inputs, D)
        x_img = torch.stack(x_img, dim=1) # (B, num_inputs, T, 512)
        # Processing text queries
        x_text = torch.stack(x_text, dim=1)  # (B, num_inputs, T, 512)
        B, K, T, _ = x_text.size()
        # x_text = rearrange(x_text, 'b k t c -> (b k) t c', b=B, t=T)
        # text_q = torch.mean(x_text, 1, True)  # (B', 1, 512)
        B_new = B*K
        text_q = repeat(self.text_q, '1 n d -> b n d', b=B_new) # (B', 1, 512)
        text_q = text_q.transpose(0,1) # (1, B', 512)
        '''
        x_text = x_text.transpose(0, 1) # (T, B', 512)
        x_text = self.pos_encoder_text(x_text)
        x_text = self.encoder_text(x_text) # (T, B', 512)
        text_q = torch.mean(x_text, 0, True) # (1, B', 512)
        '''
        # q-scoring
        # text_q = text_q.transpose(0, 1) # (B', 1, 512)
        x_img = rearrange(x_img, 'b k t c -> (b k) t c', b=B, t=T) # (B', T, 512)
        x_img = x_img.transpose(0, 1)  # (T, B', 512)
        attention1, _ = self.multihead_attn1(text_q, x_img, x_img) # (1, B', 512)
        attention1 = attention1.transpose(0, 1)
        out1 = self.to_out1(attention1)
        out1 = out1.transpose(0, 1) + text_q
        attention2, _ = self.multihead_attn2(out1, x_img, x_img)  # (1, B', 512)
        attention2 = attention2.transpose(0, 1)
        out2 = self.to_out2(attention2)
        out2 = out2.transpose(0, 1) + out1
        attention3, _ = self.multihead_attn3(out2, x_img, x_img)  # (1, B', 512)
        attention3 = attention3.transpose(0, 1) # (B', 1, 512)
        x_clip = self.to_out3(attention3) + out2.transpose(0, 1)
        # scores = torch.matmul(text_q, x_img.transpose(-1, -2)) # (B', 1, T)
        # text_attention = self.attend_text(scores/tau)
        # x_clip = torch.matmul(text_attention, x_img)
        x_clip = rearrange(x_clip.squeeze(1), '(b k) c -> b k c', b=B, k=K)
        # x_clip needs to be processed
        # x = torch.cat((x, x_clip), 2)
        x = x_clip
        x = x.transpose(0, 1) # (num_inputs, B, D+512)
        x = self.pos_encoder(x)
        x = self.encoder(x) # (num_inputs, B, D+512)
        return x[-1] # (B, D+512) return last timestep's encoding

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class MultiHeadDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        head_classes = [
            reduce((lambda x, y: x + y), cfg.MODEL.NUM_CLASSES)
        ] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        head_dim_in = MODEL_REGISTRY.get(cfg.FORECASTING.AGGREGATOR).out_dim(cfg)
        self.head = MultiTaskHead(
            dim_in=[head_dim_in],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def forward(self, x, tgts=None):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1) # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1) # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class MultiHeadDecoder_clip(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        head_classes = [
            reduce((lambda x, y: x + y), cfg.MODEL.NUM_CLASSES)
        ] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        # head_dim_in = MODEL_REGISTRY.get(cfg.FORECASTING.AGGREGATOR).out_dim(cfg) + 512
        head_dim_in = 512
        self.head = MultiTaskHead(
            dim_in=[head_dim_in],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def forward(self, x, tgts=None):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1) # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1) # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class ForecastingEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.build_clip_backbone()
        self.build_clip_aggregator()
        self.build_decoder()

    # to encode frames into a set of {cfg.FORECASTING.NUM_INPUT_CLIPS} clips
    def build_clip_backbone(self):
        cfg = self.cfg
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        backbone_config = copy.deepcopy(cfg)
        backbone_config.MODEL.NUM_CLASSES = [cfg.MODEL.MULTI_INPUT_FEATURES]
        backbone_config.MODEL.HEAD_ACT = None


        if cfg.MODEL.ARCH == "mvit":
            self.backbone = MViT(backbone_config, with_head=True)
        else:
            self.backbone = SlowFast(backbone_config, with_head=True)
        # replace with:
        # self.backbone = MODEL_REGISTRY.get(cfg.FORECASTING.BACKBONE)(backbone_config, with_head=True)

        if cfg.MODEL.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Never freeze head.
            for param in self.backbone.head.parameters():
                param.requires_grad = True


    def build_clip_aggregator(self):
        self.clip_aggregator = MODEL_REGISTRY.get(self.cfg.FORECASTING.AGGREGATOR)(self.cfg)

    def build_decoder(self):
        self.decoder = MODEL_REGISTRY.get(self.cfg.FORECASTING.DECODER)(self.cfg)

    # input = [(B, num_inp, 3, T, H, W), (B, num_inp, 3, T', H, W)]
    def encode_clips(self, x):
        # x -> [torch.Size([2, 2, 3, 8, 224, 224]), torch.Size([2, 2, 3, 32, 224, 224])]
        assert isinstance(x, list) and len(x) >= 1

        num_inputs = x[0].shape[1]
        batch = x[0].shape[0]
        features = []
        for i in range(num_inputs):
            pathway_for_input = []
            for pathway in x:
                input_clip = pathway[:, i]
                pathway_for_input.append(input_clip)

            # pathway_for_input -> [torch.Size([2, 3, 8, 224, 224]), torch.Size([2, 3,32, 224, 224])]
            input_feature = self.backbone(pathway_for_input)
            features.append(input_feature)

        return features

    # input = list of clips: [(B, D)] x {cfg.FORECASTING.NUM_INPUT_CLIPS}
    # output = (B, D') tensor after aggregation
    def aggregate_clip_features(self, x):
        return self.clip_aggregator(x)

    # input = (B, D') tensor encoding of full video
    # output = [(B, Z, #verbs), (B, Z, #nouns)] probabilities for each z
    def decode_predictions(self, x, tgts):
        return self.decoder(x, tgts)

    def forward(self, x, tgts=None):
        features = self.encode_clips(x)
        x = self.aggregate_clip_features(features)
        x = self.decode_predictions(x, tgts)
        return x

    def generate(self, x, k=1):
        x = self.forward(x)
        results = []
        for head_x in x:
            if k>1:
                preds_dist = Categorical(logits=head_x)
                preds = [preds_dist.sample() for _ in range(k)]
            elif k==1:
                preds = [head_x.argmax(2)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)

        return results

#--------------------------------------------------------------------#


@MODEL_REGISTRY.register()
class ForecastingEncoderDecoder_clip(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.build_clip_backbone()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.build_clip_aggregator()
        self.build_decoder()

    # to encode frames into a set of {cfg.FORECASTING.NUM_INPUT_CLIPS} clips
    def build_clip_backbone(self):
        cfg = self.cfg
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        backbone_config = copy.deepcopy(cfg)
        backbone_config.MODEL.NUM_CLASSES = [cfg.MODEL.MULTI_INPUT_FEATURES]
        backbone_config.MODEL.HEAD_ACT = None


        if cfg.MODEL.ARCH == "mvit":
            self.backbone = MViT(backbone_config, with_head=True)
        else:
            self.backbone = SlowFast(backbone_config, with_head=True)
        # replace with:
        # self.backbone = MODEL_REGISTRY.get(cfg.FORECASTING.BACKBONE)(backbone_config, with_head=True)

        if cfg.MODEL.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Never freeze head.
            for param in self.backbone.head.parameters():
                param.requires_grad = True


    def build_clip_aggregator(self):
        self.clip_aggregator = MODEL_REGISTRY.get(self.cfg.FORECASTING.AGGREGATOR)(self.cfg)

    def build_decoder(self):
        self.decoder = MODEL_REGISTRY.get(self.cfg.FORECASTING.DECODER)(self.cfg)

    # input = [(B, num_inp, 3, T, H, W), (B, num_inp, 3, T', H, W)]
    def encode_clips(self, x):
        # x -> [torch.Size([2, 2, 3, 8, 224, 224]), torch.Size([2, 2, 3, 32, 224, 224])]
        assert isinstance(x, list) and len(x) >= 1

        num_inputs = x[0].shape[1]
        B = x[0].shape[0]
        T = x[0].shape[3]
        features = []
        features_clip = []
        for i in range(num_inputs):
            pathway_for_input = []
            count = 0
            for pathway in x:
                input_clip = pathway[:, i]
                if count == 0:
                    clip_img = rearrange(pathway[:, i], 'b c t h w -> (b t) c h w', b=B, t=T)
                    count += 1

                pathway_for_input.append(input_clip)

            # pathway_for_input -> [torch.Size([2, 3, 8, 224, 224]), torch.Size([2, 3,32, 224, 224])]
            input_feature = self.backbone(pathway_for_input)
            with torch.no_grad():
                clip_feature = self.clip_model.encode_image(clip_img)
            clip_feature = rearrange(clip_feature, '(b t) c -> b t c', b=B, t=T)
            clip_feature = torch.mean(clip_feature, 1)
            features.append(input_feature)
            features_clip.append(clip_feature)

        return features, features_clip

    # input = list of clips: [(B, D)] x {cfg.FORECASTING.NUM_INPUT_CLIPS}
    # output = (B, D') tensor after aggregation
    def aggregate_clip_features(self, x, y):
        return self.clip_aggregator(x, y)

    # input = (B, D') tensor encoding of full video
    # output = [(B, Z, #verbs), (B, Z, #nouns)] probabilities for each z
    def decode_predictions(self, x, tgts):
        return self.decoder(x, tgts)

    def forward(self, x, tgts=None):
        features, features_clip = self.encode_clips(x)
        x = self.aggregate_clip_features(features, features_clip)
        x = self.decode_predictions(x, tgts)
        return x

    def generate(self, x, k=1):
        x = self.forward(x)
        results = []
        for head_x in x:
            if k>1:
                preds_dist = Categorical(logits=head_x)
                preds = [preds_dist.sample() for _ in range(k)]
            elif k==1:
                preds = [head_x.argmax(2)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)

        return results


@MODEL_REGISTRY.register()
class ForecastingEncoderDecoder_clip_text(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.build_clip_backbone()
        device = "cuda"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device)
        nouns = [i.strip() for i in open('text_clip/noun.txt').readlines()]
        nouns_processed = [i.split("_")[0] for i in nouns]
        text_descriptions_1 = [f"The person is using {n}" for n in nouns_processed]
        text_descriptions_2 = [f"There is a {n} in the scene" for n in nouns_processed]
        self.text_descriptions_noun = text_descriptions_1 + text_descriptions_2

        self.build_clip_aggregator()
        self.build_decoder()

    # to encode frames into a set of {cfg.FORECASTING.NUM_INPUT_CLIPS} clips
    def build_clip_backbone(self):
        cfg = self.cfg
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        backbone_config = copy.deepcopy(cfg)
        backbone_config.MODEL.NUM_CLASSES = [cfg.MODEL.MULTI_INPUT_FEATURES]
        backbone_config.MODEL.HEAD_ACT = None

        '''
        if cfg.MODEL.ARCH == "mvit":
            self.backbone = MViT(backbone_config, with_head=True)
        else:
            self.backbone = SlowFast(backbone_config, with_head=True)
        '''
        # replace with:
        # self.backbone = MODEL_REGISTRY.get(cfg.FORECASTING.BACKBONE)(backbone_config, with_head=True)

        if cfg.MODEL.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Never freeze head.
            for param in self.backbone.head.parameters():
                param.requires_grad = True


    def build_clip_aggregator(self):
        self.clip_aggregator = MODEL_REGISTRY.get(self.cfg.FORECASTING.AGGREGATOR)(self.cfg)

    def build_decoder(self):
        self.decoder = MODEL_REGISTRY.get(self.cfg.FORECASTING.DECODER)(self.cfg)

    # input = [(B, num_inp, 3, T, H, W), (B, num_inp, 3, T', H, W)]
    def encode_clips(self, x):
        # x -> [torch.Size([2, 2, 3, 8, 224, 224]), torch.Size([2, 2, 3, 32, 224, 224])]
        assert isinstance(x, list) and len(x) >= 1

        num_inputs = x[0].shape[1]
        B = x[0].shape[0]
        T = x[0].shape[3]
        features = []
        features_img = []
        features_text = []
        for i in range(num_inputs):
            pathway_for_input = []
            count = 0
            for pathway in x:
                input_clip = pathway[:, i]
                if count == 0:
                    clip_img = rearrange(pathway[:, i], 'b c t h w -> (b t) c h w', b=B, t=T)
                    count += 1

                # pathway_for_input.append(input_clip)

            # pathway_for_input -> [torch.Size([2, 3, 8, 224, 224]), torch.Size([2, 3,32, 224, 224])]
            # input_feature = self.backbone(pathway_for_input)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(clip_img).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)

                text_tokens_noun = clip.tokenize(self.text_descriptions_noun).cuda()
                text_features_noun = self.clip_model.encode_text(text_tokens_noun).float()
                text_features_noun /= text_features_noun.norm(dim=-1, keepdim=True)
                text_probs_noun = (100.0 * image_features @ text_features_noun.T).softmax(dim=-1)
                top_probs_noun, top_labels_noun = text_probs_noun.cpu().topk(1, dim=-1)
                top_noun = text_features_noun[top_labels_noun].squeeze(1)    # Noun queries input

            # clip_feature = torch.cat((image_features, top_noun), 1)
            clip_img_feature = rearrange(image_features, '(b t) c -> b t c', b=B, t=T)
            clip_text_feature = rearrange(top_noun, '(b t) c -> b t c', b=B, t=T)
            # clip_feature = torch.mean(clip_feature, 1)
            features_img.append(clip_img_feature)
            features_text.append(clip_text_feature)
            # features.append(input_feature)
            # features_clip.append(clip_feature)

        return features, features_img, features_text

    # input = list of clips: [(B, D)] x {cfg.FORECASTING.NUM_INPUT_CLIPS}
    # output = (B, D') tensor after aggregation
    def aggregate_clip_features(self, x, y, z):
        return self.clip_aggregator(x, y, z)

    # input = (B, D') tensor encoding of full video
    # output = [(B, Z, #verbs), (B, Z, #nouns)] probabilities for each z
    def decode_predictions(self, x, tgts):
        return self.decoder(x, tgts)

    def forward(self, x, tgts=None):
        features, features_img, features_text = self.encode_clips(x)
        x = self.aggregate_clip_features(features, features_img, features_text)
        x = self.decode_predictions(x, tgts)
        return x

    def generate(self, x, k=1):
        x = self.forward(x)
        results = []
        for head_x in x:
            if k>1:
                preds_dist = Categorical(logits=head_x)
                preds = [preds_dist.sample() for _ in range(k)]
            elif k==1:
                preds = [head_x.argmax(2)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)

        return results
