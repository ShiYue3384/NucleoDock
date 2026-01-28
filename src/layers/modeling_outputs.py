from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput


@dataclass
class UniMolModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    last_pair_repr: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pair_reprs: Optional[Tuple[torch.FloatTensor]] = None
    attention_weights: Optional[Tuple[torch.FloatTensor]] = None
    attention_probs: Optional[Tuple[torch.FloatTensor]] = None
    delta_encoder_pair_rep: torch.FloatTensor = None
    x_norm: torch.FloatTensor = None
    delta_encoder_pair_rep_norm: torch.FloatTensor = None


@dataclass
class UniMolPretrainingModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    last_pair_repr: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pair_reprs: Optional[Tuple[torch.FloatTensor]] = None
    delta_encoder_pair_rep: torch.FloatTensor = None
    attention_weights: Optional[Tuple[torch.FloatTensor]] = None
    attention_probs: Optional[Tuple[torch.FloatTensor]] = None
    masked_token_loss: torch.FloatTensor = None
    masked_coord_loss: torch.FloatTensor = None
    masked_dist_loss: torch.FloatTensor = None
    x_norm: torch.FloatTensor = None
    delta_encoder_pair_rep_norm: torch.FloatTensor = None
    loss: torch.FloatTensor = None


@dataclass
class UniMolDownStreamModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    last_pair_repr: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pair_reprs: Optional[Tuple[torch.FloatTensor]] = None
    attention_weights: Optional[Tuple[torch.FloatTensor]] = None
    attention_probs: Optional[Tuple[torch.FloatTensor]] = None
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


@dataclass
class UniMolBindingPoseOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    score_loss: Optional[torch.FloatTensor] = None
    distance_loss: Optional[torch.FloatTensor] = None
    coordinate_loss: Optional[torch.FloatTensor] = None
    score: torch.FloatTensor = None
    distance: torch.FloatTensor = None
    x_norm: torch.FloatTensor = None
    delta_encoder_pair_rep_norm: torch.FloatTensor = None
    coordinate: torch.FloatTensor = None


@dataclass
class DockingPoseOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    score_loss: Optional[torch.FloatTensor] = None
    affinity_loss: Optional[torch.FloatTensor] = None
    cross_loss: Optional[torch.FloatTensor] = None
    holo_loss: Optional[torch.FloatTensor] = None
    holo_1hop_loss: Optional[torch.FloatTensor] = None
    holo_2hop_loss: Optional[torch.FloatTensor] = None
    holo_3hop_loss: Optional[torch.FloatTensor] = None
    rtm_score: Optional[torch.FloatTensor] = None
    affinity_predict: Optional[torch.FloatTensor] = None
    cross_distance_predict: Optional[torch.FloatTensor] = None
    holo_distance_predict: Optional[torch.FloatTensor] = None
    mdn: Optional[Tuple[torch.FloatTensor]] = None
    # ---- 新增：样本级融合表征 [B, H] ls revise----
    transformer_fused_repr: Optional[torch.FloatTensor] = None

@dataclass
class SDDockingPoseOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    score_loss: Optional[torch.FloatTensor] = None
    score_loss2: Optional[torch.FloatTensor] = None
    affinity_loss: Optional[torch.FloatTensor] = None
    affinity_loss2: Optional[torch.FloatTensor] = None
    cross_loss: Optional[torch.FloatTensor] = None
    cross_loss2: Optional[torch.FloatTensor] = None
    holo_loss: Optional[torch.FloatTensor] = None
    holo_loss2: Optional[torch.FloatTensor] = None
    holo_1hop_loss: Optional[torch.FloatTensor] = None
    holo_2hop_loss: Optional[torch.FloatTensor] = None
    holo_3hop_loss: Optional[torch.FloatTensor] = None
    holo_1hop_loss2: Optional[torch.FloatTensor] = None
    holo_2hop_loss2: Optional[torch.FloatTensor] = None
    holo_3hop_loss2: Optional[torch.FloatTensor] = None
    sd_loss: Optional[torch.FloatTensor] = None
    affinity_predict: Optional[torch.FloatTensor] = None
    cross_distance_predict: Optional[torch.FloatTensor] = None
    holo_distance_predict: Optional[torch.FloatTensor] = None
    mdn: Optional[Tuple[torch.FloatTensor]] = None
    rtm_score: Optional[torch.FloatTensor] = None


@dataclass
class TransformerMModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    last_pair_repr: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pair_reprs: Optional[Tuple[torch.FloatTensor]] = None
    atom_output: Optional[Tuple[torch.FloatTensor]] = None
