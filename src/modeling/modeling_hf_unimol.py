import json
import logging
import os
import torch
import transformers as tfs
import torch.nn.functional as F
import torch.nn as nn
from transformers.activations import ACT2FN
from src.layers.modeling_hf_unimol_layers import TransformerEncoderWithPair
from src.layers.modeling_outputs import UniMolDownStreamModelOutput, UniMolModelOutput, UniMolPretrainingModelOutput, \
    UniMolBindingPoseOutput, DockingPoseOutput

from transformers.models.bert import BertPreTrainedModel

from src.layers.openfold.modeling_evoformer import TriAtten

logger = logging.getLogger(__name__)


class UnimolConfig(tfs.PretrainedConfig):

    def __init__(self,
                 vocab_size=100,
                 hidden_size=512,
                 pair_hidden_size=128,
                 num_hidden_layers=15,
                 num_attention_heads=64,
                 intermediate_size=2048,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 activation_dropout=0.0,
                 attention_probs_dropout_prob=0.1,
                 pooler_dropout=0.0,
                 layer_norm_eps=1e-12,
                 max_seq_len=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pad_token_id=0,
                 post_ln=False,
                 masked_token_loss=0.1,
                 masked_coord_loss=5,
                 masked_dist_loss=10,
                 x_norm_loss=0.01,
                 delta_pair_repr_norm_loss=0.01,
                 **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pair_hidden_size = pair_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.pooler_dropout = pooler_dropout
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.activation_dropout = activation_dropout
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.max_seq_len = max_seq_len
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.post_ln = post_ln
        self.masked_token_loss = masked_token_loss
        self.masked_coord_loss = masked_coord_loss
        self.masked_dist_loss = masked_dist_loss
        self.x_norm_loss = x_norm_loss
        self.delta_pair_repr_norm_loss = delta_pair_repr_norm_loss

    def to_dict(self):
        d = super(UnimolConfig, self).to_dict()
        for k, v in d.items():
            value = getattr(self, k)
            if isinstance(value, tfs.PretrainedConfig):
                d[k] = value.to_dict()
        return d


class UniMolPreTrainedModel(tfs.PreTrainedModel):
    config_class = UnimolConfig
    base_model_prefix = "unimol"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) if not config.post_ln else None
        self.activation_fn = ACT2FN[config.hidden_act] if not config.post_ln else None
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if not config.post_ln else None
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = features
        if not self.config.post_ln:
            x = self.dense(features)
            x = self.activation_fn(x)
            x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            activation_fn,
            pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = ACT2FN[activation_fn]
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
            self,
            input_dim,
            out_dim,
            activation_fn,
            hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = ACT2FN[activation_fn]

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):

    def __init__(
            self,
            heads,
            activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = ACT2FN[activation_fn]

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):

    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class UniMolModel(UniMolPreTrainedModel):

    def __init__(self, config: UnimolConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=config.num_hidden_layers,
            embed_dim=config.hidden_size,
            ffn_embed_dim=config.intermediate_size,
            attention_heads=config.num_attention_heads,
            emb_dropout=config.hidden_dropout_prob,
            dropout=config.hidden_dropout_prob,
            attention_dropout=config.attention_probs_dropout_prob,
            activation_dropout=config.activation_dropout,
            max_seq_len=config.max_seq_len,
            activation_fn=config.hidden_act,
            no_final_head_layer_norm=config.delta_pair_repr_norm_loss <= 0,
            post_ln=config.post_ln,
        )
        n_edge_type = config.vocab_size ** 2
        self.gbf_proj = NonLinearHead(config.pair_hidden_size, config.num_attention_heads, config.hidden_act)
        self.post_init()
        self.gbf = GaussianLayer(config.pair_hidden_size, n_edge_type)

    def get_dist_features(self, dist, et):
        n_node = dist.size(-1)
        gbf_feature = self.gbf(dist, et)
        gbf_result = self.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        return graph_attn_bias

    def forward(self, src_tokens=None, src_distance=None, src_edge_type=None, padding_idx=0,
                seq_rep=None,
                pair_rep=None,
                padding_mask=None, **kwargs):
        assert (src_tokens is not None and src_distance is not None and src_edge_type is not None) or (
                seq_rep is not None and pair_rep is not None), "至少需要一种输入"

        if padding_mask is None:
            padding_mask = src_tokens.eq(padding_idx)
        if seq_rep is None:
            seq_rep = self.embed_tokens(src_tokens)
        if pair_rep is None:
            pair_rep = self.get_dist_features(src_distance, src_edge_type)
        else:
            n_node = pair_rep.size(-2)
            pair_rep = pair_rep.permute(0, 3, 1, 2).contiguous()
            pair_rep = pair_rep.view(-1, n_node, n_node)
        encoder_rep, encoder_pair_rep, delta_encoder_pair_rep, x_norm, delta_encoder_pair_rep_norm = self.encoder(
            seq_rep,
            padding_mask=padding_mask,
            attn_mask=pair_rep)
        encoder_pair_rep[encoder_pair_rep == float('-inf')] = 0

        outputs = UniMolModelOutput(last_hidden_state=encoder_rep,
                                    last_pair_repr=encoder_pair_rep,
                                    delta_encoder_pair_rep=delta_encoder_pair_rep,
                                    x_norm=x_norm,
                                    delta_encoder_pair_rep_norm=delta_encoder_pair_rep_norm)

        return outputs


class UniMolForPretraining(UniMolPreTrainedModel):

    def __init__(self, config: UnimolConfig):
        super().__init__(config)
        self.config = config
        self.unimol = UniMolModel(config)
        self.lm_head = MaskLMHead(config)
        self.pair2coord_proj = NonLinearHead(config.num_attention_heads, 1,
                                             config.hidden_act) if config.masked_coord_loss > 0 else None
        self.dist_head = DistanceHead(config.num_attention_heads,
                                      config.hidden_act) if config.masked_coord_loss > 0 else None
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888

    def forward(self, src_tokens, src_distance, src_coord, src_edge_type, tokens_target, distance_target, coord_target,
                attention_mask=None, ignore_token_idx=0):
        unimol_outputs = self.unimol(src_tokens=src_tokens, src_distance=src_distance, src_edge_type=src_edge_type)
        hidden_states = unimol_outputs.last_hidden_state
        pair_repr = unimol_outputs.last_pair_repr

        attented_mask = tokens_target.ne(0)
        mask_logits = self.lm_head(hidden_states)
        mask_loss_fc = nn.CrossEntropyLoss()
        mtm_loss = mask_loss_fc(mask_logits[attented_mask], tokens_target[attented_mask])

        coords_emb = src_coord
        atom_num = src_tokens.ne(0).int().sum(dim=-1).view(-1, 1, 1, 1)
        delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
        attn_probs = self.pair2coord_proj(unimol_outputs.delta_encoder_pair_rep)
        coord_update = delta_pos / atom_num * attn_probs
        coord_update = torch.sum(coord_update, dim=2)
        encoder_coord = coords_emb + coord_update
        # 仅计算mask部分的loss
        masked_coord_loss = F.smooth_l1_loss(
            encoder_coord[attented_mask].view(-1, 3).float(),
            coord_target[attented_mask].view(-1, 3),
            reduction="mean",
            beta=1.0,
        )

        distance = self.dist_head(pair_repr)
        dist_masked_tokens = attented_mask
        masked_distance = distance[dist_masked_tokens, :]
        masked_distance_target = distance_target[dist_masked_tokens]
        non_pad_pos = masked_distance_target > 0
        masked_distance_target = (masked_distance_target.float() - self.dist_mean) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )

        loss = self.config.masked_token_loss * mtm_loss + self.config.masked_coord_loss * masked_coord_loss + self.config.masked_dist_loss * masked_dist_loss

        if unimol_outputs.x_norm is not None:
            loss = loss + self.config.x_norm_loss * unimol_outputs.x_norm

        if unimol_outputs.delta_encoder_pair_rep_norm is not None:
            loss = loss + self.config.delta_pair_repr_norm_loss * unimol_outputs.delta_encoder_pair_rep_norm

        outputs = UniMolPretrainingModelOutput(loss=loss, masked_token_loss=mtm_loss,
                                               masked_coord_loss=masked_coord_loss, masked_dist_loss=masked_dist_loss,
                                               **unimol_outputs)
        return outputs


class UniMolForClassification(UniMolPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.unimol = UniMolModel(config)
        # follow unimol
        # self.classifier = ClassificationHead(config.hidden_size, config.hidden_size, config.num_labels,
        #                                      config.hidden_act, config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, src_tokens, src_distance, src_edge_type, labels=None, **unused):
        unimol_outputs = self.unimol(src_tokens=src_tokens, src_distance=src_distance, src_edge_type=src_edge_type)
        feature = unimol_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(feature)
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels.view(-1))
        outputs = UniMolDownStreamModelOutput(loss=loss, logits=logits)
        return outputs


class UniMolForRegression(UniMolPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.unimol = UniMolModel(config)
        # follow unimol
        # self.classifier = ClassificationHead(config.hidden_size, config.hidden_size, config.num_labels,
        #                                      config.hidden_act, config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, src_tokens, src_distance, src_edge_type, labels=None, cfg=None, mean=None, std=None, **unused):
        unimol_outputs = self.unimol(src_tokens=src_tokens, src_distance=src_distance, src_edge_type=src_edge_type)
        feature = unimol_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(feature)
        loss = None
        if labels is not None:
            if mean is not None and std is not None:
                labels = (labels - mean) / std
            criterion = nn.SmoothL1Loss(reduction='sum') if 'qm' in cfg.MODEL.TASK else nn.MSELoss()
            loss = criterion(logits, labels)
        outputs = UniMolDownStreamModelOutput(loss=loss, logits=logits)
        return outputs


class UniMolForMultiLabelsClassfication(UniMolPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.unimol = UniMolModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, src_tokens, src_distance, src_edge_type, labels=None, **unused):
        unimol_outputs = self.unimol(src_tokens=src_tokens, src_distance=src_distance, src_edge_type=src_edge_type)
        feature = unimol_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(feature)
        loss = None
        if labels is not None:
            # 部分样本无标签
            indices = labels.squeeze() > -0.5
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits[indices], labels[indices].float())
        outputs = UniMolDownStreamModelOutput(loss=loss, logits=torch.sigmoid(logits))
        return outputs


class UniMolForBindingPosePrediction(UniMolPreTrainedModel):
    def __init__(self, config: UnimolConfig):
        super(UniMolForBindingPosePrediction, self).__init__(config)
        self.config = config
        self.binding_encoder = UniMolModel(config)
        self.pair2coord_proj = NonLinearHead(config.num_attention_heads, 1, config.hidden_act)
        self.dist_head = DistanceHead(config.num_attention_heads, config.hidden_act)
        self.mol_pooler = nn.Linear(config.mol_config.hidden_size, config.hidden_size)
        self.mol_pair_pooler = nn.Linear(config.mol_config.num_attention_heads, config.num_attention_heads)
        self.pro_pooler = nn.Linear(config.pro_config.hidden_size, config.hidden_size)
        self.pro_pair_pooler = nn.Linear(config.pro_config.num_attention_heads, config.num_attention_heads)
        # self.classifier2 = ClassificationHead(config.hidden_size, config.hidden_size, config.num_labels, config.hidden_act,
        #                                       config.pooler_dropout)
        self.classifier2 = nn.Linear(config.hidden_size, 1)
        self.score_mean = -4.040115437932443
        self.score_std = 1.303156261621574
        self.dis_mean = 9.358839988708496
        self.dis_std = 3.635190963745117
        self.post_init()
        self.molecule_encoder = UniMolModel(config.mol_config)
        self.pocket_encoder = UniMolModel(config.pro_config)

    @staticmethod
    def distance(m, n):
        _m = torch.mean(m, dim=2, keepdim=True)
        _n = torch.mean(n, dim=1, keepdim=True)
        return _m - _n

    def forward(self, mol_input, pro_input, src_coord=None, src_distance=None, tgt_coord=None, tgt_dis=None,
                tgt_score=None, noise_indices=None):
        mol_outputs = self.molecule_encoder(**mol_input)
        pro_outputs = self.pocket_encoder(**pro_input)

        _src_tokens = torch.cat([mol_input['src_tokens'], pro_input['src_tokens']], dim=1)
        padding_mask = _src_tokens.eq(0)
        mol_seq_rep = self.mol_pooler(mol_outputs.last_hidden_state)
        pro_seq_rep = self.pro_pooler(pro_outputs.last_hidden_state)
        mol_pair_rep = self.mol_pair_pooler(mol_outputs.last_pair_repr)
        pro_pair_rep = self.pro_pair_pooler(pro_outputs.last_pair_repr)
        bsz, seq_len = _src_tokens.shape
        _, mol_len = mol_seq_rep.shape[:2]
        seq_rep = torch.cat([mol_seq_rep, pro_seq_rep], dim=1)
        pair_rep = torch.zeros(bsz, seq_len, seq_len, self.config.num_attention_heads, device=_src_tokens.device)
        pair_rep[:, :mol_len, :mol_len] += mol_pair_rep
        pair_rep[:, mol_len:, mol_len:] += pro_pair_rep
        pair_rep[:, :mol_len, mol_len:] += self.distance(mol_pair_rep, pro_pair_rep)
        pair_rep[:, mol_len:, :mol_len] += self.distance(pro_pair_rep, mol_pair_rep)
        pair_rep = (pair_rep + pair_rep.transpose(1, 2)) / 2

        binding_outputs = self.binding_encoder(seq_rep=seq_rep, pair_rep=pair_rep, padding_mask=padding_mask)

        score = self.classifier2(binding_outputs.last_hidden_state[:, 0]).squeeze(-1)
        pair_repr = binding_outputs.last_pair_repr


        loss = torch.zeros([], device=_src_tokens.device)
        score_loss = None
        coord_loss = None
        dis_loss = None
        if tgt_score is not None:

            criterion = nn.SmoothL1Loss()
            indices = tgt_score.ne(-100)
            score_loss = criterion(score[indices], tgt_score[indices])
            loss = loss + score_loss

        if tgt_dis is not None:
            distance = self.dist_head(pair_repr)
            indices = noise_indices.unsqueeze(-1) & ~padding_mask.unsqueeze(1) & (tgt_dis[:, :mol_len] != 0)
            criterion = nn.SmoothL1Loss()
            tgt = (tgt_dis[:, :mol_len][indices] - self.dis_mean) / self.dis_std
            dis_loss = criterion(distance[:, :mol_len][indices], tgt)
            loss = loss + dis_loss

        x_norm_loss = (mol_outputs.x_norm + pro_outputs.x_norm + binding_outputs.x_norm) / 3
        if self.config.x_norm_loss > 0:
            loss = loss + self.config.x_norm_loss * x_norm_loss

        pair_norm_loss = (mol_outputs.delta_encoder_pair_rep_norm +
                          pro_outputs.delta_encoder_pair_rep_norm +
                          binding_outputs.delta_encoder_pair_rep_norm) / 3
        if self.config.delta_pair_repr_norm_loss > 0:
            loss = loss + self.config.delta_pair_repr_norm_loss * pair_norm_loss

        if (tgt_score is None) and (tgt_coord is None) and (tgt_dis is None):
            loss = None
        return UniMolBindingPoseOutput(loss=loss,
                                       score_loss=score_loss,
                                       distance_loss=dis_loss,
                                       coordinate_loss=coord_loss,
                                       score=score,
                                       x_norm=x_norm_loss,
                                       delta_encoder_pair_rep_norm=pair_norm_loss)


class UniMolForDockingPose(UniMolPreTrainedModel):
    def __init__(self, config: UnimolConfig):
        super(UniMolForDockingPose, self).__init__(config)
        self.config = config
        self.mol_model = UniMolModel(config.mol_config)
        self.pocket_model = UniMolModel(config.pocket_config)
        self.concat_encoder = UniMolModel(config)
        self.cross_distance_project = NonLinearHead(
            config.hidden_size * 2 + config.num_attention_heads, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            config.hidden_size + config.num_attention_heads, "relu"
        )

    def forward(
            self,
            mol_src_tokens,
            mol_src_distance,
            mol_src_edge_type,
            pocket_src_tokens,
            pocket_src_distance,
            pocket_src_edge_type,
            masked_tokens=None,
            distance_target=None,
            holo_distance_target=None,
            dist_threshold=0,
            **kwargs
    ):
        mol_padding_mask = mol_src_tokens.eq(0)
        pocket_padding_mask = pocket_src_tokens.eq(0)

        mol_outputs = self.mol_model(src_tokens=mol_src_tokens, src_distance=mol_src_distance,
                                     src_edge_type=mol_src_edge_type)
        mol_encoder_rep = mol_outputs.last_hidden_state
        mol_encoder_pair_rep = mol_outputs.last_pair_repr

        pocket_outputs = self.pocket_model(src_tokens=pocket_src_tokens, src_distance=pocket_src_distance,
                                           src_edge_type=pocket_src_edge_type)
        pocket_encoder_rep = pocket_outputs.last_hidden_state
        pocket_encoder_pair_rep = pocket_outputs.last_pair_repr

        bsz, mol_sz = mol_encoder_rep.shape[:2]
        pocket_sz = pocket_encoder_rep.size(1)

        concat_rep = torch.cat(
            [mol_encoder_rep, pocket_encoder_rep], dim=-2
        )  # [batch, mol_sz+pocket_sz, hidden_dim]
        concat_mask = torch.cat(
            [mol_padding_mask, pocket_padding_mask], dim=-1
        )  # [batch, mol_sz+pocket_sz]

        concat_pair_rep = torch.zeros(bsz, mol_sz + pocket_sz, mol_sz + pocket_sz, self.config.num_attention_heads,
                                      device=mol_src_tokens.device)
        concat_pair_rep[:, :mol_sz, :mol_sz] += mol_encoder_pair_rep
        concat_pair_rep[:, mol_sz:, mol_sz:] += pocket_encoder_pair_rep

        decoder_rep = concat_rep
        decoder_pair_rep = concat_pair_rep
        for i in range(self.config.recycling):
            binding_outputs = self.concat_encoder(seq_rep=decoder_rep, pair_rep=decoder_pair_rep,
                                                  padding_mask=concat_mask)
            decoder_rep = binding_outputs.last_hidden_state
            decoder_pair_rep = binding_outputs.last_pair_repr

        mol_decoder = decoder_rep[:, :mol_sz]
        pocket_decoder = decoder_rep[:, mol_sz:]

        mol_pair_decoder_rep = decoder_pair_rep[:, :mol_sz, :mol_sz, :]
        mol_pocket_pair_decoder_rep = (decoder_pair_rep[:, :mol_sz, mol_sz:, :] + decoder_pair_rep[:, mol_sz:, :mol_sz,
                                                                                  :].transpose(1, 2)) / 2.0
        mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float("-inf")] = 0

        cross_rep = torch.cat(
            [
                mol_pocket_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, pocket_sz, 4*hidden_size]

        cross_distance_predict = (
                F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
        )  # batch, mol_sz, pocket_sz

        holo_encoder_pair_rep = torch.cat(
            [
                mol_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, mol_sz, 3*hidden_size]
        holo_distance_predict = self.holo_distance_project(holo_encoder_pair_rep)  # batch, mol_sz, mol_sz

        distance_mask = distance_target.ne(0)  # 0 is padding
        if dist_threshold > 0:
            distance_mask &= (distance_target < dist_threshold)
        distance_predict = cross_distance_predict[distance_mask]
        distance_target = distance_target[distance_mask]
        distance_loss = F.mse_loss(distance_predict.float(), distance_target.float(), reduction="mean")

        ### holo distance loss
        holo_distance_mask = holo_distance_target.ne(0)  # 0 is padding
        holo_distance_predict_train = holo_distance_predict[holo_distance_mask]
        holo_distance_target = holo_distance_target[holo_distance_mask]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict_train.float(),
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
        )

        loss = distance_loss + holo_distance_loss
        return DockingPoseOutput(
            loss=loss,
            cross_loss=distance_loss,
            holo_loss=holo_distance_loss,
            cross_distance_predict=cross_distance_predict,
            holo_distance_predict=holo_distance_predict
        )


def base_architecture(cfg):
    if os.path.exists(os.path.join(cfg.MODEL.CHECKPOINT_PATH, 'config.json')):
        config = UnimolConfig.from_pretrained(cfg.MODEL.CHECKPOINT_PATH)
    else:
        config = UnimolConfig()
    return config
