import copy
import itertools
from typing import Optional
import torch.nn.functional as F
import torch
from torch.distributions import Normal
from src.modeling.graph_seq import GNN_emb, MLP 
from src.layers.modeling_outputs import DockingPoseOutput, SDDockingPoseOutput
from src.layers.openfold.modeling_evoformer import TriAtten
from src.layers.modeling_hf_unimol_layers import TransformerEncoderLayer
import torch.nn as nn

from typing import Optional         
from torch_geometric.data import Batch 
from src.modeling.modeling_hf_unimol import UniMolModel, UnimolConfig, UniMolForDockingPose, UniMolForPretraining, \
    NonLinearHead, DistanceHead


class TransformerEncoderWithEvo(nn.Module):
    def __init__(
            self,
            encoder_layers: int = 6,
            embed_dim: int = 768,
            ffn_embed_dim: int = 3072,
            attention_heads: int = 8,
            emb_dropout: float = 0.1,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.0,
            max_seq_len: int = 256,
            activation_fn: str = "gelu",
            post_ln: bool = False,
            no_final_head_layer_norm: bool = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = nn.LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = nn.LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.folds = TriAtten(c_m=self.embed_dim, c_z=attention_heads, c_hidden_opm=attention_heads,
                              c_hidden_mul=attention_heads, c_hidden_pair_att=attention_heads, no_heads_pair=2)
        self.x_ln = nn.LayerNorm(self.embed_dim)
        self.pair_ln = nn.LayerNorm(attention_heads)

    def forward(
            self,
            emb: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)
        if padding_mask is not None:
            # [bsz, n]
            fold_msa_mask = (~padding_mask).float()
            fold_pair_mask = fold_msa_mask.unsqueeze(1) * fold_msa_mask.unsqueeze(-1)
            fold_msa_mask = fold_msa_mask.unsqueeze(1)
        else:
            fold_msa_mask = torch.ones(bsz, 1, seq_len, dtype=torch.long, device=emb.device)
            fold_pair_mask = torch.ones(bsz, seq_len, seq_len, dtype=torch.long, device=emb.device)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None

        # attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        def transpose_pair(pair_repr, bsz, seq_len, c):
            return pair_repr.view(bsz, c, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()

        def de_transpose_pair(pair_repr, bsz, seq_len, c):
            return pair_repr.permute(0, 3, 1, 2).contiguous().view(bsz * c, seq_len, seq_len)

        m_pair_repr = attn_mask
        for i in range(len(self.layers)):
            x, _, attn_probs = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )
            m_pair_repr += attn_probs
        m_pair_repr[m_pair_repr == float('-inf')] = 0
        m_pair_repr = (
            m_pair_repr.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )

        x, m_pair_repr = self.folds(
            (x.unsqueeze(1), m_pair_repr),
            fold_msa_mask,
            fold_pair_mask,
            # use_lma=True
        )
        x = x.squeeze(1)
        # m_pair_repr = m_pair_repr / 6
        attn_mask = de_transpose_pair(m_pair_repr, bsz, seq_len, self.attention_heads)

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x ** 2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                    torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        token_mask = 1.0 - input_padding_mask.float()
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm


class FoldModel(UniMolModel):
    def __init__(self, config: UnimolConfig):
        super().__init__(config)
        self.encoder = TransformerEncoderWithEvo(
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
            no_final_head_layer_norm=config.delta_pair_repr_norm_loss < 0,
            post_ln=config.post_ln,
        )


class FoldForDockingPose(UniMolForDockingPose):
    def __init__(self, config: UnimolConfig):
        super(FoldForDockingPose, self).__init__(config)
        self.mol_model = FoldModel(config.mol_config)
        self.pocket_model = FoldModel(config.pocket_config)
        self.concat_encoder = FoldModel(config)

    @staticmethod
    def distance(m, n):
        _m = torch.mean(m, dim=2, keepdim=True)
        _n = torch.mean(n, dim=1, keepdim=True)
        return _m - _n

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
        # concat_pair_rep[:, :mol_sz, mol_sz:] += self.distance(mol_encoder_pair_rep, pocket_encoder_pair_rep)
        # concat_pair_rep[:, mol_sz:, :mol_sz] += self.distance(pocket_encoder_pair_rep, mol_encoder_pair_rep)
        # concat_pair_rep = (concat_pair_rep + concat_pair_rep.transpose(1, 2)) / 2

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



class RtmScoreHead(nn.Module):
    def __init__(self, hidden_size, n_gaussian=10):
        super().__init__()
        # self.z_pi = nn.Linear(hidden_size, n_gaussian)
        # self.z_sigma = nn.Linear(hidden_size, n_gaussian)
        # self.z_mu = nn.Linear(hidden_size, n_gaussian)
        self.z_pi = NonLinearHead(hidden_size, n_gaussian, 'relu')
        self.z_sigma = NonLinearHead(hidden_size, n_gaussian, 'relu')
        self.z_mu = NonLinearHead(hidden_size, n_gaussian, 'relu')

    def forward(self, features, dist, dist_mask=None, eps=1e-6, threshold=5, pred_dist=None, isomorphisms=None):
        """
        feature: [BSZ, N, M, D]
        dist: [BSZ, N, M]
        """
        if dist_mask is None:
            dist_mask = torch.ones_like(dist, dtype=torch.bool, device=dist.device)

        # _pi = self.z_pi(features)
        # pi = F.softmax(_pi, -1)
        pi = F.softmax(self.z_pi(features), -1) 
        sigma = F.elu(self.z_sigma(features)) + 1.3
        mu = F.elu(self.z_mu(features)) + 1
        _pred_dist = mu.mean(dim=-1)

        # [BSZ, N, M, 10]
        if isomorphisms is None:
            dist_mask = dist_mask & ((dist <= threshold) | (_pred_dist <= (threshold - 2)))
            score, loss = self.compute_score_and_loss(mu, sigma, pi, dist, dist_mask, eps=eps)
        else:
            bsz = features.shape[0]
            losses = [torch.inf] * bsz
            scores = [None] * bsz
            for _idx in range(bsz):
                for iso in isomorphisms[_idx]:
                    _dist = dist[_idx][iso, :]
                    _dist_mask = dist_mask[_idx] & ((_dist <= threshold) | (_pred_dist[_idx] <= (threshold - 2)))
                    _score, _loss = self.compute_score_and_loss(mu[_idx], sigma[_idx], pi[_idx], _dist, _dist_mask,
                                                                eps=eps)
                    if _loss < losses[_idx]:
                        losses[_idx] = _loss
                        scores[_idx] = _score
            loss = torch.stack(losses).mean()
            score = torch.stack(scores)

        return score, loss, (pi, mu, sigma)

    @staticmethod
    def compute_score_and_loss(mu, sigma, pi, dist, dist_mask, eps=1e-6):
        normal = Normal(mu, sigma)
        loglik = normal.log_prob(dist.unsqueeze(-1))
        candidate_loss = -torch.logsumexp(torch.log(pi + eps) + loglik, dim=-1)
        loss = candidate_loss[dist_mask].mean()

        prob = loglik.exp() * pi / (dist ** 2 + eps).unsqueeze(-1)
        prob = prob.sum(-1)
        score = torch.stack([p[m].sum() for p, m in zip(prob, dist_mask)])
        return score, loss

    @staticmethod
    def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
        normal = Normal(mu, sigma)
        # loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
        # loss = th.sum(loss * pi, dim=1)
        # loss = -th.log(loss)
        loglik = normal.log_prob(y.expand_as(normal.loc))
        loss = -torch.logsumexp(torch.log(pi + eps) + loglik, dim=1)
        return loss


class GerNA_RNAModule_Core(nn.Module):
    def __init__(self, input_dim_rna=771, GNN_depth=3, hidden_size1=128, fusion_grad_frac=0.5):
        super().__init__()
        
        self.input_dim_rna = input_dim_rna
        self.hidden_size1 = hidden_size1
        self.fusion_grad_frac = fusion_grad_frac
        
        # 1. 编码器 (Encoders)
        # 对应 GerNA: self.GCN_rna
        self.GCN_rna = GNN_emb(input_dim_rna, GNN_depth, hidden_size1, 
                                edge_attr_option=False, gnn_type='gcn')
        
        # 对应 GerNA: self.mlp_rna
        self.mlp_rna = MLP(input_dim_rna, hidden_size1)
        
        # 2. 投影层 (Projections) -用于对齐语义空间
        # 对应 GerNA: self.pairwise_rna_2d
        self.pairwise_rna_2d = nn.Linear(hidden_size1, hidden_size1)
        
        # 对应 GerNA: self.pairwise_rna_1d
        self.pairwise_rna_1d = nn.Linear(hidden_size1, hidden_size1)

    def forward(self, rna_graph_batch):
        """
        输入: PyG Batch 对象
        输出: [Total_Nodes, hidden_size1] 融合后的特征
        """
        # --- Step A: 独立编码 ---
        
        # 2D Path: GCN
        # RNA_2d_fea: [Total_Nodes, H]
        rna_2d_fea = self.GCN_rna(rna_graph_batch)
        
        # 1D Path: MLP
        # RNA_1d_fea: [Total_Nodes, H]
        rna_1d_fea = self.mlp_rna(rna_graph_batch.x)
        
        # --- Step B: 投影与激活 (Projection & Activation) ---
        # 对应 GerNA: F.leaky_relu(self.pairwise_rna_2d(RNA_2d_fea), 0.1)
        proj_2d = F.leaky_relu(self.pairwise_rna_2d(rna_2d_fea), 0.1)
        
        # 对应 GerNA: F.leaky_relu(self.pairwise_rna_1d(RNA_1d_fea), 0.1)
        proj_1d = F.leaky_relu(self.pairwise_rna_1d(rna_1d_fea), 0.1)
        
        # --- Step C: 加权融合 (Fusion) ---
        # 对应 GerNA: fusion_frac * 2D + (1-fusion_frac) * 1D
        # 注意：GerNA 在这里使用了投影后的特征进行融合
        rna_fused = self.fusion_grad_frac * proj_2d + (1 - self.fusion_grad_frac) * proj_1d
        
        return rna_fused

##ls revise
class FoldForDocking(UniMolForDockingPose):
    def __init__(self, config: UnimolConfig):
        super(FoldForDocking, self).__init__(config)
        self.mol_model = FoldModel(config.mol_config)
        self.pocket_model = FoldModel(config.pocket_config)
        self.concat_encoder = FoldModel(config)
       
        self.rna_input_dim = getattr(config, "rna_input_dim", 771)
        self.rna_gcn_depth = getattr(config, "rna_gcn_depth", 3)
        self.rna_hidden_dim = getattr(config, "rna_embed_dim", 128)


        self.rna_module_core = GerNA_RNAModule_Core(
            input_dim_rna=self.rna_input_dim,
            GNN_depth=self.rna_gcn_depth,
            hidden_size1=self.rna_hidden_dim,
            fusion_grad_frac=0.5
        )
                
       # 3. 计算拼接后的总维度
        base_cross_rep_dim = config.num_attention_heads + config.hidden_size * 2
        base_holo_rep_dim = config.num_attention_heads + config.hidden_size
        
        # 拼接维度 = 原始 + RNA_Hidden (128)
        fused_cross_rep_dim = base_cross_rep_dim + self.rna_hidden_dim
        fused_holo_rep_dim = base_holo_rep_dim + self.rna_hidden_dim
        

        
        # <--- 3. 重新定义预测头，以接受新的、更大的输入维度 ---
        self.regression_head2 = NonLinearHead(fused_cross_rep_dim, 3, 'relu')
        self.regression_head = NonLinearHead(fused_cross_rep_dim, 1, 'relu')
        self.rtm_score_head = RtmScoreHead(fused_cross_rep_dim, 10)
        self.cross_distance_project = NonLinearHead(fused_cross_rep_dim, 1, 'relu')
        self.holo_distance_project = NonLinearHead(fused_holo_rep_dim, 1, 'relu')
        #self.regression_head2 = NonLinearHead(2 * config.hidden_size + config.num_attention_heads, 3, 'relu')  
        #self.regression_head = NonLinearHead(2 * config.hidden_size + config.num_attention_heads, 1, 'relu')
        #self.rtm_score_head = RtmScoreHead(config.hidden_size * 2 + config.num_attention_heads, 10)
        ##ls revise end
    @staticmethod
    def distance(m, n):
        _m = torch.mean(m, dim=2, keepdim=True)
        _n = torch.mean(n, dim=1, keepdim=True)
        return _m - _n

    @staticmethod
    def get_k_nearest_mask(target: torch.Tensor, k=10):
        with torch.no_grad():
            mask = target.eq(0)
            _target = target.clone()
            _target[mask] = _target[mask] + 1000
            indices = _target.topk(k, dim=-1, largest=False)[1]
            rst_mask = torch.zeros_like(target, device=target.device, dtype=torch.bool)
            ind_list = itertools.product(*[list(range(i)) for i in target.shape[:-1]])
            for item in ind_list:
                rst_mask[item][indices[item]] = True
            rst_mask = rst_mask & (~mask)
        return rst_mask

    def get_k_nearest_mask_ensemble(self, target: torch.Tensor, predict: torch.Tensor, base_k=4):
        m1 = self.get_k_nearest_mask(target, k=base_k * 4)
        m2 = self.get_k_nearest_mask(predict, k=base_k * 2)
        m3 = self.get_k_nearest_mask(target.transpose(-1, -2), k=2).transpose(-1, -2)
        m4 = self.get_k_nearest_mask(predict.transpose(-1, -2), k=1).transpose(-1, -2)
        m5 = target.ne(0)
        mask = (m1 | m2 | m3 | m4) & m5
        return mask

    def load_state_dict(self, state_dict, strict=True, remove_prefix='model.'):
        new_state_dict = dict()
        if len(remove_prefix) > 0:
            for k, v in state_dict.items():
                if k.startswith(remove_prefix):
                    new_k = k[len(remove_prefix):]
                else:
                    new_k = k
                new_state_dict[new_k] = v

        else:
            new_state_dict = state_dict

        tmp_dict = dict()
        for k in new_state_dict.keys():
            if 'holo_distance_project' in k:
                hop_ks = [k.replace('holo_distance_project', f'holo_{i + 1}hop_proj') for i in range(3)]
                for hop_k in hop_ks:
                    if hop_k not in new_state_dict.keys():
                        tmp_dict[hop_k] = copy.deepcopy(new_state_dict[k])
        new_state_dict.update(tmp_dict)

        super(FoldForDocking, self).load_state_dict(new_state_dict, strict)

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
            rna_batch_data: Optional[Batch] = None, # PyG Batch      
            atom_to_res_map: Optional[torch.Tensor] = None,  # [B, N_pocket] 
            holo_distance_target=None,
            dist_threshold=0,
            score=None,
            hop_matrix=None,
            isomorphisms=None,
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
        concat_pair_rep[:, :mol_sz, mol_sz:] += self.distance(mol_encoder_pair_rep, pocket_encoder_pair_rep)
        concat_pair_rep[:, mol_sz:, :mol_sz] += self.distance(pocket_encoder_pair_rep, mol_encoder_pair_rep)
        # concat_pair_rep = (concat_pair_rep + concat_pair_rep.transpose(1, 2)) / 2

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
        mol_pair_decoder_rep = (mol_pair_decoder_rep + mol_pair_decoder_rep.transpose(1, 2)) / 2.0
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
        )
        holo_encoder_pair_rep = torch.cat(
            [
                mol_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1),
            ],
            dim=-1,
        )

         ## --- ls revise begin ---
        if rna_batch_data is not None and atom_to_res_map is not None:
           
            # Output: [Total_Nodes, rna_hidden_dim]
            full_rna_fused_feats = self.rna_module_core(rna_batch_data)
            
            # 提取 Pocket 对应的残基特征 (Map & Gather)
            pocket_rna_feats_list = []
            ptr = rna_batch_data.ptr # Batch 偏移量 [0, L1, L1+L2...]
            
            for i in range(bsz):
                # 1. offset: 当前样本在 full_feats 中的起始行
                # 当前 map: [N_pocket+2] (含 BOS/EOS padding)(包含-1表示非核酸原子)
                offset = ptr[i]
                
                # 2. 获取 Atom -> Residue 的映射 # local_indices: [N_pocket+2] (包含 -1 表示非核酸原子)
                local_indices = atom_to_res_map[i] 
                # a. 生成有效性 Mask (Valid Mask)
                # True 表示是核酸原子，False 表示是蛋白质(-1)或异常值
                # 注意：假设 LMDB 里用 -1 标记 Protein
                valid_rna_mask = local_indices.ge(0) # index >= 0 为有效   ##这生成了一个布尔掩码索引 -1 变成 False。
                # b. 生成安全索引 (Safe Indices)
                # 将 -1 (或任何负数) 临时替换为 0，防止 Index Out of Bounds 报错
                # 这里的 0 指向当前 RNA 的第 0 个残基，提取出的特征稍后会被 Mask 掉，所以没关系,这是防止程序崩溃的关键。如果直接用含 -1 的 Tensor 去索引，CUDA 会报错。我们把 -1 偷梁换柱改成 0，让它去取一个“合法的假数据”。
                safe_local_indices = torch.where(valid_rna_mask, local_indices, torch.tensor(0, device=local_indices.device))
                
                # 3. 转换为全局索引
                global_indices = safe_local_indices + offset
                
                # 4. 提取特征
                # mapped_feat: [N_pocket+2, 128],此时，原 -1 位置提取到了第 0 个残基的特征（这是脏数据）
                mapped_feat = full_rna_fused_feats[global_indices]
                # 5. 应用掩码 (Apply Mask)
                # 将无效位置 (原 -1) 的特征彻底置零
                # valid_rna_mask.unsqueeze(-1): [N_pocket+2, 1]
                mapped_feat = mapped_feat * valid_rna_mask.unsqueeze(-1).type_as(mapped_feat)


                pocket_rna_feats_list.append(mapped_feat)
            
            # 堆叠回 Batch Tensor: [B, pocket_sz, rna_hidden_dim]
            rna_pocket_tensor = torch.stack(pocket_rna_feats_list, dim=0)
            
            # Step C: 融合到 Cross Rep (Broadcast)
            # rna_pocket_tensor: [B, pocket_sz, H]
            # 扩展到: [B, 1, pocket_sz, H] -> [B, mol_sz, pocket_sz, H]
            broadcast_cross = rna_pocket_tensor.unsqueeze(1).expand(-1, mol_sz, -1, -1)
            
            # 拼接
            cross_rep = torch.cat([cross_rep, broadcast_cross], dim=-1)
            
            # D. 广播并融合 Holo Rep (Context Injection)
            # 计算 Mean 时要注意！不能把被 Mask 掉的 0 算进去拉低均值
            # 应该只对 valid_rna_mask 为 True 的部分求平均
            
            # 生成 Batch 级别的 Mask: [B, pocket_sz, 1]
            batch_valid_mask = atom_to_res_map.ge(0).unsqueeze(-1).type_as(rna_pocket_tensor)
            
            # Sum Pooling
            sum_feats = torch.sum(rna_pocket_tensor, dim=1, keepdim=True) # [B, 1, 128]
            # Count Valid Nodes
            # clamp(min=1) 防止除以 0
            valid_counts = torch.sum(batch_valid_mask, dim=1, keepdim=True).clamp(min=1) # [B, 1, 1]
            
            # Mean Pooling (Safe)
            rna_context = sum_feats / valid_counts
            # === [FIX START] ===
            # 必须先升维到 [B, 1, 1, 128] 才能 expand 到 [B, mol, mol, 128]
            rna_context = rna_context.unsqueeze(1) 
            # === [FIX END] ===
            # Broadcast & Concat
            broadcast_holo = rna_context.expand(-1, mol_sz, mol_sz, -1)
            holo_encoder_pair_rep = torch.cat([holo_encoder_pair_rep, broadcast_holo], dim=-1)
            
        else:
            # 缺失数据时的 Padding 处理 (防止维度报错)
            dummy_cross = torch.zeros(bsz, mol_sz, pocket_sz, self.rna_hidden_dim, 
                                      device=cross_rep.device, dtype=cross_rep.dtype)
            cross_rep = torch.cat([cross_rep, dummy_cross], dim=-1)
            
            dummy_holo = torch.zeros(bsz, mol_sz, mol_sz, self.rna_hidden_dim, 
                                     device=holo_encoder_pair_rep.device, dtype=holo_encoder_pair_rep.dtype)
            holo_encoder_pair_rep = torch.cat([holo_encoder_pair_rep, dummy_holo], dim=-1)
        
        ## --- ls revise end ---

        # 4. 预测 (使用增强后的表征)
        cross_distance_predict = (
                F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
        )
        holo_distance_predict = self.holo_distance_project(holo_encoder_pair_rep).squeeze(-1)
        holo_distance_predict = F.elu(holo_distance_predict) + 1

         # ===== rtm_score + distance loss =====
        if distance_target is None:
            # --------- 推理 / screening 模式 ---------
            # 用 token mask 构造 distance_mask
            distance_mask = mol_padding_mask.unsqueeze(-1) & pocket_padding_mask.unsqueeze(-2)
            ##ls 
            # rtm_score_head 用模型预测的 cross_distance_predict
            rtm_score, rtm_loss, mdn = self.rtm_score_head(
                cross_rep,
                cross_distance_predict,
                distance_mask,
                threshold=5,
            )##这里可以开6或者8试试

            # 推理时不算显式的距离 MSE loss，置 0（标量或 None 视你的返回结构而定）
            distance_loss = cross_distance_predict.new_zeros(())
        else:
            # --------- 训练模式，有真实 distance_target ---------
            # rtm_score 部分
            rtm_score, rtm_loss, mdn = self.rtm_score_head(
                cross_rep,
                distance_target,
                distance_target.ne(0),
                # threshold=dist_threshold,
            )

            # 显式的 MSE 距离 loss
            distance_mask = distance_target.ne(0)  # 0 is padding
            if dist_threshold > 0:
                distance_mask &= (distance_target < dist_threshold)

            distance_predict = cross_distance_predict[distance_mask]
            distance_target_train = distance_target[distance_mask]
            distance_loss = F.mse_loss(
                distance_predict.float(),
                distance_target_train.float(),
                reduction="mean",
            )

      
        return DockingPoseOutput(
            #loss=loss,
            score_loss=rtm_loss,
            #affinity_loss=affinity_loss,
            #cross_loss=distance_loss,
            #holo_loss=holo_distance_loss,
            #affinity_predict=affinity_pred,
            rtm_score=rtm_score,
            cross_distance_predict=cross_distance_predict,
            holo_distance_predict=holo_distance_predict,
            mdn=mdn,
        )

    def compute_distance_loss(self, cross_distance_predict, distance_target, holo_distance_predict,
                              holo_distance_target, dist_threshold=8):
        distance_mask = distance_target.ne(0)  # 0 is padding
        if dist_threshold > 0:
            distance_mask &= ((distance_target < dist_threshold) | (cross_distance_predict < dist_threshold - 2))
        distance_predict = cross_distance_predict[distance_mask]
        distance_target_train = distance_target[distance_mask]
        distance_loss = F.smooth_l1_loss(distance_predict.float(), distance_target_train.float())

        holo_distance_mask = holo_distance_target.ne(0)  # 0 is padding
        holo_distance_predict_train = holo_distance_predict[holo_distance_mask]
        holo_distance_target_train = holo_distance_target[holo_distance_mask]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict_train.float(),
            holo_distance_target_train.float(),
            reduction="mean",
            beta=1.0,
        )
        return distance_loss, holo_distance_loss



