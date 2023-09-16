from models.text_encoder import *
from models.image_encoders import *
from utils.pre_model import RobertaEncoder
import math
import torch
import torch.nn as nn

class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.fuse_type = opt.fuse_type
        self.image_output_type = opt.image_output_type
        self.zoom_value = math.sqrt(opt.tran_dim)
        self.save_image_index = 0

        self.text_config = copy.deepcopy(opt)
        # self.image_config = copy.deepcopy(self.text_model.get_config())

        self.text_config.num_attention_heads = opt.tran_dim // 64
        self.text_config.hidden_size = opt.tran_dim
        self.text_config.num_hidden_layers = opt.tran_num_layers


        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False

        self.text_image_encoder = RobertaEncoder(self.text_config)

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_cls_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )

        self.transformer_embedding_layernorm = nn.Sequential(
            nn.LayerNorm(opt.tran_dim),
            nn.Dropout(opt.l_dropout)
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.tran_dim, nhead=opt.tran_dim//64, dim_feedforward=opt.tran_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=opt.tran_num_layers)

        if self.fuse_type == 'att':
            self.output_attention = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 1)
            )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 3)
        )

    def forward(self, text_init, image_init):
        text_image_cat = torch.cat((text_init, image_init), dim=1)
        extended_attention_mask = get_extended_attention_mask(text_image_mask, text_inputs.size())
        text_image_transformer = self.text_image_encoder(text_image_cat,
                                                 attention_mask=extended_attention_mask,
                                                 head_mask=None,
                                                 encoder_hidden_states=None,
                                                 encoder_attention_mask=extended_attention_mask,
                                                 past_key_values=None,
                                                 use_cache=self.use_cache,
                                                 output_attentions=self.text_config.output_attentions,
                                                 output_hidden_states=(self.text_config.output_hidden_states),
                                                 return_dict=self.text_config.use_return_dict)
        text_image_transformer = text_image_transformer.last_hidden_state
        text_image_transformer = text_image_transformer.permute(0, 2, 1).contiguous()

        if self.fuse_type == 'max':
            text_image_output = torch.max(text_image_transformer, dim=2)[0]
        elif self.fuse_type == 'att':
            text_image_output = text_image_transformer.permute(0, 2, 1).contiguous()

            text_image_mask = text_image_mask.permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0:text_image_output.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()

            text_image_alpha = self.output_attention(text_image_output)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)

            text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)
        elif self.fuse_type == 'ave':
            text_image_length = text_image_transformer.size(2)
            text_image_output = torch.sum(text_image_transformer, dim=2) / text_image_length
        else:
            raise Exception('fuse_type设定错误')
        return text_image_output, None, None