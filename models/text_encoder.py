from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
# from transformers.models.deberta.modeling_deberta import DebertaModel, DebertaOnlyMLMHead, DebertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaLMHead, RobertaForMaskedLM
# from transformers.models.bert.modeling_bert import BertEncoder as BEncoder

import torch.nn as nn
import torch
import copy


# class BertEncoder(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super(BertEncoder, self).__init__(config)
#         self.config = config
#         self.bert = RobertaModel(config)
#         self.cls = RobertaLMHead(config)
#         self.conv1d = nn.Conv1d(32, 32, 1)
#         self.init_weights()
#         self.tie_weights()
#
#     def tie_weights(self):
#         """ Make sure we are sharing the input and output embeddings.
#             Export to TorchScript can't handle parameter sharing so we are cloning them instead.
#         """
#         self._tie_or_clone_weights(self.cls.decoder,
#                                    self.bert.embeddings.word_embeddings)
#     def get_config(self):
#         return self.config
#
#     def forward(self, input_ids=None, image=None, attention_mask=None, token_type_ids=None,
#                 position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
#         model_outputs = self.bert(input_ids,
#                                   attention_mask=attention_mask,
#                                   token_type_ids=token_type_ids,
#                                   position_ids=position_ids,
#                                   head_mask=head_mask,
#                                   inputs_embeds=inputs_embeds)
#         text_features, global_text_features = model_outputs[0], model_outputs[1]
#         text_features = self.conv1d(text_features)
#         prediction_scores = self.cls(text_features)
#         outputs = (prediction_scores,
#                    text_features, global_text_features,
#                    text_features, global_text_features)
#
#         if self.config.output_hidden_states is True:
#             outputs = outputs + (model_outputs[2],)
#
#         return outputs
class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.conv1d = nn.Conv1d(32, 32, 1)
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)
    def get_config(self):
        return self.config

    def forward(self, input_ids=None, image=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        model_outputs = self.bert(input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds)
        text_features, global_text_features = model_outputs[0], model_outputs[1]
        text_features_conv = self.conv1d(text_features)
        prediction_scores = self.cls(text_features)
        outputs = (prediction_scores,
                   text_features, global_text_features,
                   text_features_conv, global_text_features)

        if self.config.output_hidden_states is True:
            outputs = outputs + (model_outputs[2],)

        return outputs
