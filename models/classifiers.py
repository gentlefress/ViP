from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss


class SimpleClassifier(nn.Module):
    def __init__(self, input_size=768, num_labels=1):
        super(SimpleClassifier, self).__init__()
        self.classifier = nn.Linear(input_size, num_labels)
        self.num_labels = num_labels

    def forward(self, x, labels):
        logits = self.classifier(x)
        outputs = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
