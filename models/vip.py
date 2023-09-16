import torch
import torch.nn as nn
from utils.pretrain_config import cfg
from utils.GlobalAttention import func_attention, func_attention_textual
import  torch.nn.functional as F
# from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def cosine(x, y):
    s1 = torch.norm(x, p=2, dim=-1)
    s2 = torch.norm(y, p=2, dim=-1)
    return torch.matmul(s1, s2.t())
class vipUnidirectional(nn.Module):
    def __init__(self, project_size):
        self.project_size = project_size
        super(vipUnidirectional, self).__init__()
    @staticmethod
    def global_loss(cnn_code, cnn_code_aug, rnn_code, rnn_code_aug, labels, queue, queue_im, eps=1e-8):
        T = 0.05
        queue = queue.to('cuda:0')
        queue_im = queue_im.to('cuda:0')
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)
            rnn_code_aug = rnn_code_aug.unsqueeze(0)
            cnn_code_aug = cnn_code_aug.unsqueeze(0)
        """
            loss_t = -abs(cosine(rnn_code, cnn_code)[0][0].float() - cosine(rnn_spec, cnn_spec).float())
            loss_t = loss_t/(cnn_code.size(-1)//2)
        """

        cnn_code_norm = torch.norm(cnn_code, 2, dim=-1, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=-1, keepdim=True)
        rnn_code_aug_norm = torch.norm(rnn_code_aug, 2, dim=-1, keepdim=True)
        cnn_code_normalize = nn.functional.normalize(cnn_code, dim=-1)
        rnn_code_aug_normalize = nn.functional.normalize(rnn_code_aug, dim=-1)
        l_pos = torch.einsum('nc,nc->n', [cnn_code_normalize.squeeze(0), rnn_code_aug_normalize.squeeze(0)]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [cnn_code_normalize.squeeze(0), queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=-1)
        logits /= T
        label = torch.zeros(logits.size(0), dtype=torch.long).to('cuda:0')
        m_loss_it = nn.CrossEntropyLoss()(logits, label)

        rnn_code_normalize = nn.functional.normalize(rnn_code, dim=-1)
        cnn_code_aug_normalize = nn.functional.normalize(cnn_code_aug, dim=-1)
        l_pos = torch.einsum('nc,nc->n', [rnn_code_normalize.squeeze(0), cnn_code_aug_normalize.squeeze(0)]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [rnn_code_normalize.squeeze(0), queue_im.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=-1)
        logits /= T
        label = torch.zeros(logits.size(0), dtype=torch.long).to('cuda:0')
        m_loss_ti = nn.CrossEntropyLoss()(logits, label)
        m_loss = (m_loss_it + m_loss_ti) / 2
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        scores1 = torch.bmm(rnn_code, rnn_code_aug.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        norm1 = torch.bmm(rnn_code_norm, rnn_code_aug_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * cfg.SMOOTH.GAMMA3
        scores2 = scores1 / norm1.clamp(min=eps) * cfg.SMOOTH.GAMMA3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze()
        scores3 = scores0.transpose(0, 1)


        scores4 = scores2.squeeze()
        scores5 = scores2.transpose(0, 1)
        if labels is not None:
            # labels是噪声
            loss0 = nn.CrossEntropyLoss()(scores0, labels)
            loss1 = nn.CrossEntropyLoss()(scores3, labels)
            loss2 = nn.CrossEntropyLoss()(scores4, labels)
            loss3 = nn.CrossEntropyLoss()(scores5, labels)

        else:
            loss0, loss1, loss2, loss3 = None, None, None, None
        return loss0, loss1, m_loss, loss2, loss3
    @staticmethod
    def local_loss(img_features, words_emb, labels,
                   sentence_mask, batch_size):
        """
            words_emb(query): batch x nef x seq_len
            img_features(context): batch x nef x 17 x 17
        """
        att_maps = []
        similarities = []
        # att_maps_aug = []
        similarities_aug = []
        words_emb = words_emb.permute(0, 2, 1)
        # words_emd_aug = words_emd_aug.permute(0, 2, 1)


        for i in range(batch_size):
            # Get the i-th text description
            words_num = int(torch.sum(sentence_mask[i]))
            # words_num_aug = int(torch.sum(sentence_mask_aug[i]))
            # -> 1 x nef x words_num
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
            # word_aug = words_emd_aug[i, :, :words_num_aug].unsqueeze(0).contiguous()
            # -> batch_size x nef x words_num
            word = word.repeat(batch_size, 1, 1)
            # word_aug = word_aug.repeat(batch_size, 1, 1)
            # batch x nef x 7*7   nef=768
            context = img_features
            """
                word(query): batch x nef x words_num
                context: batch x nef x 7 x 7
                weiContext: batch x nef x words_num
                attn: batch x words_num x 7 x 7
            """
            weiContext, attn = func_attention(word, context, cfg.SMOOTH.GAMMA1)
            att_maps.append(attn[i].unsqueeze(0).contiguous())
            # --> batch_size x words_num x nef
            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            # --> batch_size*words_num x nef
            word = word.view(batch_size * words_num, -1)
            weiContext = weiContext.view(batch_size * words_num, -1)

            # -->batch_size*words_num
            row_sim = cosine_similarity(word, weiContext)
            # --> batch_size x words_num
            row_sim = row_sim.view(batch_size, words_num)

            row_sim.mul_(cfg.SMOOTH.GAMMA2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            # --> 1 x batch_size
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(row_sim)
            # similarities_aug.append(row_sim_aug)
        #
        # batch_size x batch_size
        similarities = torch.cat(similarities, 1)
        similarities = similarities * cfg.SMOOTH.GAMMA3
        similarities1 = similarities.transpose(0, 1)
        if labels is not None:
            loss0 = nn.CrossEntropyLoss()(similarities, labels)
            loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        else:
            loss0, loss1 = None, None
        return loss0, loss1, att_maps

    def forward(self, cnn_code, rnn_code, img_features, words_emb, cnn_code_aug, rnn_code_aug, queue, queue_im, sentence_mask, labels):
        batch_size = words_emb.size(0)
        w_loss0, w_loss1, attn_maps = self.local_loss(img_features, words_emb, labels,
                                                      sentence_mask=sentence_mask, batch_size=batch_size)
        s_loss0, s_loss1, m_loss, s_loss6, s_loss7 = self.global_loss(cnn_code, cnn_code_aug, rnn_code, rnn_code_aug, labels, queue, queue_im)
        return w_loss0, w_loss1, s_loss0, s_loss1, m_loss, s_loss6, s_loss7



class vipBidirectional(nn.Module):
    def __init__(self, project_size):
        self.project_size = project_size
        super(vipBidirectional, self).__init__()

    @staticmethod
    def global_loss(cnn_code, cnn_code_aug, rnn_code, rnn_code_aug, labels, queue, queue_im, eps=1e-8):
        # --> seq_len x batch_size x nef
        T = 0.07
        queue = queue.to('cuda:0')
        queue_im = queue_im.to('cuda:0')
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)
            rnn_code_aug = rnn_code_aug.unsqueeze(0)
            cnn_code_aug = cnn_code_aug.unsqueeze(0)
        """
            loss_t = -abs(cosine(rnn_code, cnn_code)[0][0].float() - cosine(rnn_spec, cnn_spec).float())
            loss_t = loss_t/(cnn_code.size(-1)//2)
        """

        cnn_code_norm = torch.norm(cnn_code, 2, dim=-1, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=-1, keepdim=True)
        rnn_code_aug_norm = torch.norm(rnn_code_aug, 2, dim=-1, keepdim=True)
        cnn_code_normalize = nn.functional.normalize(cnn_code, dim=-1)
        rnn_code_aug_normalize = nn.functional.normalize(rnn_code_aug, dim=-1)
        l_pos = torch.einsum('nc,nc->n', [cnn_code_normalize.squeeze(0), rnn_code_aug_normalize.squeeze(0)]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [cnn_code_normalize.squeeze(0), queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=-1)
        logits /= T
        label = torch.zeros(logits.size(0), dtype=torch.long).to('cuda:0')
        m_loss_it = nn.CrossEntropyLoss()(logits, label)

        rnn_code_normalize = nn.functional.normalize(rnn_code, dim=-1)
        cnn_code_aug_normalize = nn.functional.normalize(cnn_code_aug, dim=-1)
        l_pos = torch.einsum('nc,nc->n', [rnn_code_normalize.squeeze(0), cnn_code_aug_normalize.squeeze(0)]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [rnn_code_normalize.squeeze(0), queue_im.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=-1)
        logits /= T
        label = torch.zeros(logits.size(0), dtype=torch.long).to('cuda:0')
        m_loss_ti = nn.CrossEntropyLoss()(logits, label)
        m_loss = (m_loss_it + m_loss_ti) / 2
        # scores* / norm*: seq_len x batch_size x batch_size

        # --> batch_size x batch_size
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        scores1 = torch.bmm(rnn_code, rnn_code_aug.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        norm1 = torch.bmm(rnn_code_norm, rnn_code_aug_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * cfg.SMOOTH.GAMMA3
        scores2 = scores1 / norm1.clamp(min=eps) * cfg.SMOOTH.GAMMA3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze()
        scores3 = scores0.transpose(0, 1)

        scores4 = scores2.squeeze()
        scores5 = scores2.transpose(0, 1)
        if labels is not None:
            # labels是噪声
            loss0 = nn.CrossEntropyLoss()(scores0, labels)
            loss1 = nn.CrossEntropyLoss()(scores3, labels)
            loss2 = nn.CrossEntropyLoss()(scores4, labels)
            loss3 = nn.CrossEntropyLoss()(scores5, labels)

        else:
            loss0, loss1, loss2, loss3 = None, None, None, None
        return loss0, loss1, m_loss, loss2, loss3

    @staticmethod
    def local_loss_visual(img_features, words_emb, labels, sentence_mask, batch_size, num_blocks):
        """
            words_emb(query): batch x seq_len x nef
            img_features(context): batch x nef x 17 x 17
            sentence_mask: batch x seq_len
        """
        # seq_len = words_emb.size(1)
        words_emb = words_emb.permute(0, 2, 1)  # batch, nef, seq_len
        # att_maps = []
        similarities = []
        # chunk attention
        batch_size, image_dim, image_height, image_width = image_features.size()
        batch_size, text_length, text_dim = words_emb.size()
        img_features = image_features.view(batch_size, image_dim, self.num_blocks, -1) 
        words_emb = text_features.view(batch_size, text_dim, self.num_blocks, -1)
        
        for i in range(batch_size):
            # Get the i-th text description
            words_num = int(torch.sum(sentence_mask[i]))
            # -> 1 x nef x words_num

            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
            # -> batch_size x nef x words_num
            word = word.repeat(batch_size, 1, 1)

            # batch_size x nef x 17 x 17
            context = img_features
            weiContext, attn = func_attention(word, context, cfg.SMOOTH.GAMMA1)

            # --> batch_size x words_num x nef
            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            # --> batch_size*words_num x nef
            word = word.view(batch_size * words_num, -1)
            weiContext = weiContext.view(batch_size * words_num, -1)

            # -->batch_size*words_num
            row_sim = cosine_similarity(word, weiContext)
            # --> batch_size x words_num
            row_sim = row_sim.view(batch_size, words_num)

            row_sim.mul_(cfg.SMOOTH.GAMMA2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            # --> 1 x batch_size
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(row_sim)

        # batch_size x batch_size
        similarities = torch.cat(similarities, 1)
        similarities = similarities * cfg.SMOOTH.GAMMA3
        similarities1 = similarities.transpose(0, 1)
        if labels is not None:
             loss0 = nn.CrossEntropyLoss()(similarities, labels)
             loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        else:
             loss0, loss1 = None, None

        return loss0 + loss1

    @staticmethod
    def local_loss_textual(img_features, words_emb, labels, sentence_mask, batch_size, num_blocks):
        """
            words_emb(context): batch x seq_len x nef
            img_features(query): batch x nef x 17 x 17
            sentence_mask: batch x seq_len
        """

        img_patch_num = img_features.size(2) * img_features.size(3)
        img_features = img_features.reshape(batch_size, -1, img_patch_num)  # batch, nef, 289
        # img_features = img_features.permute(0, 2, 1)  # batch, 289, nef

        words_emb = words_emb.permute(0, 2, 1)  # batch, nef, seq_len

        similarities = []
        batch_size, image_dim, image_height, image_width = image_features.size()
        batch_size, text_length, text_dim = words_emb.size()
        img_features = image_features.view(batch_size, image_dim, self.num_blocks, -1) 
        words_emb = text_features.view(batch_size, text_dim, self.num_blocks, -1)
        for i in range(batch_size):
            # Get the i-th img description
            img = img_features[i, :, :].unsqueeze(0).contiguous()
            # -> batch_size x nef x 289
            img = img.repeat(batch_size, 1, 1)

            # batch_size x nef x seq_len
            context = words_emb
            weiContext, attn = func_attention_textual(img, context, sentence_mask, cfg.SMOOTH.GAMMA1)

            # --> batch_size x 289 x nef
            img = img.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            # --> batch_size*words_num x nef
            img = img.view(batch_size * img_patch_num, -1)
            weiContext = weiContext.view(batch_size * img_patch_num, -1)

            # -->batch_size*289
            row_sim = cosine_similarity(img, weiContext)
            # --> batch_size x 289
            row_sim = row_sim.view(batch_size, img_patch_num)

            row_sim.mul_(cfg.SMOOTH.GAMMA2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            # --> 1 x batch_size
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(row_sim)

        # batch_size x batch_size
        similarities = torch.cat(similarities, 1)
        similarities = similarities * cfg.SMOOTH.GAMMA3
        similarities1 = similarities.transpose(0, 1)
        if labels is not None:
            loss0 = nn.CrossEntropyLoss()(similarities, labels)
            loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        else:
            loss0, loss1 = None, None

        return loss0 + loss1



    def forward(self, cnn_code, rnn_code, img_features, words_emb, cnn_code_aug, rnn_code_aug, queue, queue_im, sentence_mask, labels):
        batch_size = words_emb.size(0)
        w_loss0 = self.local_loss_visual(img_features, words_emb, labels, sentence_mask=sentence_mask, batch_size=batch_size)
        w_loss1 = self.local_loss_textual(img_features, words_emb, labels, sentence_mask=sentence_mask, batch_size=batch_size)
        s_loss0, s_loss1, m_loss, s_loss6, s_loss7 = self.global_loss(cnn_code, cnn_code_aug, rnn_code, rnn_code_aug, labels, queue, queue_im)
        return w_loss0.mean(), w_loss1.mean(), s_loss0, s_loss1, m_loss, s_loss6, s_loss7
