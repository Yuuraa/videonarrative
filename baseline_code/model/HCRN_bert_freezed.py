import numpy as np
from torch.nn import functional as F
import torch.nn as nn

from .utils import *
from .CRN import CRN

from transformers import BertModel


class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill


class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=768, hidden_dim=768, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()
        
        self.dim = module_dim

        self.tanh = nn.Tanh()
        self.encoder = BertModel.from_pretrained("kykim/bert-kor-base")
        # TODO: refactor
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim
        self.proj = nn.Linear(hidden_dim, module_dim)
        self.init_layers()

    def init_layers(self):
        with torch.no_grad():
          for m in self.modules():
            if isinstance(m, nn.Linear):
              nn.init.xavier_uniform_(m.weight)
              m.bias.fill_(0.0)

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        # questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        # embed = self.tanh(self.embedding_dropout(questions_embedding))

        question_embedding = self.encoder(questions)['last_hidden_state'][:, 0, :] # [CLS] 토큰의 hidden state를 취함
        question_embedding = self.question_dropout(question_embedding)
        question_embedding = self.proj(question_embedding)

        return question_embedding


class InputUnitVisual(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(InputUnitVisual, self).__init__()

        self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)

        # TODO: change to conv1d or pretrained ViT
        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        for i in range(appearance_video_feat.size(1)):
            clip_level_motion = motion_video_feat[:, i, :]  # (bz, 2048)
            clip_level_motion_proj = self.clip_level_motion_proj(clip_level_motion)

            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance)  # (bz, 16, 512)
            # clip level CRNs
            clip_level_crn_motion = self.clip_level_motion_cond(torch.unbind(clip_level_appearance_proj, dim=1),
                                                                clip_level_motion_proj)
            clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_motion, question_embedding_proj)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_motion,
                                                                  question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output


class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitCount(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitCount, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.regression = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.regression(out)

        return out


class HCRNFreezedBertNetwork(nn.Module):
    def __init__(self, vision_dim, module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, question_type):
        super(HCRNFreezedBertNetwork, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        # if self.question_type in ['action', 'transition','none']:
        if self.question_type in ['none']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        else:
            raise ValueError(f"HCRNBertNetwork takes no such question type {self.question_type}")

        init_modules(self.modules(), w_init="xavier_uniform")

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,
                question_len):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        if self.question_type in ['frameqa', 'count']: #, 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question, question_len)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            out = self.output_unit(question_embedding, visual_embedding)
        else:
            question_embedding = self.linguistic_input_unit(question, question_len)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
        return out
