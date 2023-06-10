import math
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MultiHeadAttention,Attention, CandidateAttention
from newsEncoders import NewsEncoder
from util import try_to_install_torch_scatter_package
try_to_install_torch_scatter_package()
from sublayers import AdditiveAttention1, AdditiveAttention2


print('ok')


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(UserEncoder, self).__init__()
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.news_encoder = news_encoder
        self.device = torch.device('cuda')
        self.auxiliary_loss = None

    # Input
    # user_title_text               : [batch_size, max_history_num, max_title_length]
    # user_title_mask               : [batch_size, max_history_num, max_title_length]
    # user_title_entity             : [batch_size, max_history_num, max_title_length]
    # user_content_text             : [batch_size, max_history_num, max_content_length]
    # user_content_mask             : [batch_size, max_history_num, max_content_length]
    # user_content_entity           : [batch_size, max_history_num, max_content_length]
    # user_category                 : [batch_size, max_history_num]
    # user_subCategory              : [batch_size, max_history_num]
    # user_history_mask             : [batch_size, max_history_num]
    # user_history_graph            : [batch_size, max_history_num, max_history_num]
    # user_history_category_mask    : [batch_size, category_num]
    # user_history_category_indices : [batch_size, max_history_num]
    # user_embedding                : [batch_size, user_embedding]
    # candidate_news_representation : [batch_size, news_num, news_embedding_dim]
    # Output
    # user_representation           : [batch_size, news_embedding_dim]
    # def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
    #             user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation,user_event_type, news_event_type):
    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask,
                user_content_entity, user_category, user_subCategory, user_history_mask,\
                user_embedding, candidate_news_representation, user_event_type, news_event_type):
        raise Exception('Function forward must be implemented at sub-class')






class UserManager_NAML(nn.Module):
    def __init__(self, args,news_embedding_dim):
        super().__init__()
        self.args = args
        for k in range(args.K):
            setattr(self,f'Cand_attention_{k}',AdditiveAttention1(200, news_embedding_dim))
        #self.pos = PositionalEncoding(args.word_embedding_dim)
        self.linear = nn.Linear(news_embedding_dim + 100, 200)
        self.attention_query_vector = nn.Parameter(
            torch.empty(200).uniform_(-0.1, 0.1))

    def forward(self, his, cdd, channel_weight):

        his_ur = his.unsqueeze(1).repeat(1, cdd.size(1), 1, 1, 1)
        cdd_ur = cdd.unsqueeze(2).repeat(1, 1, his.size(1), 1, 1)

        his_ur = his_ur.contiguous().view(his_ur.size(0) * his_ur.size(1), his_ur.size(2), his_ur.size(3), -1)
        cdd_ur = cdd_ur.contiguous().view(cdd_ur.size(0) * cdd_ur.size(1), cdd_ur.size(2), cdd_ur.size(3), -1)
        temp = []
        for k in range(self.args.K):

            t= getattr(self,f'Cand_attention_{k}')(his_ur[:,:,k,:])
            temp.append(t)
        u_channel = torch.stack(temp, dim=1)
        u_rep_k = u_channel.unsqueeze(-2)

        cdd_vector = cdd.contiguous().view(cdd.size(0) * cdd.size(1), cdd.size(2), cdd.size(3)).unsqueeze(-1)
        score = torch.matmul(u_rep_k, cdd_vector).squeeze(-1)

        u_rep_k = u_rep_k.squeeze(-2)
        channel_weight = torch.cat([u_rep_k, channel_weight], dim=-1)

        temp = torch.tanh(self.linear(channel_weight))

        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
            dim=1).unsqueeze(dim=1)





        scores = torch.matmul(candidate_weights,score).squeeze(-1) #batch*np+1,1

        return scores, u_channel



class Event_nrms_per_his(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(Event_nrms_per_his, self).__init__(news_encoder, config)

        self.user_ma = UserManager_NAML(config,self.news_embedding_dim)
        self.att_channel_TEST = AdditiveAttention1(200, self.news_embedding_dim)
        self.att_channel = AdditiveAttention2(200, self.news_embedding_dim + 100)
        self.num_embedding = nn.Embedding(51, 100)
        self.channel_softmax_embedding = nn.Embedding(11, 100)
        self.dense = nn.Linear(100, 100)
        self.config = config
        self.style_att = AdditiveAttention1(200, self.news_embedding_dim)
        self.user_dense = nn.Linear(in_features=config.user_embedding_dim, out_features=config.personalized_embedding_dim, bias=True)
        self.personalizedAttention = CandidateAttention(self.news_embedding_dim, config.personalized_embedding_dim, config.attention_dim)
        self.multiheadAttention = MultiHeadAttention(config.head_num, self.news_embedding_dim, config.max_history_num, config.max_history_num, config.head_dim, config.head_dim)
        self.affine = nn.Linear(in_features=config.head_num*config.head_dim, out_features=self.news_embedding_dim, bias=True)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)
        self.view_attention = nn.Linear(self.news_embedding_dim * 2,self.news_embedding_dim)

    def initialize(self):
        nn.init.uniform_(self.num_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.channel_softmax_embedding.weight, -0.1, 0.1)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask,
                user_content_entity, user_category, user_subCategory, user_history_mask, \
                user_embedding, candidate_news_representation, user_event_type, news_event_type):
        news_num = candidate_news_representation[0].size(1)
        batch_num = candidate_news_representation[0].size(0)
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory,
                                              user_embedding)



        cdd_encode = candidate_news_representation[0]
        cdd_event_rep = candidate_news_representation[1]
        cdd_softmax = candidate_news_representation[2]
        cdd_style_decoder = candidate_news_representation[3]
        cdd_loss = candidate_news_representation[4]
        cdd_style_softmax = candidate_news_representation[-1]


        his_encode = history_embedding[0]
        his_event_rep = history_embedding[1]
        his_softmax = history_embedding[2]
        his_style_decoder = history_embedding[3]
        his_loss = history_embedding[4]
        his_style_softmax = history_embedding[-1]

        his_t = his_softmax.transpose(-1, -2)  # batch, k, his
        a = torch.ones_like(his_t)
        b = torch.zeros_like(his_t)
        news_his_t = torch.where(his_t > (1 / self.config.K), a, b)
        news_his_t = news_his_t.sum(dim=-1).long()  # batch, k
        news_his_t = self.num_embedding(news_his_t)  # batch, k, embedding
        news_his_t = news_his_t.unsqueeze(1).repeat(1, cdd_event_rep.size(1), 1, 1)
        news_his_t = news_his_t.contiguous().view(news_his_t.size(0) * news_his_t.size(1), news_his_t.size(2), -1)

        channel_weight = cdd_softmax.contiguous().view(cdd_softmax.size(0) * cdd_softmax.size(1), -1)
        channel_weight = torch.round(channel_weight * 10).long()
        channel_weight = self.dense(self.channel_softmax_embedding(channel_weight))

        score1, u_channel = self.user_ma(his_event_rep, cdd_event_rep, channel_weight)
        cdd_news_rep1 = cdd_encode.contiguous().view(cdd_encode.size(0) * cdd_encode.size(1), -1).unsqueeze(
            1).repeat(1, u_channel.size(1), 1)
        u_channel_total = self.att_channel(u_channel,news_his_t)
        cdd_news_rep = cdd_encode.contiguous().view(cdd_encode.size(0) * cdd_encode.size(1), -1).unsqueeze(-1)
        score2 = torch.matmul(u_channel_total.unsqueeze(1), cdd_news_rep).squeeze(-1)

        # Event Matching

        score = float(self.config.score1)*score1 + float(self.config.score2)*score2
        #score = score2
        score = score.contiguous().view(batch_num, -1)

        h = self.multiheadAttention(his_style_decoder, his_style_decoder, his_style_decoder, user_history_mask)
        h = F.relu(F.dropout(self.affine(h), training=self.training, inplace=True), inplace=True)
        # his_style = self.attention(h)
        # u = self.user_dense(user_embedding)
        # print(his_style.size())
        # print(u.size())
        # his_style = self.view_attention(torch.cat([his_style,u], dim=-1)).unsqueeze(dim=1)
        # # style matching
        q_d = F.relu(self.user_dense(user_embedding), inplace=True)  # [batch_size, personalized_embedding_dim]
        his_style = self.personalizedAttention(h, q_d).unsqueeze(
            dim=1)
        #his_style = self.style_att(his_style_decoder).unsqueeze(1)  # batch,1,300  * cdd_style_decoder batch,5,300
        score3 = torch.matmul(his_style, cdd_style_decoder.transpose(-1, -2)).squeeze(1)


        score=score+float(self.config.score3)*score3



        if self.training:
            hidden_softmax = torch.cat([his_softmax, cdd_softmax], dim=1).view(-1,self.config.K)  # batch,np+1+his, k
            event_softmax = torch.cat([user_event_type, news_event_type], dim=1).view(-1,self.config.K)

            hidden_style_softmax = torch.cat([his_style_softmax, cdd_style_softmax], dim=1).view(-1,self.config.K)
            style_loss = F.mse_loss(hidden_style_softmax, event_softmax, reduce=False)
            style_loss = 1+style_loss
            style_loss = torch.mean(1/style_loss)
            event_loss = F.mse_loss(hidden_softmax, event_softmax)

            if style_loss.item()>1:
                print(style_loss)

            loss = style_loss + event_loss + his_loss + cdd_loss #(loss change)
            self.auxiliary_loss = loss

        return score

