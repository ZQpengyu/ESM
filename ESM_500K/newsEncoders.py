import pickle
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Conv1D, Attention


class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])

    # Input
    # title_text          : [batch_size, news_num, max_title_length]
    # title_mask          : [batch_size, news_num, max_title_length]
    # title_entity        : [batch_size, news_num, max_title_length]
    # content_text        : [batch_size, news_num, max_content_length]
    # content_mask        : [batch_size, news_num, max_content_length]
    # content_entity      : [batch_size, news_num, max_content_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # user_embedding      : [batch_size, user_embedding_dim]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory):
        category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)                                                                           # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation)], dim=2) # [batch_size, news_num, news_embedding_dim]
        return news_representation

class Event_Extract_dec_naml2(nn.Module):
    def __init__(self, config):
        super(Event_Extract_dec_naml2, self).__init__()
        self.e1 = nn.Linear(config.cnn_kernel_num, config.cnn_kernel_num)
        self.e2 = nn.Linear(config.cnn_kernel_num*2, config.cnn_kernel_num)
        self.e3 = nn.Linear(config.cnn_kernel_num*2, config.cnn_kernel_num)

        self.ed1 = nn.Linear(config.cnn_kernel_num, config.cnn_kernel_num)
        self.ed2 = nn.Linear(config.cnn_kernel_num*2, config.cnn_kernel_num)
        self.ed3 = nn.Linear(config.cnn_kernel_num*2, config.cnn_kernel_num)

        self.sd1 = nn.Linear(config.cnn_kernel_num, config.cnn_kernel_num)
        self.sd2 = nn.Linear(config.cnn_kernel_num*2, config.cnn_kernel_num)
        self.sd3 = nn.Linear(config.cnn_kernel_num*2, config.cnn_kernel_num)


        self.event_classifer = nn.Linear(config.cnn_kernel_num, config.K)
        self.concat_classifer = nn.Linear(config.cnn_kernel_num*2, config.cnn_kernel_num)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, news_rep):
        # news_rep batch, hidden
        ehidden1 = self.dropout(F.relu(self.e1(news_rep)))
        ehidden2 = self.dropout(F.relu(self.e2(torch.cat([news_rep,ehidden1],dim=-1))))
        ehidden3 = F.relu(self.e3(torch.cat([news_rep,ehidden2],dim=-1)))

        edhidden1 = self.dropout(F.relu(self.ed1(ehidden3)))
        edhidden2 = self.dropout(F.relu(self.ed2(torch.cat([ehidden3,edhidden1],dim=-1))))
        event_decoder = F.relu(self.ed3(torch.cat([ehidden3,edhidden2],dim=-1)))

        sdhidden1 = self.dropout(F.relu(self.sd1(ehidden3)))
        sdhidden2 = self.dropout(F.relu(self.sd2(torch.cat([ehidden3,sdhidden1],dim=-1))))
        style_decoder = F.relu(self.sd3(torch.cat([ehidden3,sdhidden2],dim=-1)))

        channel_softmax = self.softmax(self.event_classifer(event_decoder))
        style_softmax = self.softmax(self.event_classifer(style_decoder))
        all_feature = self.concat_classifer(torch.cat([event_decoder,style_decoder], dim=-1))
        loss2 = F.mse_loss(all_feature,news_rep)

        loss_dec = loss2

        channel = channel_softmax.unsqueeze(-1)
        event_decoder = event_decoder.unsqueeze(1)
        event_rep = torch.mul(channel, event_decoder)
        return event_decoder, event_rep, channel_softmax, style_decoder, loss_dec, style_softmax


class Event_NAML_dsc2(NewsEncoder):
    def __init__(self, config: Config):
        super(Event_NAML_dsc2, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(in_features=config.category_embedding_dim, out_features=config.cnn_kernel_num, bias=True)
        self.subCategory_affine = nn.Linear(in_features=config.subCategory_embedding_dim, out_features=config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(in_features=config.cnn_kernel_num, out_features=config.attention_dim, bias=True)
        self.affine2 = nn.Linear(in_features=config.attention_dim, out_features=1, bias=False)
        self.event = Event_Extract_dec_naml2(config)
        self.config = config

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim]) # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                                                  # [batch_size * news_num, max_title_length, cnn_kernel_num]
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                            # [batch_size * news_num, max_content_length, cnn_kernel_num]
        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, news_num, self.cnn_kernel_num])                               # [batch_size, news_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view([batch_size, news_num, self.cnn_kernel_num])                         # [batch_size, news_num, cnn_kernel_num]
        # 4. category and subCategory encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                              # [batch_size, news_num, cnn_kernel_num]
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)                  # [batch_size, news_num, cnn_kernel_num]
        # 5. multi-view attention
        feature = torch.stack([title_representation, content_representation, category_representation, subCategory_representation], dim=2)    # [batch_size, news_num, 4, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                            # [batch_size, news_num, 4, 1]
        news_representation = (feature * alpha).sum(dim=2, keepdim=False).view([batch_size*news_num, self.cnn_kernel_num])                                                               # [batch_size, news_num, cnn_kernel_num]
        event_decoder, event_rep, channel_softmax, style_decoder, loss_dec, style_softmax = self.event(news_representation)
        event_decoder = event_decoder.view(batch_size, news_num, -1)
        event_rep = event_rep.view(batch_size, news_num, self.config.K, -1)
        channel_softmax = channel_softmax.view(batch_size, news_num, -1)
        style_decoder = style_decoder.view(batch_size, news_num, -1)
        style_softmax = style_softmax.view(batch_size, news_num, -1)
        # return event_decoder
        output = [event_decoder, event_rep, channel_softmax, style_decoder, loss_dec, style_softmax] #event_decoder, event_rep, channel_softmax, style_decoder, loss_dec
        return output




