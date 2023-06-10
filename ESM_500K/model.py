from config import Config

import torch.nn as nn

import newsEncoders
import userEncoders



class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        # For main experiments of news encoding
        self.config = config

        if config.news_encoder == 'Event_NAML_dsc2':
            self.news_encoder = newsEncoders.Event_NAML_dsc2(config)

        else:
            raise Exception(config.news_encoder + 'is not implemented')

        # For main experiments of user encoding

        if config.user_encoder == 'Event_nrms_per_his':
            self.user_encoder = userEncoders.Event_nrms_per_his(self.news_encoder, config)
        else:
            raise Exception(config.user_encoder + 'is not implemented')

        self.model_name = config.news_encoder + '-' + config.user_encoder
        self.news_embedding_dim = self.news_encoder.news_embedding_dim
        self.dropout = nn.Dropout(p=config.dropout_rate)

        if config.user_encoder in ['Event_nrms_per_his']:
            print(config.user_num)
            self.user_embedding = nn.Embedding(num_embeddings=config.user_num, embedding_dim=config.user_embedding_dim)
            self.use_user_embedding = True
        else:
            self.use_user_embedding = False

        self.click_predictor = config.click_predictor


    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])



    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity,
                user_event_type=None, news_event_type=None):
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None                                                                                                         # [batch_size, news_embedding_dim]
        
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding) # [batch_size, 1 + negative_sample_num, news_embedding_dim]

        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, user_history_mask,\
                                                user_embedding, news_representation, user_event_type, news_event_type)                           # [batch_size, 1 + negative_sample_num, news_embedding_dim]



        if self.click_predictor == 'event':
            logits = user_representation
        return logits