import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.2):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head*d_v, bias=False)
        self.fc = nn.Linear(n_head*d_v, d_model)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        residual = q
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)
        qs = self.w_qs(q).view(b, q_len, n_head, d_k)
        ks = self.w_ks(k).view(b, k_len, n_head, d_k)
        vs = self.w_vs(v).view(b, v_len, n_head, d_k)

        qs, ks, vs = qs.transpose(1,2), ks.transpose(1,2), vs.transpose(1,2)
        enc_q = self.attention(qs, ks, vs)
        enc_q = self.dropout(self.fc(enc_q.view(b,q_len,n_head*d_k)))
        enc_q += residual
        enc_q = self.layer_norm(enc_q)
        return enc_q

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.2)

    def forward(self, q, k, v):
        att = torch.matmul(q, k.transpose(-1,-2))/self.temperature
        att = self.dropout(torch.softmax(att, dim=-1))
        enc = torch.matmul(att, v)
        return enc

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_inner)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_inner, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        residule = x
        enc = self.relu(self.linear1(x))
        enc = self.dropout(self.linear2(enc))
        enc += residule
        enc= self.layernorm(enc)
        return enc


class TransEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.2):
        super(TransEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner)

    def forward(self, enc_input):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2*(hid_j//2)/d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])
        sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransEncoder(nn.Module):
    def __init__(self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, n_position=20):
        super().__init__()
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(0.2)
        self.layer_stack = nn.ModuleList([
            TransEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=0.2)
         for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq):
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)
        return enc_output

class AdditiveAttention1(nn.Module):
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim):
        super(AdditiveAttention1, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector):
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
            dim=1)

        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target

class AdditiveAttention2(nn.Module):
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim,
                 ):
        super(AdditiveAttention2, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector, num_embedding):
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(torch.cat([candidate_vector, num_embedding],dim=-1)))
        # temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
            dim=1)

        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target


class AdditiveAttention2pre(nn.Module):
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim,
                 ):
        super(AdditiveAttention2pre, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.user = nn.Linear(50, 100)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector, user):
        # batch_size, candidate_size, query_vector_dim

        user= self.user(user).repeat(1,candidate_vector.size(1),1)

        temp = torch.tanh(self.linear(torch.cat([candidate_vector, user],dim=-1)))
        # temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
            dim=1)

        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target



class MultiHeadAttention2(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, d_out, dropout=0.2):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head*d_v, bias=False)
        self.fc = nn.Linear(n_head*d_v, d_out)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, q, k, v):
        residual = q
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)
        qs = self.w_qs(q).view(b, q_len, n_head, d_k)
        ks = self.w_ks(k).view(b, k_len, n_head, d_k)
        vs = self.w_vs(v).view(b, v_len, n_head, d_k)

        qs, ks, vs = qs.transpose(1,2), ks.transpose(1,2), vs.transpose(1,2)
        enc_q = self.attention(qs, ks, vs)
        enc_q = self.dropout(self.fc(enc_q.view(b,q_len,n_head*d_k)))
        enc_q += residual
        enc_q = self.layer_norm(enc_q)
        return enc_q

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.2)

    def forward(self, q, k, v):
        att = torch.matmul(q, k.transpose(-1,-2))/self.temperature
        att = self.dropout(torch.softmax(att, dim=-1))
        enc = torch.matmul(att, v)
        return enc

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_inner)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_inner, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        residule = x
        enc = self.relu(self.linear1(x))
        enc = self.dropout(self.linear2(enc))
        enc += residule
        enc= self.layernorm(enc)
        return enc


class TransEncoderLayer2(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, d_out, dropout=0.2):
        super(TransEncoderLayer2, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, d_out, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner)

    def forward(self, enc_input):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output





class TransEncoder2(nn.Module):
    def __init__(self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, d_out, n_position=20):
        super().__init__()
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(0.2)
        self.layer_stack = nn.ModuleList([
            TransEncoderLayer2(d_model, d_inner, n_head, d_k, d_v, d_out, dropout=0.2)
         for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq):
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)
        return enc_output

class AdditiveAttention2pre2(nn.Module):
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim,
                 ):
        super(AdditiveAttention2pre2, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.user = nn.Linear(50, 200)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector, user):
        # batch_size, candidate_size, query_vector_dim

        #user= self.user(user).repeat(1,candidate_vector.size(1),1)
        #print(user.size(), candidate_vector.size())
        temp = torch.tanh(self.user(user).repeat(1,candidate_vector.size(1),1))
        # temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
            dim=1)
        #print(candidate_weights.size())
       # print(candidate_vector.size())
	#print(candidate_vector.size())
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target