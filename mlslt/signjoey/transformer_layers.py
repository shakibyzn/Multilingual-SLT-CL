# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor
from signjoey.vocabulary import LangVocabulary
from signjoey.adapter import MADX
from typing import List

# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.pwff_layer(x)


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        lang_num: int = 10,
        use_adapter: bool = False,
        sign_lang_vocab: LangVocabulary = None,
        languages: List[str] = None,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        :param lang_num:
        :param use_adapter:
        :param sign_lang_vocab:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(size, eps=1e-6)

        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.att_routing = RoutingLayer(size, lang_num=lang_num)
        self.ff_routing = RoutingLayer(size, lang_num)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.size = size
        self.lang_num = lang_num
        if use_adapter:
            print("Sign Language Vocab: ", sign_lang_vocab)
            self.sign_lang_vocab = {k: v for k, v in sign_lang_vocab.stoi.items()}
            self.sign_lang_vocab_rev = {v: k for k, v in self.sign_lang_vocab.items()}
            self.adapter_modules = nn.ModuleDict({})
            for sign_lang in languages: 
                self.adapter_modules[sign_lang] = MADX(input_size=size)
        
            print("Language Adapters: ", self.adapter_modules)
        
        else:
            self.adapter_modules = None

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor, lang: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x = self.layer_norm(x)
        h = self.src_src_att(x, x, x, mask)
        h = self.att_routing(old_x=x, x=h, lang=lang)
        
        # Apply MADX adapters if available
        if self.adapter_modules is not None:
            lang_ids, lang_lengths = torch.unique_consecutive(lang, return_counts=True)
            split_hidden_states = torch.split(h, lang_lengths.tolist(), 0)
            lang_wise_outputs = []
            
            for lang_id, split_hidden_state in zip(lang_ids, split_hidden_states):
                lang_name = self.sign_lang_vocab_rev[lang_id.item()]
                if lang_name in self.adapter_modules:
                    # Apply adapter if it exists for the language
                    lang_wise_outputs.append(self.adapter_modules[lang_name](split_hidden_state))
                else:
                    # If no adapter exists, use the hidden state directly
                    lang_wise_outputs.append(split_hidden_state)
                    
            h = torch.cat(lang_wise_outputs, 0)
            
        x = self.dropout(h) + x
        x = self.layer_norm2(x)
        h = self.feed_forward(x)
        h = self.ff_routing(old_x=x, x=h, lang=lang)
        x = self.dropout2(h) + x
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        lang_num: int = 10,
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.ff_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.att_routing = RoutingLayer(size, lang_num)
        self.cross_att_routing = RoutingLayer(size, lang_num)
        self.ff_routing = RoutingLayer(size, lang_num)

        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.lang_num = lang_num

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
        lang: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x = self.x_layer_norm(x)
        h = self.trg_trg_att(x, x, x, mask=trg_mask)
        h = self.att_routing(old_x=x, x=h, lang=lang)
        x = self.dropout(h) + x

        # source-target attention
        x = self.dec_layer_norm(x)
        h = self.src_trg_att(memory, memory, x, mask=src_mask)
        h = self.cross_att_routing(old_x=x, x=h, lang=lang)
        x = self.dropout(h) + x
        # final position-wise feed-forward layer
        x = self.ff_layer_norm(x)
        h = self.feed_forward(x)
        h = self.ff_routing(old_x=x, x=h, lang=lang)
        x = self.dropout2(h) + x
        return x


class GateLayer(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(size, size)
        self.layer2 = nn.Linear(size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.layer1(x)) + x
        x = self.layer2(x)
        return x


class RoutingLayer(nn.Module):
    def __init__(self, size: int, lang_num: int) -> None:
        super().__init__()
        self.share_w = nn.Linear(size, size, bias=False)
        self.langs_w = nn.ModuleList(
            [nn.Linear(size, size, bias=False) for _ in range(lang_num)]
        )
        self.langs_gate = nn.ModuleList([GateLayer(size) for _ in range(lang_num)])

    def forward(self, old_x: Tensor, x: Tensor, lang: Tensor) -> Tensor:
        # batch_size,sentence_len,feature_size
        batch_size, sentence_len, feature_size = x.shape
        share_h = self.share_w(x)

        gate_prob = []
        langs_h = []
        for lang_index, data, old_data in zip(lang, x, old_x):
            lang_index = int(lang_index.item())
            langs_h.append(self.langs_w[lang_index](data))
            gate_prob.append(torch.sigmoid(self.langs_gate[lang_index](old_data)))
        langs_h = torch.stack(langs_h)
        gate_prob = torch.stack(gate_prob).reshape(batch_size, sentence_len, 1)

        result = share_h * (1 - gate_prob) + langs_h * gate_prob
        return result
