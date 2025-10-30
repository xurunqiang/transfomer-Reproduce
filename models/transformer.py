import math
import copy
import re
from typing import Optional, List, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# 1. 位置编码模块
# ------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """正弦余弦位置编码，与《Attention is All You Need》原文一致"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数列用正弦，奇数列用余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # 当 d_model 为奇数时，div_term 长度比需要的偶数列多 1
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 不参与训练的缓冲区

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            加入位置编码后的张量 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


# ------------------------------------------------------------------
# 2. 缩放点积注意力
# ------------------------------------------------------------------
def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    单头或多头后的缩放点积注意力（query/key/value 均为 [batch, heads, seq_len, head_dim]）
    mask: bool Tensor，可广播到 scores 的形状；True 表示允许 attend，False 表示屏蔽。
    Returns:
        output: 注意力加权结果 [batch, heads, seq_q, head_dim]
        attn_weights: 注意力权重 [batch, heads, seq_q, seq_k]
    """
    d_k = query.size(-1)
    # 计算注意力分数: (batch, heads, seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask 应该是 bool，True 表示允许 attend；若为浮点，转换为 bool
        if mask.dtype != torch.bool:
            mask = mask.bool()
        # 兼容 broadcast：mask 可能是 [batch, 1, 1, seq_k] 或 [batch, 1, seq_q, seq_k] 等
        # 这里利用 broadcasting：直接将 ~mask 的位置置为 -inf
        scores = scores.masked_fill(~mask, float('-inf'))

    # 数值稳定性：对于整行被 mask 的情况，softmax 会产生 NaN，所以先做一个 clamp
    attn_weights = F.softmax(scores, dim=-1)
    # 对非常小数值进行数值稳定处理（可选）
    attn_weights = torch.where(torch.isfinite(attn_weights), attn_weights, torch.zeros_like(attn_weights))

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


# ------------------------------------------------------------------
# 3. 多头注意力
# ------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的维度

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query/key/value: [batch_size, seq_len, d_model]
            mask: 注意力掩码，bool 或可广播的浮点，常见形状：
                  - src_mask: [batch, 1, 1, seq_k] (padding mask)
                  - tgt_mask: [batch, 1, seq_q, seq_k] (padding + look-ahead)
        Returns:
            output: 多头注意力结果 [batch_size, seq_len_q, d_model]
            attn_weights: 注意力权重 [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # 线性投影并分拆多头
        def split_heads(x: torch.Tensor) -> torch.Tensor:
            """将张量分拆为多个头: [batch, seq_len, d_model] -> [batch, heads, seq_len, head_dim]"""
            x = x.view(batch_size, -1, self.num_heads, self.head_dim)
            return x.transpose(1, 2)  # -> [batch, heads, seq_len, head_dim]

        q = split_heads(self.w_q(query))
        k = split_heads(self.w_k(key))
        v = split_heads(self.w_v(value))

        # 准备 mask：扩展至 [batch, num_heads, seq_q, seq_k]
        attn_mask = None
        if mask is not None:
            # 如果 mask 不是 bool，则尝试转为 bool（非零为 True）
            if mask.dtype != torch.bool:
                mask = mask.bool()

            # mask 形状可能为：
            # - [batch, 1, 1, seq_k]  -> padding mask for key (broadcast over heads and seq_q)
            # - [batch, 1, seq_q, seq_k] -> target mask (already includes seq_q)
            # - [batch, seq_q, seq_k] -> maybe missing head dim
            if mask.dim() == 4 and mask.size(1) == 1:
                # [batch, 1, seq_q, seq_k] 或 [batch, 1, 1, seq_k]
                # 直接 expand 第1维到 num_heads
                attn_mask = mask.expand(-1, self.num_heads, mask.size(2), mask.size(3))
            elif mask.dim() == 4 and mask.size(1) == self.num_heads:
                attn_mask = mask  # 已经有 head 维
            elif mask.dim() == 3:
                # [batch, seq_q, seq_k] -> add head dim
                attn_mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            else:
                # 其他兼容形状，尝试广播
                attn_mask = mask.unsqueeze(1).expand(-1, self.num_heads, q.size(2), k.size(2))

        # 计算注意力
        output, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask, self.dropout
        )

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 输出投影
        return self.w_o(output), attn_weights


# ------------------------------------------------------------------
# 4. 位置-wise前馈网络
# ------------------------------------------------------------------
class PositionWiseFeedForward(nn.Module):
    """两层全连接网络，带ReLU激活和dropout"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


# ------------------------------------------------------------------
# 5. 编码器层
# ------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """Transformer编码器层: 自注意力 + 前馈网络 + 残差连接和层归一化"""

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 自注意力掩码
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


# ------------------------------------------------------------------
# 6. 解码器层
# ------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """Transformer解码器层: 掩码自注意力 + 编码器-解码器注意力 + 前馈网络"""

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            self_mask: Optional[torch.Tensor] = None,
            cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            memory: 编码器输出 [batch_size, src_seq_len, d_model]
            self_mask: 解码器自注意力掩码
            cross_mask: 编码器-解码器注意力掩码
        Returns:
            [batch_size, tgt_seq_len, d_model]
        """
        # 掩码自注意力子层
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 编码器-解码器注意力子层
        cross_output, _ = self.cross_attn(x, memory, memory, cross_mask)
        x = self.norm2(x + self.dropout2(cross_output))

        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


# ------------------------------------------------------------------
# 7. 编码器和解码器
# ------------------------------------------------------------------
class Encoder(nn.Module):
    """由多个编码器层堆叠而成的完整编码器"""

    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        # layer.self_attn.d_model 存在于 MultiHeadAttention 中
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 注意力掩码
        Returns:
            编码结果 [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """由多个解码器层堆叠而成的完整解码器"""

    def __init__(self, layer: DecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            self_mask: Optional[torch.Tensor] = None,
            cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            memory: 编码器输出 [batch_size, src_seq_len, d_model]
            self_mask: 解码器自注意力掩码
            cross_mask: 编码器-解码器注意力掩码
        Returns:
            解码结果 [batch_size, tgt_seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        return self.norm(x)


# ------------------------------------------------------------------
# 8. 完整Transformer模型
# ------------------------------------------------------------------
class Transformer(nn.Module):
    """完整的Transformer模型，包含编码器、解码器和嵌入层"""

    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            d_ff: int = 2048,
            dropout: float = 0.1,
            max_len: int = 5000,
            pad_idx: int = 0
    ):
        super().__init__()

        # 源序列嵌入和位置编码
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 目标序列嵌入和位置编码
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 编码器和解码器
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, num_layers)
        self.decoder = Decoder(decoder_layer, num_layers)

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """Xavier初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码源序列
        Args:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码
        Returns:
            编码结果 [batch_size, src_seq_len, d_model]
        """
        x = self.src_embedding(src)
        x = self.src_pos_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解码目标序列
        Args:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            memory: 编码器输出 [batch_size, src_seq_len, d_model]
            tgt_mask: 目标序列掩码
            memory_mask: 编码器输出掩码
        Returns:
            解码结果 [batch_size, tgt_seq_len, d_model]
        """
        x = self.tgt_embedding(tgt)
        x = self.tgt_pos_encoding(x)
        return self.decoder(x, memory, tgt_mask, memory_mask)

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        完整的前向传播
        Args:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            memory_mask: 编码器输出掩码
        Returns:
            输出logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, memory_mask)
        return self.output_layer(output)


# ------------------------------------------------------------------
# 9. 掩码工具函数（改为 bool mask，更稳健）
# ------------------------------------------------------------------
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    创建填充掩码，屏蔽PAD位置
    Args:
        seq: 输入序列 [batch_size, seq_len]
        pad_idx: PAD符号的索引
    Returns:
        掩码 [batch_size, 1, 1, seq_len] (bool)，True 表示可 attend
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch,1,1,seq_len]
    return mask  # dtype bool


def create_look_ahead_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    创建前瞻掩码，防止解码器看到未来信息
    Args:
        seq_len: 序列长度
        device: 设备
    Returns:
        掩码 [1, seq_len, seq_len] (bool)，上三角 (i<j) 为 False（不能看见）
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()  # True 表示被遮挡
    # 返回 True 表示可以 attend，因此取反
    return ~mask.unsqueeze(0)  # [1, seq_len, seq_len]


def create_target_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    创建目标序列掩码，结合填充掩码和前瞻掩码
    Args:
        tgt: 目标序列 [batch_size, tgt_seq_len]
        pad_idx: PAD符号的索引
    Returns:
        掩码 [batch_size, 1, tgt_seq_len, tgt_seq_len] (bool)
    """
    batch_size, tgt_len = tgt.size(0), tgt.size(1)
    pad_mask = create_padding_mask(tgt, pad_idx)  # [batch,1,1,tgt_len]
    look_ahead_mask = create_look_ahead_mask(tgt_len, tgt.device)  # [1, tgt_len, tgt_len]
    # 将 pad_mask 扩展到 [batch,1,tgt_len,tgt_len] 再与 look_ahead_mask 相与
    pad_mask_expanded = pad_mask.expand(-1, -1, tgt_len, -1)  # [batch,1,tgt_len,tgt_len]
    look_ahead_mask = look_ahead_mask.unsqueeze(1)  # [1,1,tgt_len,tgt_len]
    target_mask = pad_mask_expanded & look_ahead_mask  # broadcast to [batch,1,tgt_len,tgt_len]
    return target_mask


# ------------------------------------------------------------------
# 10. 数据预处理与工具（修正 build_vocab 等）
# ------------------------------------------------------------------
def preprocess_and_tokenize_english(sentences: List[str]) -> List[List[str]]:
    """预处理英文句子：小写、去除标点、分词（用正则更健壮）"""
    tokenized_sentences = []
    for sent in sentences:
        cleaned = sent.lower()
        # 保留空格和单词字符，替换其它为单空格
        cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
        tokens = cleaned.split()
        tokenized_sentences.append(tokens)
    return tokenized_sentences


def build_vocab(tokenized_data: List[List[str]], min_freq: int = 1) -> Tuple[dict, dict]:
    """构建词→索引、索引→词的映射（修复 Counter 的 bug）"""
    # 正确统计每个 token
    word_counter = Counter([token for sent in tokenized_data for token in sent])
    # 加入特殊token（PAD:填充, UNK:未登录词, BOS:开始, EOS:结束）
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3
    }
    # 按出现添加
    for word, freq in word_counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    inv_vocab = {idx: word for word, idx in vocab.items()}
    return vocab, inv_vocab


def encode_tokens(tokenized_data: List[List[str]], vocab: dict, add_eos: bool = True) -> List[torch.Tensor]:
    """将句子token转换为整数索引序列，可选添加EOS标记"""
    encoded_sentences = []
    for tokens in tokenized_data:
        encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        if add_eos:
            encoded.append(vocab["<EOS>"])
        encoded_sentences.append(torch.tensor(encoded, dtype=torch.long))
    return encoded_sentences


def pad_sequences(encoded_data: List[torch.Tensor], pad_idx: int = 0) -> torch.Tensor:
    """填充序列到最大长度，返回(batch_size, max_seq_len)"""
    max_seq_len = max([len(seq) for seq in encoded_data])
    padded_batch = torch.full((len(encoded_data), max_seq_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(encoded_data):
        padded_batch[i, :len(seq)] = seq
    return padded_batch



