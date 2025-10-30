from models.transformer import (preprocess_and_tokenize_english,
                                build_vocab, encode_tokens, pad_sequences, Transformer, create_padding_mask,
                                create_target_mask)
from typing import List
import torch
import torch.nn as nn
from tqdm import tqdm  # 导入tqmd进度条库


def read_chinese_sentences(file_path: str) -> List[str]:
    """
    读取文本文件中的中文句子，返回分词后的句子列表

    参数:
        file_path: 文本文件路径
    返回:
        包含分词句子的列表
    """
    chinese_sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取文件所有行，去除空行和首尾空白
            for line in file:
                stripped_line = line.strip()
                if stripped_line:  # 只处理非空行
                    chinese_sentences.append(stripped_line)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
    return chinese_sentences


def main():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 替换为你的文本文件路径
    Chinese_file_path = r"C:\Users\31392\Desktop\transfomer-Reproduce\data\chinese_2019_train_segmented.txt"
    English_file_path = r"C:\Users\31392\Desktop\transfomer-Reproduce\data\english_2019_train.txt"

    # ---------------------------
    # 1. 加载并预处理数据
    # ---------------------------
    print("加载并预处理数据...")
    chinese_sentences: List[str] = read_chinese_sentences(Chinese_file_path)
    english_sentences: List[str] = read_chinese_sentences(English_file_path)

    chinese_tokenized = [sent.split() for sent in chinese_sentences]
    english_tokenized = preprocess_and_tokenize_english(english_sentences)

    src_vocab, src_inv_vocab = build_vocab(chinese_tokenized)
    tgt_vocab, tgt_inv_vocab = build_vocab(english_tokenized)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    print(f"源语言词汇表大小: {src_vocab_size}")
    print(f"目标语言词汇表大小: {tgt_vocab_size}")

    # 编码序列
    src_encoded = encode_tokens(chinese_tokenized, src_vocab)
    tgt_encoded = encode_tokens(english_tokenized, tgt_vocab)

    # 填充序列
    pad_idx = 0
    src_padded = pad_sequences(src_encoded, pad_idx)
    tgt_padded = pad_sequences(tgt_encoded, pad_idx)

    print(f"源序列形状: {src_padded.shape}")
    print(f"目标序列形状: {tgt_padded.shape}")

    # ---------------------------
    # 2. 模型设置
    # ---------------------------
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    max_len = 512
    bos_idx = 2  # <BOS>
    eos_idx = 3  # <EOS>

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        pad_idx=pad_idx
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    src_data = src_padded.to(device)
    tgt_data = tgt_padded.to(device)

    # ---------------------------
    # 3. 训练循环（添加tqmd进度条）
    # ---------------------------
    print("\n开始训练...")
    epochs = 500  # 可自行调整
    batch_size = 4

    num_samples = src_data.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # 外层epoch进度条
    for epoch in tqdm(range(1, epochs + 1), desc="训练总进度", unit="epoch"):
        model.train()
        total_loss = 0.0

        # 内层batch进度条
        batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
        for i in batch_pbar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            src_batch = src_data[start_idx:end_idx]
            tgt_batch = tgt_data[start_idx:end_idx]

            # decoder input: BOS + tgt[:-1]
            tgt_input = torch.cat([
                torch.full((tgt_batch.size(0), 1), bos_idx, dtype=torch.long, device=device),
                tgt_batch[:, :-1]
            ], dim=1)

            src_mask = create_padding_mask(src_batch, pad_idx).to(device)  # [batch,1,1,src_len]
            tgt_mask = create_target_mask(tgt_input, pad_idx).to(device)  # [batch,1,tgt_len,tgt_len]

            logits = model(src_batch, tgt_input, src_mask, tgt_mask, src_mask)

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_batch.reshape(-1))
            total_loss += loss.item()

            # 在batch进度条上显示当前loss
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 每20个epoch打印平均损失
        if epoch % 20 == 0 or epoch == 1:
            avg_loss = total_loss / num_batches
            tqdm.write(f"Epoch {epoch:3d}/{epochs} | 平均损失: {avg_loss:.4f}")  # 使用tqdm.write避免进度条混乱

    # ---------------------------
    # 3.5 保存模型权重
    # ---------------------------
    save_path = "weights/transformer_translation.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "src_inv_vocab": src_inv_vocab,
        "tgt_inv_vocab": tgt_inv_vocab,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "dropout": dropout,
        "max_len": max_len,
        "pad_idx": pad_idx
    }, save_path)
    print(f"\n✅ 模型权重已保存到 {save_path}")

    # ---------------------------
    # 4. 推理演示（添加解码进度条）
    # ---------------------------
    print("\n开始推理演示...")
    model.eval()
    with torch.no_grad():
        test_indices = [0, 1, 2]
        src_test = src_data[test_indices]
        tgt_test = tgt_data[test_indices]
        src_mask = create_padding_mask(src_test, pad_idx).to(device)

        memory = model.encode(src_test, src_mask)

        batch_size_eval = src_test.size(0)
        max_decode_len = tgt_test.size(1)

        ys = torch.full((batch_size_eval, 1), bos_idx, dtype=torch.long, device=device)

        # 解码过程添加进度条
        decode_pbar = tqdm(range(max_decode_len - 1), desc="推理解码中", unit="step", leave=False)
        for _ in decode_pbar:
            # 如果所有序列都已生成EOS，则提前停止
            done_mask = (ys == eos_idx).all(dim=1)
            if done_mask.sum().item() == batch_size_eval:
                break

            tgt_mask = create_target_mask(ys, pad_idx).to(device)
            out = model.decode(ys, memory, tgt_mask, src_mask)
            logits = model.output_layer(out[:, -1, :])  # [batch, vocab]
            _, next_token = torch.max(logits, dim=1)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

        for i in range(batch_size_eval):
            print(f"\n样本 {i + 1}:")
            src_tokens = [src_inv_vocab[idx.item()] for idx in src_test[i] if idx != pad_idx]
            print(f"源中文: {' '.join(src_tokens)}")

            tgt_tokens = [tgt_inv_vocab[idx.item()] for idx in tgt_test[i] if idx != pad_idx and idx != eos_idx]
            print(f"目标英文: {' '.join(tgt_tokens)}")

            pred_tokens = [tgt_inv_vocab.get(idx.item(), "<UNK>") for idx in ys[i, 1:] if
                           idx != pad_idx and idx != eos_idx]
            print(f"预测英文: {' '.join(pred_tokens)}")


if __name__ == "__main__":
    main()