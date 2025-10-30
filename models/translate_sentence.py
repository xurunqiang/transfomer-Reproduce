import torch
from models.transformer import Transformer, create_padding_mask, create_target_mask  # 你定义的模块路径

# 加载模型
checkpoint = torch.load("../weights/transformer_translation.pth", map_location="cpu")

# 读取超参数
d_model = checkpoint["d_model"]
num_layers = checkpoint["num_layers"]
num_heads = checkpoint["num_heads"]
d_ff = checkpoint["d_ff"]
dropout = checkpoint["dropout"]
max_len = checkpoint["max_len"]
pad_idx = checkpoint["pad_idx"]

src_vocab = checkpoint["src_vocab"]
tgt_vocab = checkpoint["tgt_vocab"]
src_inv_vocab = checkpoint["src_inv_vocab"]
tgt_inv_vocab = checkpoint["tgt_inv_vocab"]

# 重建模型
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    dropout=dropout,
    max_len=max_len,
    pad_idx=pad_idx
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ 模型已加载完成！")

def translate_sentence(model, sentence: str, src_vocab, tgt_vocab, tgt_inv_vocab, device, max_len=50):
    model.eval()
    pad_idx = src_vocab["<PAD>"]
    bos_idx = tgt_vocab["<BOS>"]
    eos_idx = tgt_vocab["<EOS>"]

    src_tokens = sentence.split()
    src_encoded = torch.tensor([[src_vocab.get(tok, src_vocab["<UNK>"]) for tok in src_tokens]], dtype=torch.long).to(device)
    src_mask = create_padding_mask(src_encoded, pad_idx).to(device)

    with torch.no_grad():
        memory = model.encode(src_encoded, src_mask)

    ys = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

    print("🔍 解码过程:")
    for step in range(max_len):
        tgt_mask = create_target_mask(ys, pad_idx).to(device)
        out = model.decode(ys, memory, tgt_mask, src_mask)
        logits = model.output_layer(out[:, -1, :])
        next_token = torch.argmax(logits, dim=-1)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

        word = tgt_inv_vocab[next_token.item()]
        print(f"  第 {step+1} 个词 → {word}")

        if next_token.item() == eos_idx:
            break

    pred_tokens = [tgt_inv_vocab[idx.item()] for idx in ys[0, 1:] if idx.item() not in [pad_idx, eos_idx]]
    print("🟢 最终翻译结果:", " ".join(pred_tokens))
    return " ".join(pred_tokens)



# 测试
test_sentence = "早晨 妈妈 在 图书馆 做 早餐"
print("\n输入句子:", test_sentence)
pred_translation = translate_sentence(model, test_sentence, src_vocab, tgt_vocab, tgt_inv_vocab, "cpu")
print("预测翻译:", pred_translation)
