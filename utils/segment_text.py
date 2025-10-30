# -*- coding: utf-8 -*-
"""
segment_text.py
----------------
读取一个 .txt 文件，对中文进行分词（用空格隔开），
并将结果写入新的 .txt 文件中。

依赖：
    pip install jieba
"""

import jieba

def segment_text(input_path: str, output_path: str):
    # 1. 读取输入文件
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"✅ 已读取输入文件：{input_path}")

    # 2. 使用 jieba 分词
    words = jieba.lcut(text)  # 返回一个词语列表
    segmented_text = " ".join(words)

    # 3. 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(segmented_text)

    print(f"✅ 分词完成！结果已保存至：{output_path}")
    print("\n🔍 示例输出：")
    print(segmented_text[:200] + ("..." if len(segmented_text) > 200 else ""))


if __name__ == "__main__":
    # 示例：你可以根据自己的路径修改
    input_file = r"C:\Users\31392\Desktop\transfomer-Reproduce\data\chinese_2019_train.txt"     # 原始中文文件
    output_file = r"C:\Users\31392\Desktop\transfomer-Reproduce\data\chinese_2019_train_segmented.txt"  # 输出分词结果文件

    segment_text(input_file, output_file)
