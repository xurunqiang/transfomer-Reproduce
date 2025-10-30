from typing import List

import json
from typing import List, Dict


# def read_zh_en_json(file_path: str) -> List[Dict[str, str]]:
#     """
#     读取JSON文件中的中英文平行语料
#
#     参数:
#         file_path: JSON文件路径
#     返回:
#         包含{"english": ..., "chinese": ...}的列表
#     """
#     corpus = []
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             # 处理两种常见格式：1. 每行一个JSON对象；2. 标准JSON数组
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 # 解析单行JSON
#                 try:
#                     data = json.loads(line)
#                     # 验证是否包含所需字段
#                     if "english" in data and "chinese" in data:
#                         corpus.append({
#                             "english": data["english"].strip(),
#                             "chinese": data["chinese"].strip()
#                         })
#                 except json.JSONDecodeError:
#                     # 尝试将整个文件视为JSON数组（如果单行解析失败）
#                     f.seek(0)  # 回到文件开头
#                     try:
#                         array_data = json.load(f)
#                         for item in array_data:
#                             if "english" in item and "chinese" in item:
#                                 corpus.append({
#                                     "english": item["english"].strip(),
#                                     "chinese": item["chinese"].strip()
#                                 })
#                         break  # 解析数组后退出循环
#                     except json.JSONDecodeError as e:
#                         print(f"文件格式错误，无法解析为JSON：{e}")
#                         return []
#         print(f"成功读取 {len(corpus)} 条中英文平行语料")
#     except FileNotFoundError:
#         print(f"错误：文件 {file_path} 不存在")
#     except Exception as e:
#         print(f"读取文件时发生错误：{e}")
#     return corpus
#
# def read_chinese_sentences(file_path: str) -> List[str]:
#     """
#     读取文本文件中的中文句子，返回分词后的句子列表
#
#     参数:
#         file_path: 文本文件路径
#     返回:
#         包含分词句子的列表
#     """
#     chinese_sentences = []
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             # 读取文件所有行，去除空行和首尾空白
#             for line in file:
#                 stripped_line = line.strip()
#                 if stripped_line:  # 只处理非空行
#                     chinese_sentences.append(stripped_line)
#     except FileNotFoundError:
#         print(f"错误：找不到文件 {file_path}")
#     except Exception as e:
#         print(f"读取文件时发生错误：{e}")
#     return chinese_sentences


# # 使用示例
# if __name__ == "__main__":
#     # 替换为你的文本文件路径
#     file_path = "Chinese.txt"
#     chinese_sentences = read_chinese_sentences(file_path)
#
#     # 打印结果（可选）
#     print("chinese_sentences: List[str] = [")
#     for i, sentence in enumerate(chinese_sentences):
#         # 处理最后一行的逗号
#         comma = "," if i != len(chinese_sentences) - 1 else ""
#         print(f'    "{sentence}"{comma}')
#     print("]")

# 使用示例
# if __name__ == "__main__":
#     # 替换为你的JSON文件路径
#     json_file = "translation2019zh_train.json"
#     parallel_corpus = read_zh_en_json(json_file)
#
#     # 打印前3条数据（示例）
#     for i, pair in enumerate(parallel_corpus[:3]):
#         print(f"\n句对 {i + 1}:")
#         print(f"英文: {pair['english']}")
#         print(f"中文: {pair['chinese']}")

import json


def split_and_save_parallel_corpus(json_path: str):
    """
    从JSON文件中提取中文和英文，分别保存到chinese_2019.txt和english_2019.txt
    """
    try:
        # 打开两个输出文件（按行写入，一一对应）
        with open("../data/chinese_2019_train.txt", "w", encoding="utf-8") as zh_f, \
                open("../data/english_2019_train.txt", "w", encoding="utf-8") as en_f, \
                open(json_path, "r", encoding="utf-8") as json_f:

            for line in json_f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                # 解析单行JSON
                data = json.loads(line)
                # 提取中英文并去除首尾空白
                chinese = data.get("chinese", "").strip()
                english = data.get("english", "").strip()
                # 分别写入对应的文件（每行一句，保持句对对应）
                if chinese:
                    zh_f.write(chinese + "\n")
                if english:
                    en_f.write(english + "\n")  # 修正：这里写入英文文件

        print("文件保存成功：")
        print(f"中文句子已保存到 chinese_2019_train.txt")
        print(f"英文句子已保存到 english_2019_train.txt")

    except FileNotFoundError:
        print(f"错误：找不到JSON文件 {json_path}")
    except json.JSONDecodeError:
        print("错误：JSON格式解析失败，请检查文件内容")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")


# 使用示例
if __name__ == "__main__":
    json_file = "../translation2019zh_train.json"  # 替换为你的JSON文件路径
    split_and_save_parallel_corpus(json_file)