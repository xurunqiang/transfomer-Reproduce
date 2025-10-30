# 最简洁：统计行数（含空行）
line_count = sum(1 for _ in open(r'/data/english_2019_train.txt', 'r', encoding='utf-8'))
print('总行数：', line_count)
