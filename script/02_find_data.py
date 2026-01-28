import pandas as pd

# 1. 读入两个csv
true_file = "test_output/2GDI/canonical_2DGI_true.csv"
score_file = "test_output/2GDI/5/2GDI_can.csv"

df_true = pd.read_csv(true_file)
df_score = pd.read_csv(score_file)

# 确保列名正确（如果你列名不一样，这里改一下就行）
col = "canonical_smiles"

# 2. 在 score 表中建立：canonical_smiles -> 行号 的映射
# 如果一个canonical_smiles在score表中出现多次，这里只取第一行
smiles_to_line = {}

for idx, smiles in df_score[col].items():
    # csv实际行号 = index + 2（+1变成从1开始，再+1加上表头那一行）
    line_no = idx + 2
    if smiles not in smiles_to_line:
        smiles_to_line[smiles] = line_no

# 3. 在 true 表中创建一个新列，写入对应的行号
def find_line(smiles):
    return smiles_to_line.get(smiles, None)  # 找不到就是None / NaN

df_true["score_row_in_score_csv"] = df_true[col].apply(find_line)

# 4. 导出结果
output_file = "test_output/2GDI/5/2GDI_score.csv"
df_true.to_csv(output_file, index=False)

print("完成！结果已保存到：", output_file)
