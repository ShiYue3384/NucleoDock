import pandas as pd

# 输入文件路径
canonical_path = "test_output/2GDI/2GDI_can.csv"   #原始总文件（can)
true_path = "test_output/2GDI/2GDI_score.csv"  #标记了行数的真文件

# 输出文件路径
output_path = "test_output/2GDI/2GDI_screen.csv"

def main():
    # 读取主文件
    df_canonical = pd.read_csv(canonical_path)

    # 读取包含行号的文件
    df_true = pd.read_csv(true_path)

    # 从 true 文件中取出行号（文件行号：从 1 开始，包含表头）
    # 转换为 pandas 行索引：index = 行号 - 2
    row_indices = (
        df_true["score_row_in_score_csv"]
        .dropna()
        .astype(int)
        .map(lambda x: x - 2)  # 关键修改：减去 2
    )

    # 过滤掉可能越界的索引（安全起见）
    row_indices = row_indices[(row_indices >= 0) & (row_indices < len(df_canonical))]

    # 初始化 True 列为 0
    df_canonical["True"] = 0

    # 将指定行的 True 置为 1
    df_canonical.loc[row_indices, "True"] = 1

    # 保存结果
    df_canonical.to_csv(output_path, index=False)
    print(f"完成，已保存到: {output_path}")

if __name__ == "__main__":
    main()
