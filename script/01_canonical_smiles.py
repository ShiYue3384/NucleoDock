import pandas as pd
from rdkit import Chem

# 1. 读入 csv（修改成你的文件名）
input_csv = "test_output/2GDI/5/2GDI.csv"
df = pd.read_csv(input_csv)

# 2. 定义一个函数，把普通SMILES转成canonical SMILES
def to_canonical(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 解析失败时可以返回原始字符串，也可以返回None，看你习惯
        return None
    return Chem.MolToSmiles(mol, canonical=True)

# 3. 对 "smiles" 这一列应用
df["canonical_smiles"] = df["smiles"].apply(to_canonical)

# 4. 导出新的 csv
output_csv = "test_output/2GDI/5/2GDI_can.csv"
df.to_csv(output_csv, index=False)

print("Done! 已生成：", output_csv)
