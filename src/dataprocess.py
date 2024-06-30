import os
import json
from pathlib import Path

# 指定JSON文件路径
# 获取当前文件所在目录的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 构建目标文件的路径
data_dir = os.path.join(parent_dir, "data")
input_file_path = os.path.join(data_dir,'v3.json')  # 替换为你的JSON文件路径
output_dir = os.path.join(data_dir,'GossipCop_v3_origin')  # 输出目录
output_path = Path(output_dir)
# 确保输出目录存在
output_path.mkdir(parents=True, exist_ok=True)

# 读取原始JSON文件
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 准备两个新文件的字典
real_data = {}
fake_data = {}

# 遍历原始数据，根据'label'字段分类
for key, value in data.items():
    if value.get('label') == 'real':
        real_data[key] = value
    elif value.get('label') == 'fake':
        fake_data[key] = value

# 写入文件的函数
def write_data_to_json(data_dict, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data_dict, file, ensure_ascii=False, indent=4)

# 写入real.json
real_output_path = os.path.join(output_dir , 'real.json')
write_data_to_json(real_data, real_output_path)
print(f"Real data written to: {real_output_path}")

# 写入fake.json
fake_output_path = os.path.join(output_dir , 'fake.json')
write_data_to_json(fake_data, fake_output_path)
print(f"Fake data written to: {fake_output_path}")