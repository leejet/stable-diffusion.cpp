import re
import json
import sys
import os

def minify_json(input_file, output_file):
    try:
        # 1. 读取原始 JSON 文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 写入压缩后的 JSON
        # separators=(',', ':') 会删除所有的空格和换行
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
            
        print(f"成功！压缩后的文件已保存至: {output_file}")
        
        # 统计压缩率
        orig_size = os.path.getsize(input_file)
        mini_size = os.path.getsize(output_file)
        reduction = (1 - mini_size / orig_size) * 100
        print(f"原大小: {orig_size} bytes | 压缩后: {mini_size} bytes | 减小了: {reduction:.2f}%")

    except FileNotFoundError:
        print("错误：找不到输入文件。")
    except json.JSONDecodeError:
        print("错误：该文件不是有效的 JSON 格式。")
    except Exception as e:
        print(f"发生未知错误: {e}")

def extract_to_bin(source_file, output_bin):
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 这里的正则匹配数组内容（匹配 { 和 } 之间的部分）
    # 假设你的数据格式是 0xab, 0x12 ...
    match = re.search(r'\{(.*)\}', content, re.DOTALL)
    if not match:
        print("未发现数组内容")
        return

    data_str = match.group(1)
    
    # 提取所有十六进制或十进制数字
    # 这个正则可以匹配 0xFF, 0xaf, 123 等格式
    numbers = re.findall(r'(0x[0-9a-fA-F]+|[0-9]+)', data_str)

    # 转换成字节序列
    binary_data = bytearray()
    for n in numbers:
        if n.startswith('0x'):
            binary_data.append(int(n, 16))
        else:
            binary_data.append(int(n, 10))

    with open(output_bin, 'wb') as f:
        f.write(binary_data)
    print(f"转换完成：{output_bin}, 大小: {len(binary_data)} 字节")

#extract_to_bin('vocab_umt5.hpp', 'umt5_tokenizer.json')
minify_json("umt5_tokenizer.json.bak", "umt5_tokenizer.json")
