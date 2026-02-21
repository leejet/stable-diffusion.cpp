import re
import os
import sys

def file_to_c_array(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    cont = []
    for i, b in enumerate(data):
        val = ""
        if (i + 1) % 16 == 0:
            val = "\n"
        val += f'{b}';
        cont.append(val)

    return ",".join(cont)

def process_header(input_path, output_hpp):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r'char\s+(\w+)\[\]\s*=?\s*\{\s*#embed\s+"([^"]+)"\s*\};')
    matches = pattern.findall(content)
    if not matches:
        print("No #embed found in vocab.hpp")
        return

    out_content = f'#pragma once\n\n'
    for var_name, file_path in matches:
        print(f"Embedding {file_path} into {var_name}...")
        hex_data = file_to_c_array("../"+file_path) # in 'script'
        out_content += f"static const unsigned char {var_name}[] = {{\n{hex_data}\n}};\n"

    with open(output_hpp, "w") as f: f.write(out_content)

if __name__ == "__main__":
    # Usage: python embed_fix.py <vocab.hpp> <out.hpp>
    process_header(sys.argv[1], sys.argv[2])
