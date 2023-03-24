# -*- coding:utf-8 -*-
# Author: lqxu

import os 

source_ext = ".md"
target_ext = ".zhihu.md"


def convert_file(source_file_path: str, target_file_path: str):
    
    last_line_has_content = cur_line_has_content = True
    
    with open(source_file_path, "r", encoding="utf-8") as reader:
        with open(target_file_path, "w", encoding="utf-8") as writer:
            for line in reader:
                
                new_line = convert_line(line)
                
                if len(new_line) == 0:
                    continue
                
                if len(new_line.strip()) == 0:  # 包含空白字符
                    cur_line_has_content = False
                else:
                    cur_line_has_content = True
                
                if cur_line_has_content or last_line_has_content:
                    writer.write(new_line)
                
                last_line_has_content = cur_line_has_content
                    
    pass 


def convert_line(source_line: str):
    
    """
    在知乎编辑器中:
        1. 所有行内公式的标识符是: $$, 不是: $
        2. 只允许有两级标题
        3. [TOC] 目录标识是没用的
    """

    # 处理标题
    if source_line.startswith("###"):
        return source_line.replace("###", "##")
    elif source_line.startswith("##"):
        return source_line.replace("##", "#")
    elif source_line.startswith("#"):
        return ""

    # 处理目录标识
    if "[TOC]" in source_line:
        return ""

    # 处理公式
    if source_line.startswith("$$"):
        return source_line
    return source_line.replace("$", "$$")


if __name__ == "__main__":
    root_dir = os.path.abspath("./")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if ".git" in dirpath:
            continue
        for filename in filenames:
            if filename.startswith("README"):
                continue
            if filename.endswith(target_ext):
                continue
            if not filename.endswith(source_ext):
                continue

            source_file_path = os.path.join(dirpath, filename)

            target_file_path = os.path.join(
                dirpath, 
                os.path.splitext(filename)[0] + target_ext
            )
            
            if not os.path.exists(target_file_path):
                print(f"开始转化 {source_file_path} 文件 ... ")
                convert_file(source_file_path, target_file_path)
                print(f"{source_file_path} 文件转换完成")
