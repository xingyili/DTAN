import os
import re
import shutil

# 设定扫描的根目录（当前目录）
ROOT_DIR = "."
# 匹配 strftime 中包含冒号的正则，例如 strftime("%H:%M:%S")
# 它会寻找类似 %H:%M 或 %H:%M:%S 的模式
PATTERN = re.compile(r'strftime\(["\'](.*?)%H:%M(:%S)?(.*?)["\']\)')

def backup_and_fix():
    for root, dirs, files in os.walk(ROOT_DIR):
        # 跳过虚拟环境和 git 目录
        if ".venv" in root or ".git" in root or "__pycache__" in root:
            continue
            
        for file in files:
            if file.endswith(".py") and file != "fix_paths.py":
                file_path = os.path.join(root, file)
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 检查是否包含非法格式
                if PATTERN.search(content):
                    print(f"🔍 发现冲突文件: {file_path}")
                    
                    # 1. 创建备份 (.bak)
                    backup_path = file_path + ".bak"
                    if not os.path.exists(backup_path):
                        shutil.copy2(file_path, backup_path)
                        print(f"   📦 已备份至: {file}")
                    
                    # 2. 执行替换（将冒号改为连字符）
                    # 例如 %H:%M:%S -> %H-%M-%S
                    new_content = PATTERN.sub(lambda m: m.group(0).replace(":", "-"), content)
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"   ✅ 已修复时间格式并保存。")

if __name__ == "__main__":
    print("🚀 开始扫描并修复 Windows 路径非法字符...")
    backup_and_fix()
    print("\n✨ 处理完成！如果程序运行正常，你可以手动删除 .bak 文件。")