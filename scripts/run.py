"""
运行脚本
"""
import sys
from pathlib import Path

# 添加当前目录到 Python 路径
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from mcp.start import mcp

if __name__ == "__main__":
    mcp.run()
