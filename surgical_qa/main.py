# -*- coding: utf-8 -*-
"""
企业级知识库 - 主入口
启动流程: 加载数据 → 构建/加载索引 → 启动 Gradio 服务
"""

import os
import sys

# 确保当前目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import launch_app


def check_environment():
    """检查环境配置"""
    print("\n" + "=" * 60)
    print("环境检查")
    print("=" * 60)

    # 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        print(f"✓ DASHSCOPE_API_KEY: 已配置 ({api_key[:8]}...)")
    else:
        print("✗ DASHSCOPE_API_KEY: 未配置！")
        print("  请设置环境变量:")
        print('  $env:DASHSCOPE_API_KEY="your-api-key"')
        print("  获取地址: https://dashscope.console.aliyun.com/")
        return False

    # 检查数据文件
    from config import CSV_FILE, WORD2VEC_MODEL_PATH
    if os.path.exists(CSV_FILE):
        print(f"✓ 数据文件: {CSV_FILE}")
    else:
        print(f"✗ 数据文件不存在: {CSV_FILE}")
        return False

    if os.path.exists(WORD2VEC_MODEL_PATH):
        print(f"✓ Word2Vec 模型: {WORD2VEC_MODEL_PATH}")
    else:
        print(f"✗ Word2Vec 模型不存在: {WORD2VEC_MODEL_PATH}")
        print("  请先运行 enterprise_word2vec.py 训练模型")
        return False

    # 检查依赖
    try:
        import gradio
        print(f"✓ Gradio: {gradio.__version__}")
    except ImportError:
        print("✗ Gradio 未安装，请运行: pip install gradio")
        return False

    try:
        import pymilvus
        print(f"✓ Milvus: {pymilvus.__version__}")
    except ImportError:
        print("✗ Milvus 未安装，请运行: pip install pymilvus")
        return False

    try:
        import jieba
        print(f"✓ jieba: 已安装")
    except ImportError:
        print("✗ jieba 未安装，请运行: pip install jieba")
        return False

    try:
        import gensim
        print(f"✓ gensim: {gensim.__version__}")
    except ImportError:
        print("✗ gensim 未安装，请运行: pip install gensim")
        return False

    try:
        import openai
        print(f"✓ openai: {openai.__version__}")
    except ImportError:
        print("✗ openai 未安装，请运行: pip install openai")
        return False

    try:
        from PIL import Image
        print(f"✓ Pillow: 已安装")
    except ImportError:
        print("✗ Pillow 未安装，请运行: pip install Pillow")
        return False

    print("\n✓ 环境检查通过！")
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("    企业级知识库")
    print("    Enterprise Knowledge Base")
    print("=" * 60)

    # 环境检查
    if not check_environment():
        print("\n环境检查失败，请根据提示修复问题后重试。")
        sys.exit(1)

    # 启动应用
    print("\n启动应用...")
    launch_app()


if __name__ == '__main__':
    main()
