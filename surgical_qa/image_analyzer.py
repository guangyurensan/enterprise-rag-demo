# -*- coding: utf-8 -*-
"""
企业级知识库 - 图片分析模块
使用 Qwen-VL 视觉语言模型分析用户上传的图片，
将图片内容转化为文字描述，供后续 RAG 检索使用
"""

import os
import base64
from openai import OpenAI
from config import VL_MODEL, IMAGE_ANALYSIS_PROMPT


class ImageAnalyzer:
    """图片分析器"""

    def __init__(self):
        """初始化视觉模型客户端"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量！")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        print(f"[ImageAnalyzer] 初始化完成，模型: {VL_MODEL}")

    def analyze_image(self, image_path, user_question=""):
        """
        分析图片，返回文字描述
        :param image_path: 图片文件路径
        :param user_question: 用户附带的问题（如"这个表格是什么意思"）
        :return: 图片分析结果文字描述
        """
        if not os.path.exists(image_path):
            return f"图片文件不存在: {image_path}"

        # 读取图片并转为 base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # 判断图片格式
        ext = image_path.rsplit('.', 1)[-1].lower()
        format_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}
        mime_type = format_map.get(ext, 'jpeg')
        image_url = f"data:image/{mime_type};base64,{image_data}"

        # 构建用户消息（图片 + 问题）
        user_content = [{"type": "image_url", "image_url": {"url": image_url}}]

        if user_question:
            user_content.append({"type": "text", "text": f"用户问题：{user_question}\n请结合图片回答用户的问题。"})
        else:
            user_content.append({"type": "text", "text": "请分析这张图片。"})

        try:
            response = self.client.chat.completions.create(
                model=VL_MODEL,
                messages=[
                    {"role": "system", "content": IMAGE_ANALYSIS_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=1000
            )

            analysis = response.choices[0].message.content
            print(f"[ImageAnalyzer] 图片分析完成，描述长度: {len(analysis)} 字符")
            return analysis

        except Exception as e:
            error_msg = f"图片分析失败: {str(e)}"
            print(f"[ImageAnalyzer] {error_msg}")
            return error_msg

    def analyze_image_base64(self, image_base64, user_question=""):
        """
        分析 base64 编码的图片（Gradio 直接传入）
        :param image_base64: base64 编码的图片数据
        :param user_question: 用户附带的问题
        :return: 图片分析结果文字描述
        """
        # 构建图片 URL（直接使用 base64 data URI）
        image_url = f"data:image/jpeg;base64,{image_base64}"

        user_content = [{"type": "image_url", "image_url": {"url": image_url}}]

        if user_question:
            user_content.append({"type": "text", "text": f"用户问题：{user_question}\n请结合图片回答用户的问题。"})
        else:
            user_content.append({"type": "text", "text": "请分析这张图片。"})

        try:
            response = self.client.chat.completions.create(
                model=VL_MODEL,
                messages=[
                    {"role": "system", "content": IMAGE_ANALYSIS_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=1000
            )

            analysis = response.choices[0].message.content
            print(f"[ImageAnalyzer] 图片分析完成，描述长度: {len(analysis)} 字符")
            return analysis

        except Exception as e:
            error_msg = f"图片分析失败: {str(e)}"
            print(f"[ImageAnalyzer] {error_msg}")
            return error_msg

    def get_search_query(self, image_analysis, user_question=""):
        """
        将图片分析结果转化为适合检索的查询文本
        :param image_analysis: 图片分析描述
        :param user_question: 用户原始问题
        :return: 用于检索的组合查询
        """
        parts = []
        if user_question:
            parts.append(user_question)
        if image_analysis:
            # 提取分析中的关键词句（取前200字符作为核心描述）
            core_desc = image_analysis[:200].strip()
            parts.append(core_desc)

        return ' '.join(parts)



