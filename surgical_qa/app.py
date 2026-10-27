# -*- coding: utf-8 -*-
"""
企业级知识库 - Gradio 交互界面
提供 Web 端对话界面，支持文档管理、图片上传、检索结果展示
"""

import os
import json
import gradio as gr
from data_loader import load_csv_data, get_all_departments
from retriever import KnowledgeRetriever
from document_manager import DocumentManager
from access_control import UserContext, AccessPolicy, clean_groups, audit_logger
from config import ALL_SUPPORTED_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from version_policy import build_display_name
from evaluator import EvalSetManager, EvalRunner, GrayComparator, save_report, list_reports, format_report_md


# ============================================================
# 全局变量
# ============================================================
retriever = None
categories = []  # 文档分类


def init_system():
    """初始化系统组件"""
    global retriever, categories

    print("\n" + "=" * 60)
    print("企业级知识库 - 初始化")
    print("=" * 60)

    # 初始化文档管理器
    doc_manager = DocumentManager()

    # 加载数据（CSV + 已上传的文档）
    records = []
    try:
        records = load_csv_data()
    except Exception:
        print("[INFO] 无 CSV 数据")

    # 加载已上传的文档
    try:
        from data_loader import load_documents_as_records
        doc_records = load_documents_as_records(doc_manager)
        records.extend(doc_records)
        print(f"[OK] 加载 {len(doc_records)} 个文档 chunks")
    except Exception as e:
        print(f"[WARN] 加载文档失败：{e}")

    # 获取文档分类
    categories = get_all_departments(records)

    # 初始化检索器（启用向量索引）
    retriever = KnowledgeRetriever(
        records,
        use_vector_index=True, doc_manager=doc_manager
    )

    print("\n系统初始化完成！")
    print("提示: 请切换到「📁 文档管理」Tab 上传企业文档")
    return records


# ============================================================
# 对话处理函数
# ============================================================
def chat_stream_fn(message, history, category, user_role, user_groups_str):
    """
    流式对话处理函数（纯文字），集成 Query 改写 + 缓存 + 混合检索
    """
    if not message.strip():
        yield "请输入您的问题。"
        return

    if retriever is None:
        yield "系统尚未初始化完成，请稍后再试。"
        return

    try:
        dept = category if category and category != '全部' else ''

        # 构建用户上下文（骨架层一期固化，ACL 不可绕过）
        groups = clean_groups(user_groups_str) if user_groups_str else ["全员"]
        user_ctx = UserContext(role=user_role or "employee", groups=groups)

        # 首次对话检测
        is_first_message = len(history) == 0

        # 检查缓存
        cached = retriever.cache_manager.get(message, dept)
        if cached:
            yield f"\u26a1 *缓存命中*\n\n---\n\n{cached}"
            return

        # 格式化对话历史
        conversation_history = retriever.query_rewriter.format_history(history)

        # 带 Query 改写的检索（使用混合检索，强制 ACL 过滤）
        results, rewrite_info = retriever.retrieve_with_rewrite(
            message, conversation_history=conversation_history,
            department=dept if dept else None, use_hybrid=True,
            user_ctx=user_ctx,
        )

        # 过滤低相似度结果
        # 新检索主干：按 rerank_score 过滤；降级模式：按 rrf_score 过滤
        results = [
            r for r in results
            if r.get('rerank_score', r.get('rrf_score', 0.0)) > 0
        ]
        # 按 rerank 分数降序
        results.sort(key=lambda x: x.get('rerank_score', x.get('rrf_score', 0.0)), reverse=True)

        # 零召回必须在这里截断：把空上下文交给模型，等于让它凭通用知识作答，
        # 而「不编造」不该只靠 prompt 约束。同时省掉一次无效生成。
        if not results:
            yield ("根据现有知识库未找到与该问题相关的内容。\n\n"
                   "可能原因：相关制度尚未上传，或当前账号无权查看。"
                   "可以换个说法再问一次。")
            return

        # 流式生成回答
        full_response = ""

        # 显示改写信息（仅当有改写时）
        if rewrite_info['rewrite_type'] != "无需改写":
            yield f"🔄 *Query改写 ({rewrite_info['rewrite_type']})*: {rewrite_info['rewritten_query']}\n\n---\n\n"

        for chunk in retriever.generate_answer(message, results):
            full_response += chunk
            prefix = f"🔄 *Query改写 ({rewrite_info['rewrite_type']})*: {rewrite_info['rewritten_query']}\n\n---\n\n" if rewrite_info['rewrite_type'] != "无需改写" else ""
            yield prefix + full_response

        # 添加参考资料信息（带引文、置信度，已按相似度排序）
        ref_info = "\n\n---\n**📚 参考资料:**\n"
        if results:
            for i, r in enumerate(results[:3], 1):
                record = r.get('record', {})
                score = r.get('rerank_score', r.get('rrf_score', 0.0))
                # rerank_score 范围 0~1
                confidence = "🟢 高" if score >= 0.5 else "🟡 中" if score >= 0.2 else "⚪ 低"
                question_preview = record.get('question', '')[:40] + "..." if len(record.get('question', '')) > 40 else record.get('question', '')
                dept = record.get('department', '')

                # 引文信息
                doc_name = r.get('doc_name', '') or record.get('doc_name', '')
                page_number = r.get('page_number', -1) or record.get('page_number', -1)
                parent_title = r.get('parent_title', '') or record.get('parent_title', '')

                cite_parts = []
                # 与 API 引用、文档列表共用同一套命名规则
                cite_doc = build_display_name(
                    r.get('base_title', '') or record.get('base_title', ''),
                    r.get('version_label', '') or record.get('version_label', ''),
                    doc_name,
                )
                if cite_doc:
                    cite_parts.append(cite_doc)
                if page_number and page_number >= 0:
                    cite_parts.append(f"P{page_number}")
                if parent_title:
                    cite_parts.append(parent_title)
                cite_str = ' | '.join(cite_parts) if cite_parts else ''

                ref_info += f"- **[{dept}]** {question_preview}\n"
                if cite_str:
                    ref_info += f"  📎 {cite_str}\n"
                ref_info += f"  相关度: {score:.4f} | 置信度: {confidence}\n"
        else:
            ref_info += "\n*未检索到相关内容*\n"

        prefix = f"🔄 *Query改写 ({rewrite_info['rewrite_type']})*: {rewrite_info['rewritten_query']}\n\n---\n\n" if rewrite_info['rewrite_type'] != "无需改写" else ""
        final_output = prefix + full_response + ref_info
        yield final_output

        # 缓存回答（仅缓存成功回答）
        try:
            retriever.cache_manager.set(message, full_response, dept)
        except Exception:
            pass



    except Exception as e:
        yield f"抱歉，处理您的问题时出现错误: {str(e)}"


def chat_with_image_fn(image, message, category):
    """
    带图片的对话处理函数
    :param image: 上传的图片
    :param message: 用户文字问题
    :param category: 选择的分类
    :return: AI 回答 + 图片分析
    """
    if image is None:
        yield "请先上传一张图片。"
        return

    if retriever is None:
        yield "系统尚未初始化完成，请稍后再试。"
        return

    try:
        # 保存临时图片文件
        import tempfile
        import numpy as np
        from PIL import Image

        # Gradio 传入的 image 可能是 numpy array 或文件路径
        if isinstance(image, np.ndarray):
            # numpy array → 保存为临时文件
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, 'temp_upload.png')
            img = Image.fromarray(image)
            img.save(temp_path)
            image_path = temp_path
        elif isinstance(image, str):
            image_path = image
        else:
            # Gradio 可能传入 PIL Image
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, 'temp_upload.png')
            image.save(temp_path)
            image_path = temp_path

        dept = category if category else None
        query = message.strip() if message else ""

        # Step 1: 分析图片
        yield "🔍 正在分析图片内容，请稍候..."

        image_analysis = retriever.image_analyzer.analyze_image(image_path, query)

        # Step 2: 展示图片分析结果
        yield f"📷 **图片分析结果:**\n\n{image_analysis}\n\n---\n\n🔍 正在检索相关知识..."

        # Step 3: 检索相关知识
        search_query = retriever.image_analyzer.get_search_query(image_analysis, query)
        results = retriever.retrieve(search_query, department=dept)

        # Step 4: 生成回答
        enhanced_query = f"{query}\n\n[图片分析结果]: {image_analysis}"

        full_response = ""
        for chunk in retriever.generate_answer(enhanced_query, results):
            full_response += chunk
            yield f"📷 **图片分析:**\n{image_analysis}\n\n---\n\n💬 **回答:**\n{full_response}"

        # 添加参考资料（带引文）
        ref_info = "\n\n---\n**📚 参考资料:**\n"
        for i, r in enumerate(results[:3], 1):
            record = r.get('record', {})
            score = r.get('rerank_score', r.get('rrf_score', r.get('score', 0.0)))
            dept = record.get('department', '')
            doc_name = r.get('doc_name', '') or record.get('doc_name', '')
            page_number = r.get('page_number', -1) or record.get('page_number', -1)
            cite_parts = []
            if doc_name:
                cite_parts.append(doc_name)
            if page_number and page_number >= 0:
                cite_parts.append(f"P{page_number}")
            cite_str = ' | '.join(cite_parts) if cite_parts else ''
            ref_info += f"- [{dept}] 相关度: {score:.3f}"
            if cite_str:
                ref_info += f" | 📎 {cite_str}"
            ref_info += "\n"

        yield f"📷 **图片分析:**\n{image_analysis}\n\n---\n\n💬 **回答:**\n{full_response}{ref_info}"

    except Exception as e:
        yield f"抱歉，处理您的问题时出现错误: {str(e)}"


# ============================================================
# Gradio 界面构建
# ============================================================
def create_ui():
    """创建 Gradio 界面"""

    # 预设示例问题（通用）
    DEFAULT_EXAMPLES = [
        ["公司年假政策是什么？"],
        ["如何提交报销申请？"],
        ["项目进度如何查询？"],
        ["员工培训计划有哪些？"],
        ["公司规章制度在哪里查看？"],
        ["绩效考核标准是什么？"],
    ]

    def get_examples_for_category(cat):
        """根据分类获取预设问题"""
        return DEFAULT_EXAMPLES

    with gr.Blocks(
        title="企业级知识库",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .disclaimer { font-size: 0.85em; color: #666; padding: 10px; background: #f5f5f5; border-radius: 5px; }
        .image-section { border: 2px dashed #ccc; border-radius: 10px; padding: 15px; margin: 10px 0; }
        """
    ) as demo:

        gr.Markdown(
            """
            # 🏢 企业级知识库

            基于三管线文档解析 + RAG 检索增强生成，支持 PDF/Word/Excel/HTML/邮件等多种格式

            支持 **文字提问** / **图片+文字** / **企业文档管理** 三种模式
            """,
            elem_classes="main-header"
        )

        # Tab 切换：文字问答 / 图片问答
        with gr.Tabs():
            # ====== Tab 1: 文字问答 ======
            with gr.Tab("💬 文字问答"):
                with gr.Row():
                    with gr.Column(scale=3):
                        # 用户身份（骨架层一期固化）
                        with gr.Accordion("🔐 用户身份设置", open=False):
                            user_role_dropdown = gr.Dropdown(
                                choices=["admin", "manager", "employee", "guest"],
                                value="employee",
                                label="用户角色",
                                interactive=True
                            )
                            user_groups_input = gr.Textbox(
                                label="用户组（逗号分隔）",
                                placeholder="如: 人事部,全员",
                                value="全员"
                            )

                        category_dropdown = gr.Dropdown(
                            choices=["全部"] + categories,
                            value="全部",
                            label="🏷️ 选择分类（可选）",
                            interactive=True
                        )
                        # 示例问题
                        example_radio = gr.Radio(
                            choices=[q[0] for q in DEFAULT_EXAMPLES],
                            label="💡 点击选择问题",
                            interactive=True
                        )
                        chatbot_text = gr.ChatInterface(
                            fn=chat_stream_fn,
                            additional_inputs=[category_dropdown, user_role_dropdown, user_groups_input],
                            type="messages",
                            title=None,
                        )
                        # 分类切换时更新示例
                        category_dropdown.change(
                            fn=lambda cat: gr.Radio(choices=[q[0] for q in get_examples_for_category(cat)], value=None),
                            inputs=[category_dropdown],
                            outputs=[example_radio]
                        )
                        # 点击示例问题时填入聊天框
                        example_radio.change(
                            fn=lambda q: q,
                            inputs=[example_radio],
                            outputs=[chatbot_text.textbox]
                        )

            # ====== Tab 2: 图片问答 ======
            with gr.Tab("📷 图片问答"):
                gr.Markdown(
                    """
                    ### 📸 上传图片
                    支持上传：文档截图、表格、图表、流程图等
                    """
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        # 图片上传
                        image_input = gr.Image(
                            label="上传图片",
                            type="numpy",
                            sources=["upload", "clipboard"],
                        )
                        # 文字问题
                        image_question = gr.Textbox(
                            label="补充描述（可选）",
                            placeholder="如：这个图表是什么意思？/ 流程怎么走？/ 数据是否正常？",
                            lines=2
                        )
                        # 部门选择
                        img_category_dropdown = gr.Dropdown(
                            choices=["全部"] + categories,
                            value="全部",
                            label="选择分类（可选）",
                            interactive=True
                        )
                        # 提交按钮
                        submit_btn = gr.Button(
                            "🔍 分析图片并生成回答",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=3):
                        # 回答展示
                        image_answer = gr.Markdown(
                            value="*上传图片后点击按钮，系统将分析图片并生成回答*",
                            label="回答"
                        )

                # 绑定按钮事件
                submit_btn.click(
                    fn=chat_with_image_fn,
                    inputs=[image_input, image_question, img_category_dropdown],
                    outputs=[image_answer]
                )

                # 示例
                gr.Examples(
                    examples=[
                        ["公司年假政策是什么？"],
                        ["差旅报销标准是什么？"],
                        ["绩效考核流程怎么走？"],
                        ["这个文档怎么下载？"],
                    ],
                    inputs=[image_question],
                    label="💡 示例问题"
                )

            # ====== Tab 3: 检索策略对比 ======
            with gr.Tab("📊 策略对比"):
                gr.Markdown("### 🔍 检索策略对比工具")
                gr.Markdown("输入查询，对比 BM25 / 向量检索 / 混合检索 三种策略的效果")

                with gr.Row():
                    with gr.Column(scale=2):
                        compare_query = gr.Textbox(
                            label="查询文本",
                            placeholder="如：请假怎么申请",
                            lines=2
                        )
                        compare_btn = gr.Button("🚀 开始对比", variant="primary")

                    with gr.Column(scale=3):
                        compare_result = gr.Markdown(
                            value="*输入查询后点击按钮查看对比结果*",
                            label="对比结果"
                        )

                def do_compare(query):
                    if not query.strip() or retriever is None:
                        return "请输入有效的查询文本"
                    try:
                        comparison = retriever.compare_retrieval_strategies(query)
                        result = f"### 查询: {comparison['query']}\n\n"
                        
                        # BM25
                        result += "**BM25 Top3:**\n"
                        if comparison['bm25_top3']:
                            for r in comparison['bm25_top3']:
                                result += f"- [{r['department']}] {r['question']}...\n"
                        else:
                            result += "- 无结果\n"
                        
                        # 向量检索
                        result += "\n**向量检索 Top3:**\n"
                        if comparison['vector_top3']:
                            for r in comparison['vector_top3']:
                                result += f"- [{r['department']}] {r['question']}...\n"
                        else:
                            result += "- ⚠️ 向量索引未启用\n"
                        
                        # 混合检索
                        result += "\n**混合检索 Top3:**\n"
                        if comparison['hybrid_top3']:
                            for r in comparison['hybrid_top3']:
                                result += f"- [{r['department']}] {r['question']}...\n"
                        else:
                            result += "- 无结果\n"
                        
                        return result
                    except Exception as e:
                        return f"对比失败: {str(e)}"

                compare_btn.click(fn=do_compare, inputs=[compare_query], outputs=[compare_result])

            # ====== Tab 4: 文档管理 ======
            with gr.Tab("📁 文档管理"):
                gr.Markdown(
                    """
                    ### 📄 企业文档管理
                    支持上传 PDF、Word、Excel、HTML、邮件等文档，自动解析并入库

                    **三管线智能解析:**
                    - **MinerU**: 论文、财报、中文密集、公式/扫描件 PDF
                    - **Docling**: 合同、docx/pptx、英文文档
                    - **Unstructured**: Excel、HTML、邮件、通用办公
                    """
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        # 文档上传
                        doc_upload = gr.File(
                            label="上传文档",
                            file_types=list(ALL_SUPPORTED_EXTENSIONS),
                            file_count="multiple",
                            type="filepath"
                        )
                        # 解析器选择
                        parser_select = gr.Radio(
                            choices=["自动选择", "MinerU", "Docling", "Unstructured"],
                            value="自动选择",
                            label="解析器（默认自动选择）",
                            interactive=True
                        )
                        # 标签输入
                        doc_tags = gr.Textbox(
                            label="文档标签（逗号分隔，可选）",
                            placeholder="如: 人事,财务,制度",
                            lines=1
                        )
                        # 文档权限设置（骨架层一期固化）
                        doc_access_level = gr.Dropdown(
                            choices=["public", "internal", "restricted", "confidential"],
                            value="internal",
                            label="访问级别",
                            interactive=True
                        )
                        doc_allowed_groups = gr.Textbox(
                            label="允许组（逗号分隔，restricted 时必填）",
                            placeholder="如: 人事部,财务部",
                            lines=1
                        )
                        # 上传按钮
                        upload_btn = gr.Button(
                            "📤 上传并解析文档",
                            variant="primary",
                            size="lg"
                        )
                        upload_result = gr.Markdown(value="")

                    with gr.Column(scale=3):
                        # 文档列表
                        gr.Markdown("### 📚 已入库文档")
                        doc_list = gr.Dataframe(
                            headers=["Doc ID", "文档名称", "类型", "解析器", "块数", "版本", "状态", "权限"],
                            datatype=["str", "str", "str", "str", "number", "str", "str", "str"],
                            interactive=False,
                            wrap=True
                        )
                        refresh_docs_btn = gr.Button("🔄 刷新文档列表", size="sm")

                        # 文档操作区（更新/删除/恢复）
                        with gr.Accordion("🔧 文档操作（更新/删除/恢复）", open=False):
                            manage_doc_id = gr.Textbox(
                                label="文档 ID（从列表复制）",
                                placeholder="粘贴要操作的 doc_id"
                            )
                            with gr.Row():
                                update_file = gr.File(
                                    label="更新文件（上传新文件替换）",
                                    file_types=list(ALL_SUPPORTED_EXTENSIONS),
                                    file_count="single"
                                )
                                update_btn = gr.Button("🔄 更新文档", variant="primary", size="sm")
                            with gr.Row():
                                delete_btn = gr.Button("🗑️ 软删除文档", variant="stop", size="sm")
                                restore_btn = gr.Button("♻️ 恢复文档", size="sm")
                            manage_result = gr.Markdown(value="")

                        # 文档统计
                        gr.Markdown("### 📊 文档库统计")
                        doc_stats = gr.Markdown(value="")

                def handle_upload(files, parser_choice, tags_input, access_level, allowed_groups_input):
                    if not files:
                        return "⚠️ 请先选择文件", get_doc_list(), get_doc_stats()

                    parser_map = {
                        "自动选择": None, "MinerU": "mineru",
                        "Docling": "docling", "Unstructured": "unstructured",
                    }
                    parser = parser_map.get(parser_choice)
                    tags = [t.strip() for t in tags_input.split(',') if t.strip()] if tags_input else []

                    # confidential 写入管控（骨架层一期固化）
                    if access_level == "confidential":
                        # 一期硬编码：仅 admin 可写入 confidential
                        # TODO: 二期接入用户认证系统
                        pass  # 一期不阻断，仅告警
                        import logging
                        logging.getLogger(__name__).warning(
                            f"[ACL] confidential 文档上传，需确认操作者身份"
                        )

                    # restricted 组名清洗 + 告警（策略层：一期只清洗告警，不阻断）
                    cleaned_groups = ""
                    if access_level == "restricted" and allowed_groups_input:
                        group_list = clean_groups(allowed_groups_input)
                        cleaned_groups = ','.join(group_list)
                        if not group_list:
                            import logging
                            logging.getLogger(__name__).warning(
                                "[ACL] restricted 文档未设置允许组，将默认为 all"
                            )
                    elif allowed_groups_input:
                        group_list = clean_groups(allowed_groups_input)
                        cleaned_groups = ','.join(group_list)

                    results = []
                    # 构建临时用户上下文（一期默认 admin，二期接入认证系统）
                    _upload_ctx = UserContext(role="admin", groups=["全员"])
                    for file_path in files:
                        try:
                            parsed = retriever.add_document(
                                file_path, parser=parser, tags=tags,
                                access_level=access_level,
                                allowed_groups=cleaned_groups or 'all',
                            )
                            status = "✅ 成功" if parsed.status == "success" else f"❌ {parsed.error_msg[:50]}"
                            results.append(
                                f"- **{parsed.filename}**: {status} | "
                                f"解析器: {parsed.parser_used} | "
                                f"块数: {len(parsed.chunks)} | "
                                f"权限: {access_level} | "
                                f"耗时: {parsed.parse_time:.1f}s"
                            )
                            # 审计日志
                            audit_logger.log_document_operation(
                                user_ctx=_upload_ctx,
                                operation="upload",
                                doc_id=parsed.doc_id,
                                success=(parsed.status == "success"),
                                detail=f"file={parsed.filename}, parser={parsed.parser_used}, level={access_level}",
                            )
                        except Exception as e:
                            results.append(f"- **{os.path.basename(file_path)}**: ❌ 错误: {str(e)[:80]}")
                            audit_logger.log_document_operation(
                                user_ctx=_upload_ctx,
                                operation="upload",
                                doc_id="",
                                success=False,
                                detail=f"file={os.path.basename(file_path)}, error={str(e)[:100]}",
                            )

                    result_text = "\n".join(results)
                    return result_text, get_doc_list(), get_doc_stats()

                def get_doc_list():
                    if retriever is None:
                        return []
                    docs = retriever.list_documents()
                    return [
                        [
                            d.get('doc_id', ''),
                            build_display_name(
                                d.get('base_title', ''), d.get('version_label', ''),
                                d.get('filename', ''),
                            ),
                            d.get('file_type', ''),
                            d.get('parser_used', ''),
                            d.get('chunk_count', 0),
                            d.get('version_label', '') or '',
                            d.get('parse_status', ''),
                            d.get('access_level', 'internal'),
                        ]
                        for d in docs
                    ]

                def get_doc_stats():
                    if retriever is None:
                        return "系统未初始化"
                    stats = retriever.get_document_stats()
                    result = f"**文档总数**: {stats.get('total_documents', 0)}\n\n"
                    result += f"**总块数**: {stats.get('total_chunks', 0)}\n\n"
                    result += f"**总大小**: {stats.get('total_size_human', '0 B')}\n\n"

                    parser_dist = stats.get('parser_distribution', {})
                    if parser_dist:
                        result += "**解析器分布**:\n"
                        for p, cnt in parser_dist.items():
                            result += f"- {p}: {cnt} 个文档\n"

                    type_dist = stats.get('file_type_distribution', {})
                    if type_dist:
                        result += "\n**文件类型分布**:\n"
                        for t, cnt in type_dist.items():
                            result += f"- {t}: {cnt} 个\n"

                    # 解析器可用性
                    parser_avail = stats.get('parser_availability', {})
                    if parser_avail:
                        result += "\n**解析器状态**:\n"
                        for name, avail in parser_avail.items():
                            status = "✅ 可用" if avail else "❌ 未安装"
                            result += f"- {name}: {status}\n"

                    return result

                upload_btn.click(
                    fn=handle_upload,
                    inputs=[doc_upload, parser_select, doc_tags, doc_access_level, doc_allowed_groups],
                    outputs=[upload_result, doc_list, doc_stats]
                )
                refresh_docs_btn.click(fn=lambda: (get_doc_list(), get_doc_stats()),
                                       outputs=[doc_list, doc_stats])

                # 文档操作按钮处理函数（带审计日志）
                _doc_ctx = UserContext(role="admin", groups=["全员"])

                def handle_update(doc_id, new_file):
                    if not doc_id or not new_file:
                        return "⚠️ 请输入文档 ID 并上传新文件", get_doc_list(), get_doc_stats()
                    try:
                        parsed = retriever.update_document(doc_id, new_file)
                        if parsed.status == "success":
                            audit_logger.log_document_operation(_doc_ctx, "update", doc_id, True, f"new_chunks={len(parsed.chunks)}")
                            return f"✅ 文档已更新 | 版本: {parsed.doc_id} | 新块数: {len(parsed.chunks)}", get_doc_list(), get_doc_stats()
                        else:
                            audit_logger.log_document_operation(_doc_ctx, "update", doc_id, False, parsed.error_msg[:100])
                            return f"❌ 更新失败: {parsed.error_msg[:80]}", get_doc_list(), get_doc_stats()
                    except Exception as e:
                        audit_logger.log_document_operation(_doc_ctx, "update", doc_id, False, str(e)[:100])
                        return f"❌ 错误: {str(e)[:80]}", get_doc_list(), get_doc_stats()

                def handle_delete(doc_id):
                    if not doc_id:
                        return "⚠️ 请输入文档 ID", get_doc_list(), get_doc_stats()
                    success = retriever.remove_document(doc_id)
                    audit_logger.log_document_operation(_doc_ctx, "delete", doc_id, success)
                    if success:
                        return f"✅ 文档已软删除（可从回收站恢复）: {doc_id}", get_doc_list(), get_doc_stats()
                    else:
                        return f"❌ 删除失败，请检查 doc_id 是否正确", get_doc_list(), get_doc_stats()

                def handle_restore(doc_id):
                    if not doc_id:
                        return "⚠️ 请输入文档 ID", get_doc_list(), get_doc_stats()
                    success = retriever.restore_document(doc_id)
                    audit_logger.log_document_operation(_doc_ctx, "restore", doc_id, success)
                    if success:
                        return f"✅ 文档已恢复: {doc_id}", get_doc_list(), get_doc_stats()
                    else:
                        return f"❌ 恢复失败，请检查 doc_id 是否正确", get_doc_list(), get_doc_stats()

                update_btn.click(
                    fn=handle_update,
                    inputs=[manage_doc_id, update_file],
                    outputs=[manage_result, doc_list, doc_stats]
                )
                delete_btn.click(
                    fn=handle_delete,
                    inputs=[manage_doc_id],
                    outputs=[manage_result, doc_list, doc_stats]
                )
                restore_btn.click(
                    fn=handle_restore,
                    inputs=[manage_doc_id],
                    outputs=[manage_result, doc_list, doc_stats]
                )

                # 初始化时加载文档列表
                doc_list.value = get_doc_list()
                doc_stats.value = get_doc_stats()

        # ========================================
        # Tab 3: 审计日志
        # ========================================
        with gr.Tab("📊 审计日志"):
            gr.Markdown("### 🔍 审计日志查询（企业合规追溯）")
            with gr.Row():
                with gr.Column(scale=1):
                    audit_date = gr.Textbox(
                        label="查询日期",
                        placeholder="YYYY-MM-DD（留空=今天）",
                        value=""
                    )
                    audit_event_type = gr.Dropdown(
                        choices=["全部", "search", "doc_access", "document_op"],
                        value="全部",
                        label="事件类型"
                    )
                    audit_role = gr.Dropdown(
                        choices=["全部", "admin", "manager", "employee", "guest"],
                        value="全部",
                        label="用户角色"
                    )
                    audit_limit = gr.Slider(10, 200, value=50, step=10, label="最大条数")
                    query_audit_btn = gr.Button("🔍 查询日志", variant="primary")

                with gr.Column(scale=2):
                    audit_stats_display = gr.Markdown(label="统计概览")
                    audit_log_display = gr.Dataframe(
                        headers=["时间", "事件", "角色", "查询/操作", "结果数", "耗时(ms)", "通道"],
                        label="审计日志",
                        interactive=False
                    )

            def query_audit_logs(date_str, event_type, role, limit):
                date = date_str.strip() if date_str and date_str.strip() else None
                et = None if event_type == "全部" else event_type
                r = None if role == "全部" else role

                # 查询日志
                logs = audit_logger.query_logs(date=date, event_type=et, role=r, limit=int(limit))

                # 统计概览
                stats = audit_logger.get_stats(date=date)
                stats_md = f"**总记录**: {stats['total']} | **平均耗时**: {stats['avg_latency_ms']}ms\n\n"
                if stats['by_event']:
                    stats_md += "**事件分布**: " + " | ".join(f"{k}: {v}" for k, v in stats['by_event'].items()) + "\n\n"
                if stats['by_role']:
                    stats_md += "**角色分布**: " + " | ".join(f"{k}: {v}" for k, v in stats['by_role'].items())

                # 转换为表格数据
                table_data = []
                for entry in logs:
                    ts = entry.get('timestamp', '')[:19]
                    et = entry.get('event_type', '')
                    role = entry.get('user_role', '')
                    if et == 'search':
                        action = entry.get('query', '')[:40]
                        count = entry.get('result_count', 0)
                        latency = entry.get('latency_ms', 0)
                        channels = ','.join(entry.get('channels', []))
                    elif et == 'document_op':
                        action = f"{entry.get('operation', '')} {entry.get('doc_id', '')[:12]}"
                        count = 1 if entry.get('success') else 0
                        latency = 0
                        channels = '-'
                    elif et == 'doc_access':
                        action = f"doc={entry.get('doc_id', '')[:12]}"
                        count = 1 if entry.get('access_granted') else 0
                        latency = 0
                        channels = '-'
                    else:
                        action = '-'
                        count = 0
                        latency = 0
                        channels = '-'
                    table_data.append([ts, et, role, action, count, latency, channels])

                return stats_md, table_data

            query_audit_btn.click(
                fn=query_audit_logs,
                inputs=[audit_date, audit_event_type, audit_role, audit_limit],
                outputs=[audit_stats_display, audit_log_display]
            )

        # ========================================
        # Tab 4: 评测与灰度
        # ========================================
        with gr.Tab("🧪 评测与灰度"):
            gr.Markdown("### 🧪 检索质量评测 & 灰度对比")

            eval_mgr = EvalSetManager()

            with gr.Row():
                with gr.Column(scale=1):
                    eval_set_name = gr.Dropdown(
                        choices=eval_mgr.list_sets() or ["seed"],
                        value="seed" if "seed" in eval_mgr.list_sets() else "",
                        label="评测集"
                    )
                    eval_top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K")
                    run_eval_btn = gr.Button("🚀 运行评测", variant="primary")

                with gr.Column(scale=2):
                    eval_result_display = gr.Markdown(label="评测报告")

            # 灰度对比区
            gr.Markdown("---")
            gr.Markdown("### 🔄 灰度对比（A/B Test）")
            with gr.Row():
                with gr.Column():
                    gray_use_rerank = gr.Checkbox(label="启用 Rerank", value=True)
                    gray_rerank_topn = gr.Slider(1, 20, value=5, step=1, label="Rerank Top-N")
                with gr.Column():
                    gray_vector_topk = gr.Slider(5, 50, value=20, step=5, label="向量召回 Top-K")
                    gray_bm25_topk = gr.Slider(5, 50, value=20, step=5, label="BM25 召回 Top-K")
            run_gray_btn = gr.Button("🔬 运行灰度对比")
            gray_result_display = gr.Markdown(label="灰度对比报告")

            # 历史报告
            gr.Markdown("---")
            gr.Markdown("### 📊 历史报告")
            with gr.Row():
                history_reports = gr.Dataframe(
                    headers=["文件", "时间", "条数", "Recall@K", "MRR", "命中率", "延迟(ms)"],
                    label="历史评测报告",
                    interactive=False
                )
                refresh_reports_btn = gr.Button("🔄 刷新")

            def run_evaluation(set_name, top_k):
                if retriever is None:
                    return "⚠️ 系统未初始化"
                items = eval_mgr.load_set(set_name)
                if not items:
                    return f"⚠️ 评测集 '{set_name}' 为空或不存在"

                runner = EvalRunner(retriever)
                report = runner.run_eval(items, top_k=int(top_k))

                # 保存报告
                save_report(report, f"eval_{set_name}")

                return format_report_md(report)

            def run_gray_comparison(set_name, use_rerank, rerank_topn, vector_topk, bm25_topk):
                if retriever is None:
                    return "⚠️ 系统未初始化"
                items = eval_mgr.load_set(set_name)
                if not items:
                    return f"⚠️ 评测集 '{set_name}' 为空或不存在"

                comparator = GrayComparator(retriever)
                experiment_config = {
                    "use_rerank": use_rerank,
                    "rerank_top_n": int(rerank_topn),
                    "vector_top_k": int(vector_topk),
                    "bm25_top_k": int(bm25_topk),
                }
                comparison = comparator.compare(
                    items,
                    experiment_config=experiment_config,
                    top_k=5,
                )

                # 格式化对比报告
                deltas = comparison.get("deltas", {})
                md = "## 灰度对比报告\n\n"
                md += "| 指标 | 基线 | 实验 | 差值 | 变化 |\n"
                md += "|------|------|------|------|------|\n"
                for metric, v in deltas.items():
                    md += f"| {metric} | {v['baseline']} | {v['experiment']} | {v['delta']} | {v['delta_pct']} |\n"

                save_report(comparison, f"gray_{set_name}")
                return md

            def refresh_report_list():
                reports = list_reports(limit=20)
                table = []
                for r in reports:
                    m = r.get("metrics", {})
                    table.append([
                        r.get("file", ""),
                        r.get("timestamp", "")[:19],
                        r.get("total", 0),
                        f"{m.get('avg_recall@k', 0):.2%}",
                        f"{m.get('avg_mrr', 0):.4f}",
                        f"{m.get('hit_rate', 0):.2%}",
                        f"{m.get('avg_latency_ms', 0):.0f}",
                    ])
                return table

            run_eval_btn.click(
                fn=run_evaluation,
                inputs=[eval_set_name, eval_top_k],
                outputs=[eval_result_display]
            )
            run_gray_btn.click(
                fn=run_gray_comparison,
                inputs=[eval_set_name, gray_use_rerank, gray_rerank_topn, gray_vector_topk, gray_bm25_topk],
                outputs=[gray_result_display]
            )
            refresh_reports_btn.click(
                fn=refresh_report_list,
                outputs=[history_reports]
            )

        # 底部信息
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 系统信息")
                gr.Markdown(
                    f"""
                    **企业文档库** | **数据分类**: {len(categories)} 个

                    **核心能力**:
                    - 🔍 混合召回 (向量 + BM25 + RRF 融合)
                    - 🎯 Rerank 重排 (gte-rerank 精排)
                    - 🤖 智能回答 (DeepSeek-V3)
                    - 📷 图片分析 (Qwen-VL 视觉模型)
                    - 🔄 Query改写 (上下文理解)
                    - ⚡ 智能缓存 (降低成本)
                    - 🔁 API重试 (自动容错)

                    **文档解析 (三管线)**:
                    - 📄 MinerU: 论文/财报/中文密集/公式/扫描件
                    - 📑 Docling: 合同/docx/pptx/英文
                    - 📊 Unstructured: Excel/HTML/邮件/通用办公
                    """
                )

            with gr.Column():
                gr.Markdown("### ⚡ 缓存状态")
                def get_cache_stats():
                    if retriever is None:
                        return "系统未初始化"
                    stats = retriever.cache_manager.get_stats()
                    return f"""
**缓存条目**: {stats['total_entries']} / {stats['max_size']}

**命中次数**: {stats['total_hits']}

**过期时间**: {stats['ttl_hours']:.0f} 小时
"""
                cache_stats = gr.Markdown(value=get_cache_stats())
                with gr.Row():
                    refresh_cache_btn = gr.Button("🔄 刷新", size="sm")
                    clear_cache_btn = gr.Button("🗑️ 清空缓存", size="sm")
                refresh_cache_btn.click(fn=lambda: get_cache_stats(), outputs=[cache_stats])
                clear_cache_btn.click(
                    fn=lambda: (retriever.cache_manager.clear(), "缓存已清空！") if retriever else (None, "系统未初始化"),
                    outputs=[cache_stats]
                )

            with gr.Column():
                gr.Markdown("### 💾 对话导出")
                gr.Markdown("将当前对话保存为 Markdown 文件")
                export_btn = gr.Button("📥 导出对话记录", size="sm")
                export_result = gr.Markdown(value="")

                def export_conversation():
                    if retriever is None:
                        return "系统未初始化"
                    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exports')
                    os.makedirs(export_dir, exist_ok=True)
                    from datetime import datetime
                    filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    filepath = os.path.join(export_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write("# 企业级知识库 - 对话记录\n\n")
                        f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
                        f.write("*对话内容来自 Gradio 界面*\n")
                    return f"✅ 已导出到: `{filepath}`"

                export_btn.click(fn=export_conversation, outputs=[export_result])

            with gr.Column():
                gr.Markdown("### ⚠️ 免责声明")
                gr.Markdown(
                    """
                    <div class="disclaimer">
                    本系统基于企业文档库提供智能问答服务，回答内容仅供参考。
                    如有重要决策，请结合实际情况并咨询相关专业人员。
                    </div>
                    """,
                )

        gr.Markdown(
            """
            ---
            *Powered by Milvus + DeepSeek-V3 + Qwen-VL | 三管线: MinerU + Docling + Unstructured*
            """
        )

    return demo


# ============================================================
# 启动入口
# ============================================================
def launch_app():
    """启动应用"""
    # 初始化系统
    init_system()

    # 创建并启动界面
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == '__main__':
    launch_app()
