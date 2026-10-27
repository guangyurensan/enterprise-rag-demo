# -*- coding: utf-8 -*-
"""
企业级知识库 - 前端对接 API 服务（FastAPI）
==============================================

职责：把检索/问答/文档管理能力以稳定 HTTP 契约暴露给前端。
检索质量与治理逻辑全部在 surgical_qa 模块内，本层不做业务逻辑。

启动：
    cd case-能问答助手
    python api_server.py                    # 默认 0.0.0.0:8000
    python api_server.py --port 8080

接口契约（前端按此消费，勿直接依赖内部 record 结构）：
    GET  /api/health                 服务与组件健康状态
    POST /api/chat                   问答（一次性返回）
    POST /api/chat/stream            问答（SSE 流式：references → tokens → done）
    POST /api/search                 仅检索（调试/来源展示）
    GET  /api/documents              文档列表（仅现行版）
    POST /api/documents              上传文档（multipart，自动触发版本治理）
    DELETE /api/documents/{doc_id}   删除文档（软删除）

引用结构 references[]：
    doc_name      原始文件名，用于追溯/下载
    display_name  展示名（干净制度名 + 版本号），前端直接显示这个
    snippet/score/parent_title/page_number

空结果约定：检索无命中时 answer 为固定文案、references 为空，
不会调用模型（避免「无资料也硬答」）。

环境变量：DEEPSEEK_API_KEY、DASHSCOPE_API_KEY（rerank）、Milvus 运行中、CORS_ORIGINS
安全说明（2026-10-25 SSO 地基）：所有业务接口要求 Bearer JWT（POST /api/auth/login
换取），身份由服务端从令牌解出，请求体里的 user_role/user_groups 仅在
AUTH_REQUIRED=0 的本地评测模式下生效。令牌 8 小时有效，HS256 签名，
密钥见 surgical_qa/auth.py。接入企业微信/钉钉 SSO 时只需替换 auth_provider。
"""

import os
import sys
import json
import uuid
import shutil
import tempfile
import argparse
import logging
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'surgical_qa'))

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# 展示名规则与 Gradio、引用侧共用一份，必须放在上面的 sys.path 插入之后
from version_policy import build_display_name  # noqa: E402

app = FastAPI(title="企业知识库问答 API", version="1.0.0")

# 前端通常在另一个端口/域名上调本服务，没有 CORS 会被浏览器直接拦掉。
# 默认放开，生产环境用 CORS_ORIGINS 收紧成具体域名。
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 检索为空时不再交给模型「自由发挥」，由代码直接给出统一口径
NO_HIT_ANSWER = ("根据现有知识库未找到与该问题相关的内容。\n\n"
                 "可能原因：相关制度尚未上传，或当前账号无权查看。")

# 模块级单例（进程生命周期内复用，首次启动约 10~30s）
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from document_manager import DocumentManager
        from data_loader import load_all_records
        from retriever import KnowledgeRetriever

        dm = DocumentManager()
        records = load_all_records(doc_manager=dm)
        _retriever = KnowledgeRetriever(records, use_vector_index=True, doc_manager=dm)
        logger.info("检索主干初始化完成: %d chunks, %d 现行文档", len(records), len(dm.list_documents()))
    return _retriever


def build_user_ctx(role: Optional[str], groups: Optional[List[str]]):
    """前端用户身份 → ACL 上下文（默认全员可读）"""
    from access_control import UserContext
    return UserContext(role=role or "employee", groups=groups or ["全员"])


# ============================================================
# 认证（2026-10-25 SSO 地基）
# 身份一律从 Bearer 令牌解出，不再信任请求体里的 user_role。
# AUTH_REQUIRED=0 可关闭鉴权（仅本地评测/压测用），生产严禁。
# ============================================================
from fastapi import Depends  # noqa: E402
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # noqa: E402

AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "1") != "0"
_bearer = HTTPBearer(auto_error=False)


def resolve_identity(
    cred: HTTPAuthorizationCredentials = Depends(_bearer),
) -> Optional["UserContext"]:
    """从 Bearer JWT 解出身份 → UserContext（含 user_id，审计落痕用）。

    AUTH_REQUIRED=0（本地评测/压测兼容模式）时返回 None，
    由各端点回退到请求体身份字段；生产环境恒为鉴权。
    """
    from access_control import UserContext

    if not AUTH_REQUIRED:
        return None

    if cred is None:
        raise HTTPException(status_code=401, detail="未登录：缺少 Authorization Bearer 令牌")
    # verify_token_active = 签名/有效期校验 + 吊销黑名单（登出即失效）
    from auth import verify_token_active
    claims = verify_token_active(cred.credentials)
    if not claims:
        raise HTTPException(status_code=401, detail="令牌无效、已过期或已登出，请重新登录")
    return UserContext(
        role=claims.get("role", "employee"),
        groups=claims.get("groups", ["全员"]),
        user_id=claims.get("sub", ""),
        username=claims.get("username", ""),
        display_name=claims.get("display_name", ""),
    )


def fallback_ctx(user_ctx, req_role=None, req_groups=None):
    """鉴权关闭时的评测兼容回退：用请求体身份字段（生产模式 user_ctx 恒非空）"""
    return user_ctx if user_ctx is not None else build_user_ctx(req_role, req_groups)


@app.post("/api/auth/login")
def auth_login(body: dict):
    """账号密码登录 → 签发 JWT。凭据错误统一 401（不区分账号不存在/密码错）"""
    from auth import auth_provider, issue_token, Identity

    ident = auth_provider.authenticate(body or {})
    if ident is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = issue_token(ident.to_claims())
    return {
        "token": token,
        "expires_in": 8 * 3600,
        "user": {
            "user_id": ident.user_id,
            "username": ident.username,
            "display_name": ident.display_name,
            "role": ident.role,
            "groups": ident.groups,
        },
    }


@app.get("/api/auth/me")
def auth_me(user_ctx=Depends(resolve_identity)):
    """校验令牌有效性并返回当前身份（前端刷新页面时用）"""
    return {
        "user_id": user_ctx.user_id,
        "username": user_ctx.username,
        "display_name": user_ctx.display_name,
        "role": user_ctx.role,
        "groups": user_ctx.groups,
    }


@app.post("/api/auth/logout")
def auth_logout(cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """登出：当前令牌立即进吊销黑名单（服务端失效，不只是前端删缓存）"""
    from auth import revoke_token
    if cred is None or not revoke_token(cred.credentials):
        # 令牌本来就无效也返回成功——登出永远"成功"，不给探测口子
        return {"logged_out": True}
    return {"logged_out": True}


def build_references(results: List[dict]) -> List[dict]:
    """检索结果 → 前端引用结构（契约稳定，内部字段不再外泄）"""
    refs = []
    for r in results:
        record = r.get("record", {}) or {}
        # 展示名用干净制度名 + 版本；doc_name 保留原始文件名供下载/追溯
        base_title = r.get("base_title", "") or record.get("base_title", "")
        version_label = r.get("version_label", "") or record.get("version_label", "")
        doc_name = r.get("doc_name", "") or record.get("doc_name", "")
        display_name = build_display_name(base_title, version_label, doc_name)
        refs.append({
            "chunk_id": record.get("chunk_id", ""),
            "doc_id": r.get("doc_id", "") or record.get("doc_id", ""),
            "doc_name": r.get("doc_name", "") or record.get("doc_name", ""),
            "display_name": display_name,
            "page_number": r.get("page_number", -1) if r.get("page_number", -1) is not None else -1,
            "parent_title": r.get("parent_title", "") or record.get("parent_title", ""),
            "snippet": (r.get("content", "") or "")[:200],
            "score": round(r.get("rerank_score", r.get("rrf_score", 0)) or 0, 4),
        })
    return refs


# ============================================================
# 请求/响应模型
# ============================================================
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    user_role: Optional[str] = Field(None, description="用户角色，如 admin/employee")
    user_groups: Optional[List[str]] = Field(None, description="用户所属权限组")
    history: List[dict] = Field(default_factory=list, description="对话历史 [{role, content}]")
    top_k: int = Field(5, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_role: Optional[str] = None
    user_groups: Optional[List[str]] = None
    top_k: int = Field(5, ge=1, le=20)


def _history_text(history: List[dict]) -> str:
    """对话历史 → Query 改写所需的文本（最近 6 条）"""
    parts = [f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:]]
    return "\n".join(parts)


# ============================================================
# 问答
# ============================================================
def _audit_answer(user_ctx, query: str, answer: str, results, refused: bool,
                  latency_ms: float, llm_called: bool = True):
    """答案侧审计（等保可追溯：系统最终返回了什么）。审计失败不影响主流程。"""
    try:
        from access_control import audit_logger
        cite_docs = []
        for r in results or []:
            dn = r.get("doc_name", "")
            if dn and dn not in cite_docs:
                cite_docs.append(dn)
        audit_logger.log_answer(
            user_ctx=user_ctx, query=query,
            answer_chars=len(answer or ""), refused=refused,
            citation_docs=cite_docs, latency_ms=latency_ms,
            llm_called=llm_called,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"答案审计写入失败（不影响响应）: {e}")


@app.post("/api/chat")
def chat(req: ChatRequest, user_ctx=Depends(resolve_identity)):
    import time
    user_ctx = fallback_ctx(user_ctx, req.user_role, req.user_groups)
    retriever = get_retriever()
    _t0 = time.time()

    results, rewrite_info = retriever.retrieve_with_rewrite(
        req.query, conversation_history=_history_text(req.history),
        top_k=req.top_k, user_ctx=user_ctx,
    )
    rewrite = {
        "type": rewrite_info.get("rewrite_type", ""),
        "rewritten_query": rewrite_info.get("rewritten_query", ""),
        "sub_queries": rewrite_info.get("sub_queries", []),
    }
    if not results:
        _audit_answer(user_ctx, req.query, NO_HIT_ANSWER, [], True,
                      (time.time() - _t0) * 1000, llm_called=False)
        return {"answer": NO_HIT_ANSWER, "references": [], "rewrite": rewrite}

    answer = "".join(retriever.generate_answer(req.query, results)).strip()
    _audit_answer(user_ctx, req.query, answer, results, False,
                  (time.time() - _t0) * 1000)

    return {
        "answer": answer,
        "references": build_references(results),
        "rewrite": rewrite,
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, user_ctx=Depends(resolve_identity)):
    """SSE 流式问答。事件序列: references → 多个 token → done"""
    import time
    user_ctx = fallback_ctx(user_ctx, req.user_role, req.user_groups)
    retriever = get_retriever()

    def event(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        _t0 = time.time()
        results, rewrite_info = retriever.retrieve_with_rewrite(
            req.query, conversation_history=_history_text(req.history),
            top_k=req.top_k, user_ctx=user_ctx,
        )
        yield event({"type": "references", "references": build_references(results)})

        if not results:
            # 零召回不调模型，直接给统一口径（前端按 done 收尾即可）
            _audit_answer(user_ctx, req.query, NO_HIT_ANSWER, [], True,
                          (time.time() - _t0) * 1000, llm_called=False)
            yield event({"type": "token", "content": NO_HIT_ANSWER})
            yield event({"type": "done"})
            return

        parts = []
        try:
            for token in retriever.generate_answer(req.query, results):
                parts.append(token)
                yield event({"type": "token", "content": token})
        finally:
            # 流式中断/异常也要留痕：finally 保证答案审计不丢
            _audit_answer(user_ctx, req.query, "".join(parts), results, False,
                          (time.time() - _t0) * 1000)

        yield event({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ============================================================
# 检索
# ============================================================
@app.post("/api/search")
def search(req: SearchRequest, user_ctx=Depends(resolve_identity)):
    user_ctx = fallback_ctx(user_ctx, req.user_role, req.user_groups)
    retriever = get_retriever()
    results = retriever.retrieve_hybrid(req.query, top_k=req.top_k, user_ctx=user_ctx)
    return {"query": req.query, "results": build_references(results)}


# ============================================================
# 文档管理
# ============================================================
@app.get("/api/documents")
def list_documents(include_deleted: bool = False, user_ctx=Depends(resolve_identity)):
    user_ctx = fallback_ctx(user_ctx)  # 鉴权关闭时的评测兼容
    retriever = get_retriever()
    docs = retriever.doc_manager.list_documents()
    out = []
    for d in docs:
        if not include_deleted and (d.get("is_deleted") or not d.get("is_current", True)):
            continue
        out.append({
            "doc_id": d.get("doc_id", ""),
            "filename": d.get("filename", ""),
            # 与对话引用同一个名字：前端不用再自己拼，也不会出现
            # 「列表是 ..._1792565911.pdf、引用是干净制度名」的割裂
            "display_name": build_display_name(
                d.get("base_title", ""), d.get("version_label", ""), d.get("filename", "")
            ),
            "base_title": d.get("base_title", ""),
            "version_label": d.get("version_label", ""),
            "is_current": d.get("is_current", True),
            "chunk_count": d.get("chunk_count", 0),
            # 密级与允许组（2026-10-24 ACL 定密后前端可视化）
            "access_level": d.get("access_level", "internal"),
            "allowed_groups": d.get("allowed_groups", "all"),
            "uploaded_at": d.get("uploaded_at", d.get("updated_at", "")),
        })
    return {"total": len(out), "documents": out}


def build_version_notice(version_report: dict):
    """把版本治理结果翻译成前端可直接展示的状态与提示。

    上传了却被告知「你是旧版」是最容易踩的坑——用户以为制度已更新，
    实际系统仍在用旧版。所以把判定结果放进响应体，而不是只写日志。
    """
    vr = version_report or {}
    if vr.get("is_current") is False:
        return "superseded", (
            "该文件被判为旧版并已退场，不会被检索到。"
            "如果它其实是最新版本，请检查文件名里的版本号是否高于现行版。"
        )
    if vr.get("superseded_others"):
        return "current_replacing_old", "已作为现行版入库，同制度的旧版已自动退场。"
    if vr.get("conflicts"):
        c = vr["conflicts"][0]
        return "current", (
            f"已作为现行版入库。注意：{c.get('marked_current')} 标注了「现行最新」，"
            f"但 {c.get('higher_version')} 版本号更高，已按版本号采用后者。"
        )
    return "current", "已作为现行版入库。"


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    tags: str = Form("", description="逗号分隔标签"),
    user_ctx=Depends(resolve_identity),
):
    from starlette.concurrency import run_in_threadpool
    from config import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB, ALL_SUPPORTED_EXTENSIONS

    user_ctx = fallback_ctx(user_ctx)
    retriever = get_retriever()
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALL_SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"不支持的文件类型: {suffix or '(无后缀)'}")

    # 落盘文件名必须保留原始文件名：registry 的 filename/base_title/版本治理
    # 都取自 basename，用 kb_upload_<uuid>.xlsx 这种临时名会让展示名和
    # 版本治理全部失真（2026-10-24 实测 9 份 xlsx 全部被注册成 uuid 名）。
    original_name = os.path.basename(file.filename or "upload") or "upload"
    tmp_dir = os.path.join(tempfile.gettempdir(), f"kb_upload_{uuid.uuid4().hex}")
    tmp_path = os.path.join(tmp_dir, original_name)
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        # 分块落盘：一次性 read() 会把 200MB 上限变成内存上限，大文件直接撑爆进程
        size = 0
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 {MAX_UPLOAD_SIZE_MB}MB 上限",
                    )
                f.write(chunk)

        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        # 解析是同步阻塞的（docling 单份可跑 100s+），丢线程池避免卡住整个事件循环
        parsed = await run_in_threadpool(
            retriever.add_document, tmp_path, tags=tag_list or None
        )
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if parsed.status != "success":
        raise HTTPException(status_code=400, detail=f"文档入库失败: {parsed.status} {parsed.error_msg}")

    # 文档操作审计（API 层此前漏记，与 Gradio 台账对齐）
    try:
        from access_control import audit_logger
        audit_logger.log_document_operation(
            user_ctx, "upload", parsed.doc_id, True, f"file={file.filename}",
        )
    except Exception:  # noqa: BLE001
        pass

    info = retriever.doc_manager.get_document_info(parsed.doc_id) or {}
    vr = (parsed.metadata or {}).get("version_report") or {}
    version_status, notice = build_version_notice(vr)

    return {
        "doc_id": parsed.doc_id,
        "filename": info.get("filename", file.filename),
        "base_title": info.get("base_title", ""),
        "version_label": info.get("version_label", ""),
        "chunk_count": info.get("chunk_count", 0),
        "version_status": version_status,
        "superseded_others": vr.get("superseded_others", []),
        "version_conflicts": vr.get("conflicts", []),
        "notice": notice,
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, user_ctx=Depends(resolve_identity)):
    user_ctx = fallback_ctx(user_ctx)
    retriever = get_retriever()
    if not retriever.remove_document(doc_id):
        raise HTTPException(status_code=404, detail=f"文档不存在或已删除: {doc_id}")
    try:
        from access_control import audit_logger
        audit_logger.log_document_operation(user_ctx, "delete", doc_id, True)
    except Exception:  # noqa: BLE001
        pass
    return {"deleted": True, "doc_id": doc_id}


# ============================================================
# 健康
# ============================================================
@app.get("/api/audit/summary")
def audit_summary(date: str = None, user_ctx=Depends(resolve_identity)):
    """审计概览（供运维页展示）：今日事件统计 + 敏感命中明细。

    敏感命中 = 检索结果里出现 confidential/restricted 密级文档的查询。
    仅管理员可看；AUTH_REQUIRED=0 的评测模式下不限制（本机调试）。
    """
    if user_ctx is not None and user_ctx.role != "admin":
        raise HTTPException(status_code=403, detail="审计概览仅系统管理员可查看")
    from access_control import audit_logger
    stats = audit_logger.get_stats(date=date)
    sensitive = audit_logger.query_logs(
        date=date, sensitive_only=True, limit=10
    )
    # 明细只出必要字段，避免把审计原始结构整个抛给前端
    return {
        "date": date,
        "stats": stats,
        "recent_sensitive": [
            {
                "time": e.get("timestamp", "")[:19],
                "role": e.get("user_role", ""),
                "query": e.get("query", ""),
                "hit_levels": e.get("hit_levels", []),
            }
            for e in reversed(sensitive)  # 最新在前
        ],
    }


@app.get("/api/health")
def health():
    try:
        retriever = get_retriever()
        docs = retriever.doc_manager.list_documents()
        degraded = getattr(retriever, "degraded", False)
        return {
            # 向量通道故障时服务降级为仅 BM25，仍可用但召回质量下降——运维要能看到
            "status": "degraded" if degraded else "ok",
            "degraded_reason": getattr(retriever, "degraded_reason", "") if degraded else "",
            "chunks": len(retriever.records),
            "current_documents": sum(1 for d in docs if d.get("is_current", True) and not d.get("is_deleted")),
            "vector_index": retriever.use_vector_index,
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)[:200]}


@app.get("/api/metrics")
def api_metrics():
    """运维指标出口：计数器 + 延迟分布（avg/p95/max）。

    看什么：search_empty 持续走高说明召回出问题；llm_error/rerank 熔断说明
    外部依赖不稳；parse_fail 对应上传解析失败率。
    """
    from metrics import snapshot
    snap = snapshot()
    # 追加运行时组件状态，方便一眼看全
    try:
        retriever = get_retriever()
        snap["components"] = {
            "vector_index": retriever.use_vector_index,
            "degraded": getattr(retriever, "degraded", False),
            "chunks": len(retriever.records),
        }
    except Exception as e:  # noqa: BLE001
        snap["components"] = {"error": str(e)[:100]}
    return snap


# ============================================================
# 前端静态资源（零构建 SPA，挂载在所有 /api 路由之后——
# FastAPI 按注册顺序匹配，API 路由优先，其余路径落到静态文件）
# ============================================================
from fastapi.staticfiles import StaticFiles  # noqa: E402

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
