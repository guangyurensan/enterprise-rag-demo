# -*- coding: utf-8 -*-
"""
企业级知识库 - 评测与灰度模块

核心职责：
1. 评测集加载与管理（JSON 格式）
2. 自动化评测（Recall@K、MRR、命中率、延迟）
3. 评测报告生成
4. A/B 灰度对比（两套配置并行，结果对比）

设计原则：
- 评测模式跳过审计日志和上下文脱敏，确保指标纯净
- 评测集格式统一，支持增量追加
- 灰度对比结果可持久化
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# 评测集默认路径
EVAL_SET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'eval_sets'
)
EVAL_REPORT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'eval_reports'
)


# ============================================================
# 评测集数据结构
# ============================================================
# 每条评测数据格式：
# {
#     "id": "q001",
#     "query": "请假怎么申请",
#     "expected_doc_ids": ["doc_001", "doc_002"],  # 期望命中的文档 ID
#     "expected_keywords": ["请假", "审批"],          # 期望命中的关键词
#     "category": "人事",                             # 分类标签
#     "difficulty": "easy"                            # easy/medium/hard
# }


# ============================================================
# 评测集管理
# ============================================================
class EvalSetManager:
    """评测集加载与管理"""

    def __init__(self, eval_dir: str = None):
        self.eval_dir = eval_dir or EVAL_SET_DIR
        os.makedirs(self.eval_dir, exist_ok=True)

    def list_sets(self) -> List[str]:
        """列出所有评测集"""
        files = [f for f in os.listdir(self.eval_dir) if f.endswith('.json')]
        return [f.replace('.json', '') for f in files]

    def load_set(self, name: str = "seed") -> List[Dict]:
        """加载指定评测集"""
        path = os.path.join(self.eval_dir, f"{name}.json")
        if not os.path.exists(path):
            logger.warning(f"[Eval] 评测集不存在: {name}")
            return []

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get("items", [])

    def save_set(self, name: str, items: List[Dict], metadata: Dict = None):
        """保存评测集"""
        path = os.path.join(self.eval_dir, f"{name}.json")
        data = {
            "name": name,
            "metadata": metadata or {
                "created_at": datetime.now().isoformat(),
                "description": "",
            },
            "item_count": len(items),
            "items": items,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[Eval] 评测集已保存: {name} ({len(items)} 条)")

    def add_item(self, name: str, item: Dict):
        """向评测集追加条目"""
        items = self.load_set(name)
        # 自动生成 ID
        if "id" not in item:
            item["id"] = f"q{len(items) + 1:03d}"
        items.append(item)
        self.save_set(name, items)


# ============================================================
# 评测指标计算
# ============================================================
class EvalMetrics:
    """评测指标计算器"""

    @staticmethod
    def recall_at_k(results: List[Dict], expected_doc_ids: List[str], k: int = 5) -> float:
        """
        Recall@K：前 K 条结果中命中的期望文档比例
        """
        if not expected_doc_ids:
            return 1.0  # 无期望则视为完全命中

        hit_doc_ids = set()
        for r in results[:k]:
            doc_id = r.get('doc_id', '') or r.get('record', {}).get('doc_id', '')
            if doc_id:
                hit_doc_ids.add(doc_id)

        expected_set = set(expected_doc_ids)
        if not expected_set:
            return 1.0

        hit_count = len(hit_doc_ids & expected_set)
        return hit_count / len(expected_set)

    @staticmethod
    def mrr(results: List[Dict], expected_doc_ids: List[str]) -> float:
        """
        MRR (Mean Reciprocal Rank)：第一个命中的倒数排名
        """
        expected_set = set(expected_doc_ids)
        for i, r in enumerate(results):
            doc_id = r.get('doc_id', '') or r.get('record', {}).get('doc_id', '')
            if doc_id in expected_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def keyword_hit_rate(results: List[Dict], keywords: List[str]) -> float:
        """
        关键词命中率：期望关键词在结果内容中的覆盖率
        """
        if not keywords:
            return 1.0

        # 拼接所有结果内容
        all_text = ""
        for r in results:
            content = r.get('content', '')
            record = r.get('record', {})
            answer = record.get('answer', '')
            question = record.get('question', '')
            all_text += f"{content} {answer} {question} "

        hit_count = sum(1 for kw in keywords if kw in all_text)
        return hit_count / len(keywords)

    @staticmethod
    def hit_any(results: List[Dict], expected_doc_ids: List[str]) -> bool:
        """是否命中任意一个期望文档"""
        expected_set = set(expected_doc_ids)
        for r in results:
            doc_id = r.get('doc_id', '') or r.get('record', {}).get('doc_id', '')
            if doc_id in expected_set:
                return True
        return False


# ============================================================
# 评测执行器
# ============================================================
class EvalRunner:
    """评测执行器"""

    def __init__(self, retriever):
        """
        :param retriever: KnowledgeRetriever 实例
        """
        self.retriever = retriever
        self.metrics = EvalMetrics()

    def _resolve_expected(self, item: Dict):
        """
        版本感知的期望项解析（评估集不绑定具体 doc_id，换数据照样可用）：
        - expected_base_titles: 制度名列表，动态解析为该制度"现行版"的 doc_id
          （多版本制度下，召回旧版不会命中期望——专治版本污染类 badcase）
        - expected_doc_ids: 仍然直接生效（兼容旧格式）

        :return: (expected_doc_ids, missing_base_titles)

        为什么要把 missing 显式返回：期望标题解析失败时该期望项会被静默跳过，
        若全部跳过则 expected 为空 → recall_at_k 兜底返回 1.0，用例变成"永久通过"
        的空壳（2026-10-27 语料改名导致的假绿事故）。这类失效必须在门禁里可见。
        """
        expected = list(item.get("expected_doc_ids", []))
        base_titles = item.get("expected_base_titles", [])
        missing = []
        if not base_titles:
            return expected, missing
        try:
            from version_policy import resolve_current_map
            doc_manager = getattr(self.retriever, 'doc_manager', None)
            if doc_manager is None:
                logger.warning("[Eval] retriever 无 doc_manager，无法解析制度名")
                return expected, list(base_titles)
            current_map = resolve_current_map(doc_manager.registry.documents)
            for bt in base_titles:
                if bt in current_map:
                    expected.append(current_map[bt])
                else:
                    logger.warning(f"[Eval] 制度 '{bt}' 无现行版文档，该期望项跳过")
                    missing.append(bt)
        except Exception as e:
            logger.warning(f"[Eval] 解析 expected_base_titles 失败: {e}")
            missing = list(base_titles)
        return expected, missing

    def run_eval(
        self,
        eval_items: List[Dict],
        top_k: int = 5,
        user_ctx=None,
    ) -> Dict:
        """
        执行评测

        :param eval_items: 评测条目列表
        :param top_k: 取前 K 条结果计算指标
        :param user_ctx: 用户上下文（默认 admin，确保无 ACL 干扰）
        :return: 评测报告
        """
        from access_control import UserContext
        if user_ctx is None:
            user_ctx = UserContext(role="admin", groups=["全员"])

        total = len(eval_items)
        recall_scores = []
        mrr_scores = []
        kw_scores = []
        hit_count = 0
        latencies = []
        details = []
        missing_cases = []          # 期望部分/全部解析失败的用例
        empty_expectation_cases = []  # 期望全空 → recall 恒 1.0 的空壳用例

        for i, item in enumerate(eval_items):
            query = item.get("query", "")
            expected_ids, missing_titles = self._resolve_expected(item)
            keywords = item.get("expected_keywords", [])
            category = item.get("category", "")
            difficulty = item.get("difficulty", "medium")

            case_id = item.get("id", f"q{i+1:03d}")
            if missing_titles:
                miss_info = {"id": case_id, "query": query,
                             "missing_titles": missing_titles}
                missing_cases.append(miss_info)
                if not expected_ids:
                    # 兜底口径：recall_at_k 对空期望返回 1.0，该用例已失去检验能力
                    empty_expectation_cases.append(miss_info)

            # 执行检索（评测模式：跳过脱敏/审计，测量纯净检索质量）
            start = time.time()
            try:
                results = self.retriever.retrieve_hybrid(
                    query, top_k=top_k, user_ctx=user_ctx, eval_mode=True
                )
            except Exception as e:
                logger.error(f"[Eval] 检索失败: {query} -> {e}")
                results = []
            latency = (time.time() - start) * 1000

            # 计算指标
            recall = self.metrics.recall_at_k(results, expected_ids, k=top_k)
            mrr = self.metrics.mrr(results, expected_ids)
            kw_hit = self.metrics.keyword_hit_rate(results, keywords)
            hit_any = self.metrics.hit_any(results, expected_ids)

            recall_scores.append(recall)
            mrr_scores.append(mrr)
            kw_scores.append(kw_hit)
            if hit_any:
                hit_count += 1
            latencies.append(latency)

            details.append({
                "id": item.get("id", f"q{i+1:03d}"),
                "query": query,
                "category": category,
                "difficulty": difficulty,
                "recall@k": round(recall, 3),
                "mrr": round(mrr, 3),
                "keyword_hit": round(kw_hit, 3),
                "hit_any": hit_any,
                "latency_ms": round(latency, 1),
                "top_doc": results[0].get('doc_id', '') if results else '',
                "top_score": round(results[0].get('rerank_score', results[0].get('score', 0)) or 0, 4) if results else 0,
                "missing_titles": missing_titles,
            })

        # 汇总
        avg_recall = sum(recall_scores) / total if total else 0
        avg_mrr = sum(mrr_scores) / total if total else 0
        avg_kw = sum(kw_scores) / total if total else 0
        avg_latency = sum(latencies) / total if total else 0
        hit_rate = hit_count / total if total else 0

        # 按分类统计
        by_category = {}
        for d in details:
            cat = d.get("category", "未分类")
            if cat not in by_category:
                by_category[cat] = {"count": 0, "recall_sum": 0, "hit_count": 0}
            by_category[cat]["count"] += 1
            by_category[cat]["recall_sum"] += d["recall@k"]
            if d["hit_any"]:
                by_category[cat]["hit_count"] += 1

        category_summary = {}
        for cat, v in by_category.items():
            category_summary[cat] = {
                "count": v["count"],
                "avg_recall": round(v["recall_sum"] / v["count"], 3) if v["count"] else 0,
                "hit_rate": round(v["hit_count"] / v["count"], 3) if v["count"] else 0,
            }

        report = {
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "top_k": top_k,
            "metrics": {
                "avg_recall@k": round(avg_recall, 4),
                "avg_mrr": round(avg_mrr, 4),
                "avg_keyword_hit": round(avg_kw, 4),
                "hit_rate": round(hit_rate, 4),
                "avg_latency_ms": round(avg_latency, 1),
                # 期望文档无法解析的用例数（>0 表示评测集有洞，非指标波动）
                "missing_expectation_count": len(missing_cases),
                # 期望全空、recall 恒为 1.0 的空壳用例数（最危险的假绿）
                "empty_expectation_count": len(empty_expectation_cases),
            },
            "category_summary": category_summary,
            "details": details,
            # 期望解析失败明细：门禁据此报错，避免"用例静默失效、指标照绿"
            "missing_expectation": {
                "cases_with_missing": missing_cases,
                "cases_without_expectation": empty_expectation_cases,
            },
        }

        return report


# ============================================================
# 灰度对比（A/B Test）
# ============================================================
class GrayComparator:
    """
    灰度对比器：对比两套检索配置的效果

    用法：
    1. 基线配置（当前线上）
    2. 实验配置（待上线）
    3. 同一评测集跑两遍，对比指标差异
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def compare(
        self,
        eval_items: List[Dict],
        baseline_config: Dict = None,
        experiment_config: Dict = None,
        top_k: int = 5,
    ) -> Dict:
        """
        执行 A/B 对比评测

        :param eval_items: 评测条目
        :param baseline_config: 基线配置覆盖（如 {"use_rerank": True, "top_k": 5}）
        :param experiment_config: 实验配置覆盖
        :param top_k: 取前 K 条
        :return: 对比报告
        """
        runner = EvalRunner(self.retriever)

        # 基线评测
        logger.info("[Gray] 开始基线评测...")
        baseline_report = runner.run_eval(eval_items, top_k=top_k)

        # 实验评测（如果有配置覆盖，临时修改）
        if experiment_config:
            # 临时应用实验配置
            original_configs = self._apply_config(experiment_config)
            logger.info("[Gray] 开始实验评测...")
            experiment_report = runner.run_eval(eval_items, top_k=top_k)
            # 恢复原配置
            self._restore_config(original_configs)
        else:
            experiment_report = baseline_report  # 无变化

        # 对比
        baseline_metrics = baseline_report["metrics"]
        experiment_metrics = experiment_report["metrics"]

        deltas = {}
        for key in baseline_metrics:
            b = baseline_metrics[key]
            e = experiment_metrics[key]
            delta = e - b
            pct = (delta / b * 100) if b != 0 else 0
            deltas[key] = {
                "baseline": b,
                "experiment": e,
                "delta": round(delta, 4),
                "delta_pct": f"{pct:+.1f}%",
            }

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "total_items": len(eval_items),
            "deltas": deltas,
            "baseline_report": baseline_report,
            "experiment_report": experiment_report,
        }

        return comparison

    def _apply_config(self, config: Dict) -> Dict:
        """临时应用配置覆盖，返回原始配置用于恢复"""
        from retrieval_config import RERANK, VECTOR_RECALL, BM25_RECALL
        original = {
            "RERANK": dict(RERANK),
            "VECTOR_RECALL": dict(VECTOR_RECALL),
            "BM25_RECALL": dict(BM25_RECALL),
        }

        if "use_rerank" in config:
            RERANK["enabled"] = config["use_rerank"]
        if "rerank_top_n" in config:
            RERANK["top_n"] = config["rerank_top_n"]
        if "vector_top_k" in config:
            VECTOR_RECALL["top_k"] = config["vector_top_k"]
        if "bm25_top_k" in config:
            BM25_RECALL["top_k"] = config["bm25_top_k"]

        return original

    def _restore_config(self, original: Dict):
        """恢复原始配置"""
        from retrieval_config import RERANK, VECTOR_RECALL, BM25_RECALL
        RERANK.update(original.get("RERANK", {}))
        VECTOR_RECALL.update(original.get("VECTOR_RECALL", {}))
        BM25_RECALL.update(original.get("BM25_RECALL", {}))


# ============================================================
# 报告持久化
# ============================================================
def save_report(report: Dict, name: str = "eval_report") -> str:
    """保存评测报告到文件"""
    os.makedirs(EVAL_REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EVAL_REPORT_DIR, f"{name}_{ts}.json")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"[Eval] 报告已保存: {path}")
    return path


def list_reports(limit: int = 10) -> List[Dict]:
    """列出最近的评测报告"""
    if not os.path.exists(EVAL_REPORT_DIR):
        return []

    files = sorted(
        [f for f in os.listdir(EVAL_REPORT_DIR) if f.endswith('.json')],
        reverse=True
    )[:limit]

    reports = []
    for f in files:
        path = os.path.join(EVAL_REPORT_DIR, f)
        try:
            with open(path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            reports.append({
                "file": f,
                "timestamp": data.get("timestamp", ""),
                "total": data.get("total", 0),
                "metrics": data.get("metrics", {}),
            })
        except Exception:
            pass

    return reports


def format_report_md(report: Dict) -> str:
    """将评测报告格式化为 Markdown"""
    if not report:
        return "暂无评测数据"

    metrics = report.get("metrics", {})
    md = f"## 评测报告 ({report.get('timestamp', '')[:19]})\n\n"
    md += f"**评测条数**: {report.get('total', 0)} | **Top-K**: {report.get('top_k', 5)}\n\n"
    md += "### 核心指标\n\n"
    md += f"- **Recall@K**: {metrics.get('avg_recall@k', 0):.2%}\n"
    md += f"- **MRR**: {metrics.get('avg_mrr', 0):.4f}\n"
    md += f"- **关键词命中率**: {metrics.get('avg_keyword_hit', 0):.2%}\n"
    md += f"- **命中率 (Hit Any)**: {metrics.get('hit_rate', 0):.2%}\n"
    md += f"- **平均延迟**: {metrics.get('avg_latency_ms', 0):.1f}ms\n\n"

    # 分类统计
    cat_summary = report.get("category_summary", {})
    if cat_summary:
        md += "### 分类统计\n\n"
        md += "| 分类 | 条数 | Recall | 命中率 |\n"
        md += "|------|------|--------|--------|\n"
        for cat, v in cat_summary.items():
            md += f"| {cat} | {v['count']} | {v['avg_recall']:.2%} | {v['hit_rate']:.2%} |\n"

    # 明细（前10条）
    details = report.get("details", [])
    if details:
        md += f"\n### 评测明细（前 {min(10, len(details))} 条）\n\n"
        md += "| ID | 查询 | Recall | MRR | 命中 | 延迟(ms) |\n"
        md += "|-----|------|--------|-----|------|----------|\n"
        for d in details[:10]:
            hit = "✅" if d.get("hit_any") else "❌"
            md += f"| {d['id']} | {d['query'][:20]} | {d['recall@k']:.2%} | {d['mrr']:.3f} | {hit} | {d['latency_ms']:.0f} |\n"

    return md



