# -*- coding: utf-8 -*-
"""
企业级知识库 - 缓存管理模块
对相同/相似问题缓存结果，减少重复 API 调用，降低成本

策略：
- 精确匹配缓存：相同问题直接返回
- TTL 过期机制：默认 24 小时过期
- 容量限制：最多缓存 1000 条，超出淘汰最旧的
"""

import os
import json
import time
from hashlib import md5


# 缓存配置
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'qa_cache.json')
MAX_CACHE_SIZE = 1000
DEFAULT_TTL = 86400  # 24 小时


class CacheManager:
    """问答缓存管理器"""

    def __init__(self, ttl=DEFAULT_TTL, max_size=MAX_CACHE_SIZE, version=""):
        self.ttl = ttl
        self.max_size = max_size
        # 语料指纹：索引重建/文档增删后它变化，旧缓存自动失效。
        # 否则「重建了索引但用户还拿到旧答案」——缓存只按问题文本做 key，
        # 底层数据换了它并不知情。
        self.version = version or ""
        self.cache = {}
        self._load_cache()
        print(f"[CacheManager] 初始化完成，已有 {len(self.cache)} 条缓存，TTL={ttl}s")

    def _load_cache(self):
        """从文件加载缓存"""
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self):
        """保存缓存到文件"""
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CacheManager] 保存失败: {e}")

    def _make_key(self, query, department=""):
        """生成缓存 key（含语料指纹，索引变了旧 key 自然不再命中）"""
        raw = f"{query.strip().lower()}|{department}|{self.version}"
        return md5(raw.encode('utf-8')).hexdigest()

    def get(self, query, department=""):
        """
        查询缓存
        :param query: 用户问题
        :param department: 部门
        :return: 缓存的回答，未命中返回 None
        """
        key = self._make_key(query, department)
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # 检查是否过期
        if time.time() - entry['timestamp'] > self.ttl:
            del self.cache[key]
            self._save_cache()
            return None

        print(f"  [缓存命中] {query[:30]}...")
        entry['hit_count'] += 1
        self._save_cache()
        return entry['answer']

    def set(self, query, answer, department=""):
        """
        写入缓存
        :param query: 用户问题
        :param answer: AI 回答
        :param department: 部门
        """
        key = self._make_key(query, department)

        # 容量淘汰：超出上限时删除最旧的
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'query': query,
            'department': department,
            'answer': answer,
            'timestamp': time.time(),
            'hit_count': 0
        }
        self._save_cache()

    def clear(self):
        """清空缓存"""
        self.cache = {}
        self._save_cache()
        print("[CacheManager] 缓存已清空")

    def get_stats(self):
        """获取缓存统计"""
        total_hits = sum(e['hit_count'] for e in self.cache.values())
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_size,
            'total_hits': total_hits,
            'ttl_hours': self.ttl / 3600
        }
