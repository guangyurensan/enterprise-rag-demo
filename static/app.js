/* 企业知识库问答前端逻辑
 * 依赖的后端契约（api_server.py）：
 *  - POST /api/auth/login   {username,password} → {token, user}
 *  - GET  /api/auth/me      Bearer 令牌 → 当前身份
 *  - POST /api/chat/stream  SSE: references → token* → done（身份由令牌决定）
 *  - GET  /api/documents    {total, documents:[{doc_id,display_name,version_label,chunk_count,access_level,allowed_groups,uploaded_at}]}
 *  - POST /api/documents    multipart(file, tags) → {status, message, doc_id, ...}
 *  - DELETE /api/documents/{doc_id}
 *  - GET  /api/health       {status, chunks, current_documents, vector_index, degraded_reason}
 *  - GET  /api/metrics      {counters, latency, health_signals, components}
 *  - GET  /api/audit/summary 今日审计统计（仅管理员）
 */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  user: null,         // {user_id, username, display_name, role, groups}
  token: localStorage.getItem("kb_token") || "",
  history: [],        // [{role, content}] 最近若干轮，供查询改写
  streaming: false,
};

const ROLE_LABEL = {
  admin: "系统管理员", manager: "部门经理",
  employee: "普通员工", guest: "访客",
};

/* ---------------- 通用 ---------------- */
function toast(msg, ms = 2600) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), ms);
}

async function api(path, options = {}) {
  // 统一带上登录令牌（登录接口本身无需）
  if (state.token) {
    options.headers = { ...(options.headers || {}), Authorization: "Bearer " + state.token };
  }
  const res = await fetch(path, options);
  if (res.status === 401 && !path.startsWith("/api/auth/login")) {
    // 令牌缺失/过期：清会话回登录页
    doLogout(true);
    throw new Error("登录已过期，请重新登录");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res;
}

/* ---------------- 登录 / 登出（SSO 地基） ---------------- */
function showLogin() {
  $("loginMask").classList.remove("hidden");
  $("userBox").style.display = "none";
  $("loginError").classList.add("hidden");
}

function showUser(user) {
  state.user = user;
  $("loginMask").classList.add("hidden");
  $("userBox").style.display = "flex";
  $("userDisplayName").textContent = user.display_name || user.username;
  $("userRoleLabel").textContent = ROLE_LABEL[user.role] || user.role;
  $("userAvatar").textContent = (user.display_name || user.username || "?").slice(0, 1);
  state.history = [];  // 换身份后历史上下文不可复用（可见范围变了）
}

function doLogout(silent) {
  // 服务端吊销令牌（进黑名单，被偷走的令牌也失效）；失败不阻塞本地登出
  if (state.token) {
    fetch("/api/auth/logout", {
      method: "POST",
      headers: { Authorization: "Bearer " + state.token },
    }).catch(() => {});
  }
  state.token = "";
  state.user = null;
  localStorage.removeItem("kb_token");
  // 整页刷新（2026-10-25 修复：只清状态不清 DOM，换账号登录会看到
  // 上一个账号的聊天记录——刷新后聊天/文档/运维全部归零，无残留）
  location.reload();
}

async function doLogin() {
  const username = $("loginUsername").value.trim();
  const password = $("loginPassword").value;
  const err = $("loginError");
  if (!username || !password) {
    err.textContent = "请输入用户名和密码";
    err.classList.remove("hidden");
    return;
  }
  $("loginBtn").disabled = true;
  try {
    const data = await (await api("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    })).json();
    state.token = data.token;
    localStorage.setItem("kb_token", data.token);
    showUser(data.user);
    toast("欢迎，" + (data.user.display_name || data.user.username));
  } catch (e) {
    err.textContent = e.message || "登录失败";
    err.classList.remove("hidden");
  } finally {
    $("loginBtn").disabled = false;
  }
}

$("loginBtn").addEventListener("click", doLogin);
$("loginPassword").addEventListener("keydown", (e) => {
  if (e.key === "Enter") doLogin();
});
$("loginUsername").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("loginPassword").focus();
});
$("logoutBtn").addEventListener("click", () => doLogout(false));

// 启动时：有令牌则校验并恢复会话，无效则回登录页
(async function boot() {
  if (!state.token) { showLogin(); return; }
  try {
    const me = await (await api("/api/auth/me")).json();
    showUser(me);
  } catch (e) {
    // 401 时 api() 已清令牌并刷新；走到这里多是服务不可达——
    // 只回登录页，不删令牌，避免网络抖动触发刷新死循环
    showLogin();
  }
})();

/* ---------------- 视图切换 ---------------- */
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
    $("view-" + btn.dataset.view).classList.remove("hidden");
    if (btn.dataset.view === "docs") loadDocs();
    if (btn.dataset.view === "ops") loadOps();
  });
});

/* ---------------- 服务状态点 ---------------- */
async function pingHealth() {
  const dot = $("svcDot").querySelector(".dot");
  const txt = $("svcText");
  try {
    const h = await (await api("/api/health")).json();
    if (h.status === "ok" && h.vector_index) {
      dot.className = "dot dot-ok"; txt.textContent = `服务正常 · ${h.current_documents} 文档`;
    } else {
      dot.className = "dot dot-warn"; txt.textContent = h.degraded_reason || "降级运行";
    }
  } catch (_) {
    dot.className = "dot dot-bad"; txt.textContent = "服务不可达";
  }
}
pingHealth();
setInterval(pingHealth, 30000);

/* ---------------- 智能问答 ---------------- */
const chatScroll = $("chatScroll");
const chatInput = $("chatInput");

function addMessage(kind, text) {
  $("chatWelcome") && $("chatWelcome").remove();
  const div = document.createElement("div");
  div.className = "msg " + (kind === "user" ? "msg-user" : "msg-bot");
  div.innerHTML = `
    <div class="msg-avatar">${kind === "user" ? "👤" : "🤖"}</div>
    <div class="msg-content"><div class="msg-bubble"></div></div>`;
  div.querySelector(".msg-bubble").textContent = text;
  chatScroll.appendChild(div);
  chatScroll.scrollTop = chatScroll.scrollHeight;
  return div.querySelector(".msg-bubble");
}

function addReferences(refs) {
  if (!refs || !refs.length) return null;
  const wrap = document.createElement("div");
  wrap.className = "ref-panel";
  wrap.innerHTML = `<div class="ref-title">📎 来源引用（${refs.length} 条，点击展开原文片段）</div>`;
  refs.forEach((r, i) => {
    const page = r.page_number > 0 ? ` · 第${r.page_number}页` : "";
    const card = document.createElement("div");
    card.className = "ref-card";
    card.innerHTML = `
      <div><span class="ref-name">[${i + 1}] ${escapeHtml(r.display_name || r.doc_name)}</span>
      <span class="ref-meta">${escapeHtml(r.parent_title || "")}${page} · 相关度 ${(r.score * 100).toFixed(1)}%</span></div>
      <div class="ref-snippet">${escapeHtml(r.snippet || "")}</div>`;
    card.addEventListener("click", () => card.classList.toggle("open"));
    wrap.appendChild(card);
  });
  chatScroll.appendChild(wrap);
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function sendChat() {
  const q = chatInput.value.trim();
  if (!q || state.streaming) return;
  chatInput.value = "";
  autoGrow();
  addMessage("user", q);
  state.streaming = true;
  $("sendBtn").disabled = true;

  // 机器人气泡（流式占位）
  const bubble = addMessage("bot", "");
  bubble.innerHTML = '<span class="cursor"></span>';
  let acc = "";
  let refs = null;
  const t0 = performance.now();

  try {
    const res = await api("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // 身份由服务端从登录令牌解出——前端不再传 user_role（传了也会被忽略）
      body: JSON.stringify({
        query: q, top_k: 5,
        history: state.history.slice(-6),
      }),
    });
    // 解析 SSE：按空行分帧，每帧 data: {...}
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const frames = buf.split("\n\n");
      buf = frames.pop();
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const evt = JSON.parse(line.slice(6));
        if (evt.type === "references") {
          refs = evt.references;
        } else if (evt.type === "token") {
          acc += evt.content;
          bubble.innerHTML = escapeHtml(acc) + '<span class="cursor"></span>';
          chatScroll.scrollTop = chatScroll.scrollHeight;
        } else if (evt.type === "done") {
          /* 收尾标记 */
        }
      }
    }
    bubble.textContent = acc || "（空回答）";
    addReferences(refs);
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    $("chatFootnote").textContent = `本轮耗时 ${dt}s · 回答由知识库检索生成，请注意核实`;
    state.history.push({ role: "user", content: q }, { role: "assistant", content: acc });
  } catch (e) {
    bubble.textContent = "请求失败：" + e.message;
  } finally {
    state.streaming = false;
    $("sendBtn").disabled = false;
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }
}

$("sendBtn").addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
chatInput.addEventListener("input", autoGrow);
function autoGrow() {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
}
document.querySelectorAll(".sug").forEach((b) =>
  b.addEventListener("click", () => { chatInput.value = b.dataset.q; sendChat(); }));

/* ---------------- 文档管理 ---------------- */
// 密级徽章映射（与 access_control.py 四级一致）
const LEVEL_META = {
  public:       { label: "公开", cls: "lvl-public" },
  internal:     { label: "内部", cls: "lvl-internal" },
  restricted:   { label: "受限", cls: "lvl-restricted" },
  confidential: { label: "机密", cls: "lvl-confidential" },
};

function levelBadge(level) {
  const m = LEVEL_META[level] || LEVEL_META.internal;
  return `<span class="lvl-badge ${m.cls}">${m.label}</span>`;
}

async function loadDocs() {
  const tbody = $("docTbody");
  try {
    const data = await (await api("/api/documents")).json();
    // 密级分布摘要
    const dist = {};
    for (const d of data.documents) {
      const l = d.access_level || "internal";
      dist[l] = (dist[l] || 0) + 1;
    }
    $("docSummary").textContent =
      `共 ${data.total} 份现行文档 · ` +
      (LEVEL_META.internal.label + " " + (dist.internal || 0)) + " · " +
      (LEVEL_META.restricted.label + " " + (dist.restricted || 0)) + " · " +
      (LEVEL_META.confidential.label + " " + (dist.confidential || 0)) + " · " +
      (LEVEL_META.public.label + " " + (dist.public || 0));
    if (!data.documents.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="td-empty">暂无文档</td></tr>';
      return;
    }
    tbody.innerHTML = "";
    for (const d of data.documents) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="doc-name">${escapeHtml(d.display_name || d.filename)}</td>
        <td>${d.version_label ? `<span class="ver-tag">${escapeHtml(d.version_label)}</span>` : "—"}</td>
        <td>${levelBadge(d.access_level)}${d.allowed_groups && d.allowed_groups !== "all" ? `<span class="grp-tag">${escapeHtml(d.allowed_groups)}</span>` : ""}</td>
        <td>${d.chunk_count}</td>
        <td>${(d.uploaded_at || "").slice(0, 16).replace("T", " ")}</td>
        <td><button class="btn btn-sm btn-danger" data-id="${d.doc_id}" data-name="${escapeHtml(d.display_name)}">删除</button></td>`;
      tr.querySelector("button").addEventListener("click", () => deleteDoc(d.doc_id, d.display_name || d.filename));
      tbody.appendChild(tr);
    }
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="6" class="td-empty">加载失败：${escapeHtml(e.message)}</td></tr>`;
  }
}

async function deleteDoc(docId, name) {
  if (!confirm(`确认删除「${name}」？\n\n该文档的向量与关键词索引将同步移除，检索立即不再命中。`)) return;
  try {
    await api(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
    toast("已删除：" + name);
    loadDocs();
  } catch (e) {
    toast("删除失败：" + e.message);
  }
}

$("uploadBtn").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", uploadFile);
$("docRefreshBtn").addEventListener("click", loadDocs);

async function uploadFile() {
  const file = $("fileInput").files[0];
  if (!file) return;
  const tags = $("tagsInput").value.trim();
  const notice = $("uploadNotice");
  notice.className = "upload-notice warn";
  notice.textContent = `正在上传并解析「${file.name}」，文档解析可能需要一分钟左右，请勿关闭页面…`;
  notice.classList.remove("hidden");
  try {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("tags", tags);
    const data = await (await api("/api/documents", { method: "POST", body: fd })).json();
    const cls = data.status === "superseded" ? "bad" : "ok";
    notice.className = "upload-notice " + cls;
    notice.textContent = data.message || "上传成功";
    if (data.status !== "superseded") toast("入库成功：" + file.name);
    loadDocs();
  } catch (e) {
    notice.className = "upload-notice bad";
    notice.textContent = "上传失败：" + e.message;
  } finally {
    $("fileInput").value = "";
  }
}

/* ---------------- 运维状态 ---------------- */
function pill(status, okText, warnText, badText) {
  const map = { ok: ["ok", okText], unknown: ["warn", warnText], degraded: ["bad", badText] };
  const [cls, text] = map[status] || ["warn", status];
  return `<span class="status-pill ${cls}">● ${text}</span>`;
}

async function loadOps() {
  try {
    const h = await (await api("/api/health")).json();
    $("healthBody").innerHTML = `
      <div class="kv"><span>服务状态</span><b>${h.status === "ok" ? "正常" : "异常"}</b></div>
      <div class="kv"><span>现行文档</span><b>${h.current_documents}</b></div>
      <div class="kv"><span>知识块 / 向量</span><b>${h.chunks}</b></div>
      <div class="kv"><span>向量索引</span><b>${h.vector_index ? "已构建" : "未构建"}</b></div>
      ${h.degraded_reason ? `<div class="kv"><span>降级原因</span><b>${escapeHtml(h.degraded_reason)}</b></div>` : ""}`;

    const m = await (await api("/api/metrics")).json();
    const hs = m.health_signals || {};
    $("vectorBody").innerHTML = `
      <div class="kv"><span>通道状态</span><b>${pill(hs.vector_channel_status, "正常", "样本不足", "已降级")}</b></div>
      <div class="kv"><span>向量召回占比</span><b>${hs.vector_recall_ratio == null ? "—" : (hs.vector_recall_ratio * 100).toFixed(1) + "%"}</b></div>
      <div class="kv"><span>通道故障次数</span><b>${hs.vector_recall_errors ?? 0}</b></div>
      <div style="font-size:12px;color:#9ca3af;margin-top:6px">${escapeHtml(hs.note || "")}</div>`;

    const c = m.counters || {};
    const counterKeys = ["search_total", "search_empty", "search_bm25_only", "llm_success", "llm_error", "llm_timeout", "vector_recall_error"];
    $("counterBody").innerHTML = `<div class="counter-grid">` + counterKeys.map((k) =>
      `<div class="counter-item"><div class="cv">${c[k] ?? 0}</div><div class="ck">${k}</div></div>`).join("") + `</div>`;

    const lat = m.latency || {};
    const latRows = Object.entries(lat).map(([k, v]) =>
      `<div class="kv"><span>${k}</span><b>avg ${v.avg_ms}ms · p95 ${v.p95_ms}ms · max ${v.max_ms}ms（${v.count} 次）</b></div>`).join("");
    $("latencyBody").innerHTML = latRows || "暂无样本——发起一次问答后回来看";

    // 审计概览：今日事件统计 + 敏感命中明细（谁查过涉密内容）
    try {
      const a = await (await api("/api/audit/summary")).json();
      const s = a.stats || {};
      const evDesc = Object.entries(s.by_event || {})
        .map(([k, v]) => `${k === "search" ? "检索" : k === "answer" ? "问答" : k === "document_op" ? "文档操作" : k} ${v}`).join(" · ") || "暂无";
      const sensitiveRows = (a.recent_sensitive || []).map((e) =>
        `<div class="kv"><span>${escapeHtml(e.time.replace("T", " "))} · ${escapeHtml(e.role)}</span><b>${escapeHtml(e.query.slice(0, 30))}（${e.hit_levels.join("/")}）</b></div>`).join("");
      $("auditBody").innerHTML = `
        <div class="kv"><span>事件总数</span><b>${s.total ?? 0}（${evDesc}）</b></div>
        <div class="kv"><span>涉密命中</span><b>${pill((s.sensitive_hits ?? 0) > 0 ? "degraded" : "ok", "无", "无", `${s.sensitive_hits ?? 0} 次`)}</b></div>
        <div class="kv"><span>拒答次数</span><b>${s.refused ?? 0}</b></div>
        ${sensitiveRows
          ? `<div style="font-size:12px;color:#9ca3af;margin:8px 0 4px">最近涉密查询（最多 10 条）</div>${sensitiveRows}`
          : `<div style="font-size:12px;color:#9ca3af;margin-top:6px">今日无涉密文档命中记录</div>`}`;
    } catch (e2) {
      $("auditBody").innerHTML =
        `<div style="font-size:12px;color:#9ca3af">${escapeHtml(e2.message)}</div>`;
    }
  } catch (e) {
    $("healthBody").textContent = "加载失败：" + e.message;
  }
}
setInterval(() => {
  if (!$("view-ops").classList.contains("hidden")) loadOps();
}, 15000);
