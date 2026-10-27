# -*- coding: utf-8 -*-
"""
服务看护进程（最小服务化方案）

用法:
    python serve.py --port 8765
    （可注册到 Windows 任务计划程序，实现开机自启）

职责:
- 拉起 api_server.py 子进程
- 每 POLL_INTERVAL 秒轮询 /api/health
- 连续 FAIL_THRESHOLD 次失败（或子进程退出）→ 杀掉重启，指数退避
- 重启记录写入 logs/service_watchdog.log

不做的事（有意留白）:
- 不做多实例/负载均衡——单机场景先解决"挂了没人管"
- 不做指标采集轮询——/api/metrics 已就绪，接 Prometheus 时再说
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

POLL_INTERVAL = 15        # 健康轮询间隔（秒）
WARMUP_SEC = 90           # 启动宽限期：加载模型/Milvus 连接需要时间
FAIL_THRESHOLD = 3        # 连续失败 N 次判定为不可用
MAX_BACKOFF = 300         # 重启退避上限（秒）


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "service_watchdog.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/health", timeout=10
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_server(port: int) -> tuple:
    cmd = [sys.executable, os.path.join(BASE_DIR, "api_server.py"), "--port", str(port)]
    # 子进程 stdout/stderr 追加落盘：真实崩溃时 traceback 可回溯
    # （服务自身日志仅输出到 stdout，若直通看护终端，崩溃现场会随终端关闭丢失）
    log_path = os.path.join(LOG_DIR, "service.log")
    fh = open(log_path, "a", encoding="utf-8", buffering=1)  # 行缓冲
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fh.write(f"\n===== 子进程启动 {ts} port={port} =====\n")
    proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=fh, stderr=subprocess.STDOUT)
    log(f"服务子进程已启动 pid={proc.pid} port={port} (输出→logs/service.log)")
    return proc, fh


def stop_server(proc: subprocess.Popen):
    if proc.poll() is None:
        log(f"终止子进程 pid={proc.pid}")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            log("子进程未响应 terminate，已 kill")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    port = args.port

    log(f"看护进程启动，目标端口 {port}")
    backoff = 5
    while True:
        proc, child_log = start_server(port)
        fails = 0
        started = time.time()

        # 监控循环：子进程活着就轮询健康，死了立即进入重启
        while proc.poll() is None:
            # 宽限期内不判失败（模型加载可能要一两分钟）
            if time.time() - started < WARMUP_SEC:
                time.sleep(POLL_INTERVAL)
                continue
            if health_ok(port):
                if fails:
                    log(f"健康恢复（此前连续失败 {fails} 次）")
                fails = 0
                backoff = 5  # 恢复后重置退避
            else:
                fails += 1
                log(f"健康检查失败 {fails}/{FAIL_THRESHOLD}")
                if fails >= FAIL_THRESHOLD:
                    log("判定服务不可用，重启")
                    stop_server(proc)
                    break
            time.sleep(POLL_INTERVAL)

        # 走到这里：子进程退出 或 健康失败重启
        exit_code = proc.poll()
        if exit_code is not None:
            log(f"子进程退出，exit code={exit_code}（详情见 logs/service.log）")
        child_log.close()
        log(f"{backoff}s 后重启（指数退避）")
        time.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("看护进程手动停止")
