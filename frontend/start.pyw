"""双击启动 HonorGuide 页面；全部业务操作在浏览器中完成。"""

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import ProxyHandler, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().with_name("app.py")
URL = "http://127.0.0.1:8501"
LOG = Path(tempfile.gettempdir()) / "honorguide-streamlit.log"


def healthy() -> bool:
    try:
        with build_opener(ProxyHandler({})).open(URL + "/_stcore/health", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def find_python() -> Path:
    candidates = []
    registry = Path.home() / ".conda" / "environments.txt"
    if registry.is_file():
        candidates.extend(
            Path(line.strip()) / "python.exe"
            for line in registry.read_text(encoding="utf-8").splitlines()
            if Path(line.strip()).name.lower() == "honorguide"
        )
    candidates.extend([
        Path(sys.prefix) / "envs" / "honorguide" / "python.exe",
        Path(sys.executable),
    ])
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        check = subprocess.run(
            [str(candidate), "-c", "import importlib.util; raise SystemExit(importlib.util.find_spec('streamlit') is None)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=10,
        )
        if check.returncode == 0:
            return candidate
    raise RuntimeError("找不到已安装 Streamlit 的 honorguide 环境，请检查环境与依赖。")


def main() -> None:
    if healthy():
        webbrowser.open(URL)
        return
    python = find_python()
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    with LOG.open("ab") as log:
        process = subprocess.Popen(
            [str(python), "-m", "streamlit", "run", str(APP),
             "--server.address", "127.0.0.1", "--server.port", "8501",
             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    for _ in range(60):
        if process.poll() is not None:
            break
        if healthy():
            webbrowser.open(URL)
            return
        time.sleep(0.5)
    raise RuntimeError(f"页面启动失败，请查看日志：{LOG}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if os.name == "nt":
            ctypes.windll.user32.MessageBoxW(0, str(exc), "HonorGuide 启动失败", 0x10)
        else:
            raise
