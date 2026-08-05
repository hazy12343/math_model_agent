import subprocess
import sys
from pathlib import Path
from langchain_core.tools import tool


class PaperSearchTool:
    def __init__(self, skill_root: Path):
        self.skill_root = Path(skill_root)
        self._script = self.skill_root / "tools/paper_search/scripts/hybrid_scholar.py"

    @property
    def search_tool(self):
        @tool
        def search_papers(query: str, limit: int = 5) -> str:
            """搜索学术论文。query 为搜索关键词，limit 为返回数量（默认5）"""
            if not self._script.exists():
                return "[论文搜索脚本不存在]"
            try:
                result = subprocess.run(
                    [sys.executable, str(self._script), "--query", query, "--limit", str(limit), "--json"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(self._script.parent)
                )
                output = result.stdout
                if len(output) > 5000:
                    output = output[:5000] + "\n...(结果已截断)"
                return output or f"[搜索无结果] stderr: {result.stderr}"
            except subprocess.TimeoutExpired:
                return "[搜索超时]"
            except Exception as e:
                return f"[搜索失败: {e}]"
        return search_papers

    @property
    def search_openalex_tool(self):
        @tool
        def search_openalex(query: str, limit: int = 5) -> str:
            """仅使用 OpenAlex 搜索学术论文"""
            if not self._script.exists():
                return "[论文搜索脚本不存在]"
            try:
                result = subprocess.run(
                    [sys.executable, str(self._script), "--query", query, "--limit", str(limit), "--openalex-only", "--json"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(self._script.parent)
                )
                return result.stdout[:5000] or f"[搜索无结果]"
            except Exception as e:
                return f"[搜索失败: {e}]"
        return search_openalex