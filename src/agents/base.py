import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from src.config import AppConfig
from src.tools.skill_loader import SkillLoader


class StopRequested(Exception):
    pass


class BaseAgent:
    def __init__(self, config: AppConfig, role_name: str):
        self.config = config
        self.role_name = role_name
        self.skill_loader = SkillLoader(config.skill_root)
        self._llm: Optional[ChatOpenAI] = None
        self._system_prompt: Optional[str] = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=True,
            )
        return self._llm

    def load_system_prompt(self) -> str:
        raise NotImplementedError

    def get_tools(self) -> List[BaseTool]:
        return []

    def invoke(
        self,
        messages: List[BaseMessage],
        user_input: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        if system_prompt is not None:
            self._system_prompt = system_prompt
        elif self._system_prompt is None:
            self._system_prompt = self.load_system_prompt()

        full_messages = [SystemMessage(content=self._system_prompt)]
        full_messages.extend(messages)

        if user_input:
            full_messages.append(HumanMessage(content=user_input))

        full_response = []
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"  [{self.role_name}] LLM 思考中...", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)

        chunk_count = 0
        for chunk in self.llm.stream(full_messages):
            token = chunk.content
            if token:
                print(token, end="", file=sys.stderr, flush=True)
                full_response.append(token)
                chunk_count += 1
                if chunk_count % 20 == 0 and self._check_stop():
                    print(f"\n[用户中断]", file=sys.stderr, flush=True)
                    raise StopRequested("用户中断了思考")

        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"  [{self.role_name}] 完成\n", file=sys.stderr, flush=True)

        return "".join(full_response)

    @staticmethod
    def _check_stop() -> bool:
        try:
            import streamlit as st
        except ImportError:
            return False
        try:
            return st.session_state.get("stop_requested", False)
        except Exception:
            return False

    def _read_file(self, path: Path) -> str:
        if not path.exists():
            return f"[文件不存在: {path}]"
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(path)
        elif suffix in (".xlsx", ".xls"):
            return self._read_excel(path)
        elif suffix in (".csv", ".txt", ".md", ".py", ".json", ".xml", ".yaml", ".yml"):
            return path.read_text(encoding="utf-8", errors="ignore")
        else:
            try:
                return path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return f"[无法读取二进制文件: {path.name}，大小: {path.stat().st_size} bytes]"

    def _read_pdf(self, path: Path) -> str:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            parts = []
            for i, page in enumerate(reader.pages[:10]):
                text = (page.extract_text() or "").strip()
                if text:
                    parts.append(f"--- 第{i+1}页 ---\n{text}")
            return "\n\n".join(parts) if parts else "[PDF 无法提取文本，可能是扫描版]"
        except ImportError:
            return f"[PyPDF2 未安装，无法读取 PDF: {path.name}]"
        except Exception as e:
            return f"[读取 PDF 失败: {e}]"

    def _read_excel(self, path: Path) -> str:
        try:
            import pandas as pd
            xls = pd.ExcelFile(str(path))
            parts = [f"Excel 文件: {path.name}，工作表: {', '.join(xls.sheet_names)}"]
            for sheet in xls.sheet_names[:5]:
                df = pd.read_excel(str(path), sheet_name=sheet)
                parts.append(f"\n--- 工作表: {sheet} ({df.shape[0]}行 × {df.shape[1]}列) ---\n{df.head(50).to_string()}")
            return "\n".join(parts)
        except ImportError:
            return f"[pandas 未安装，无法读取 Excel: {path.name}]"
        except Exception as e:
            return f"[读取 Excel 失败: {e}]"

    def _load_role_skill(self, role_path: str) -> str:
        return self.skill_loader.load_skill(role_path)

    def _load_reference(self, ref_path: str) -> str:
        return self.skill_loader.load_reference(ref_path)

    def _load_algorithm(self, algo_name: str) -> str:
        return self.skill_loader.load_algorithm(algo_name)