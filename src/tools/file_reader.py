from pathlib import Path
from typing import Optional
from langchain_core.tools import BaseTool, tool


class FileReaderTool:
    def __init__(self, skill_root: Path):
        self.skill_root = Path(skill_root)

    @property
    def read_pdf_tool(self):
        @tool
        def read_pdf(file_path: str) -> str:
            """读取 PDF 文件内容。file_path 为文件路径"""
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages[:10]:
                    text += page.extract_text() or ""
                return text[:5000] if text else "[PDF 无法提取文本]"
            except ImportError:
                return "[PyPDF2 未安装，无法读取 PDF]"
            except Exception as e:
                return f"[读取 PDF 失败: {e}]"
        return read_pdf

    @property
    def read_excel_tool(self):
        @tool
        def read_excel(file_path: str, sheet_name: str = "") -> str:
            """读取 Excel 文件内容。file_path 为文件路径，sheet_name 可选指定工作表"""
            try:
                import pandas as pd
                if sheet_name:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                else:
                    df = pd.read_excel(file_path)
                return df.head(50).to_string()
            except ImportError:
                return "[pandas/openpyxl 未安装，无法读取 Excel]"
            except Exception as e:
                return f"[读取 Excel 失败: {e}]"
        return read_excel

    @property
    def read_text_tool(self):
        @tool
        def read_text(file_path: str) -> str:
            """读取文本文件内容。file_path 为文件路径"""
            try:
                return Path(file_path).read_text(encoding="utf-8", errors="ignore")[:5000]
            except Exception as e:
                return f"[读取文件失败: {e}]"
        return read_text