import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Optional
from src.ui.components import (
    render_progress_bar,
    render_stage_indicator,
    render_deliverables,
    render_quality_gates,
)


FILE_ICONS = {
    ".pdf": "📕",
    ".xlsx": "📊",
    ".xls": "📊",
    ".csv": "📄",
    ".txt": "📝",
    ".docx": "📘",
    ".doc": "📘",
    ".png": "🖼️",
    ".jpg": "🖼️",
    ".jpeg": "🖼️",
}


def _get_file_icon(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return FILE_ICONS.get(ext, "📎")


def _get_file_size_str(path: str) -> str:
    try:
        size = os.path.getsize(path)
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"
    except Exception:
        return "未知"


def render_step_upload():
    uploaded_files = st.file_uploader(
        "上传题目 PDF、Excel 附件等",
        accept_multiple_files=True,
        type=["pdf", "xlsx", "xls", "csv", "txt", "docx", "png", "jpg", "jpeg"],
        key="step_uploader",
        label_visibility="collapsed",
    )
    if uploaded_files:
        data_dir = st.session_state.config.project_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            tmp_path = data_dir / uf.name
            tmp_path.write_bytes(uf.getvalue())
            abs_path = str(tmp_path.resolve())
            if abs_path not in st.session_state.uploaded_file_paths:
                st.session_state.uploaded_file_paths.append(abs_path)
        st.session_state.workflow_state["problem_files"] = st.session_state.uploaded_file_paths

    if not st.session_state.uploaded_file_paths:
        st.info("👆 请上传题目文件（PDF / Excel / 文本等），支持多文件")
        return uploaded_files

    st.success(f"✅ 已上传 {len(st.session_state.uploaded_file_paths)} 个文件")

    for fp in st.session_state.uploaded_file_paths:
        fname = Path(fp).name
        icon = _get_file_icon(fname)
        size_str = _get_file_size_str(fp)
        col1, col2, col3 = st.columns([0.4, 2.6, 1])
        with col1:
            st.markdown(f"### {icon}")
        with col2:
            st.markdown(f"**{fname}**")
            st.caption(f"📏 {size_str}  ·  `{fp}`")
        with col3:
            if st.button("🗑️", key=f"del_{fp}", help=f"移除 {fname}"):
                st.session_state.uploaded_file_paths.remove(fp)
                st.session_state.workflow_state["problem_files"] = st.session_state.uploaded_file_paths
                st.rerun()

    col_clear, _ = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ 清空全部附件", key="clear_all_files"):
            st.session_state.uploaded_file_paths = []
            st.session_state.workflow_state["problem_files"] = []
            st.rerun()

    st.markdown("---")

    with st.expander("📖 预览附件内容"):
        for fp in st.session_state.uploaded_file_paths:
            fname = Path(fp).name
            ext = Path(fp).suffix.lower()
            st.markdown(f"**{_get_file_icon(fname)} {fname}**")
            try:
                if ext == ".pdf":
                    _preview_pdf(fp)
                elif ext in (".xlsx", ".xls", ".csv"):
                    _preview_excel(fp)
                elif ext in (".txt", ".md", ".py"):
                    _preview_text(fp)
                else:
                    st.caption(f"（{ext} 文件不支持预览）")
            except Exception as e:
                st.caption(f"预览失败: {e}")
            st.markdown("---")

    return uploaded_files


def _preview_pdf(file_path: str):
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        st.caption(f"共 {len(reader.pages)} 页，预览前 3 页")
        text_parts = []
        for i, page in enumerate(reader.pages[:3]):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                text_parts.append(f"**第 {i+1} 页**\n{page_text[:1500]}")
        if text_parts:
            st.text_area(
                "PDF 内容预览",
                "\n\n".join(text_parts),
                height=250,
                disabled=True,
                key=f"pdf_preview_{hash(file_path)}",
                label_visibility="collapsed",
            )
        else:
            st.caption("（无法提取文字，可能是扫描版 PDF）")
    except ImportError:
        st.caption("（PyPDF2 未安装）")
    except Exception as e:
        st.caption(f"（读取失败: {e}）")


def _preview_excel(file_path: str):
    try:
        import pandas as pd
        xls = pd.ExcelFile(file_path)
        st.caption(f"工作表: {', '.join(xls.sheet_names)}")
        for sheet in xls.sheet_names[:3]:
            df = pd.read_excel(file_path, sheet_name=sheet)
            st.markdown(f"**{sheet}** ({df.shape[0]} 行 × {df.shape[1]} 列)")
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    except ImportError:
        st.caption("（pandas/openpyxl 未安装）")
    except Exception as e:
        st.caption(f"（读取失败: {e}）")


def _preview_text(file_path: str):
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        st.text_area(
            "文本内容",
            content[:3000],
            height=200,
            disabled=True,
            key=f"txt_preview_{hash(file_path)}",
            label_visibility="collapsed",
        )
    except Exception as e:
        st.caption(f"（读取失败: {e}）")


def render_step_problem():
    problem_text = st.text_area(
        "输入题目描述",
        height=180,
        placeholder="在此粘贴题目内容，或描述你的问题...",
        key="step_problem_input",
        label_visibility="collapsed",
    )
    return problem_text


def render_step_mode():
    st.markdown("#### 工作模式")

    mode = st.radio(
        "选择处理方式",
        ["🔄 完整流程", "🧠 仅建模分析", "💻 仅编程实现", "📝 仅撰写论文"],
        index=0,
        key="mode_radio",
        label_visibility="collapsed",
    )

    st.markdown("#### 配置选项")

    col1, col2 = st.columns(2)
    with col1:
        competition = st.selectbox(
            "竞赛类型",
            ["cumcm", "mcm-icm", "other"],
            index=0,
            key="mode_competition",
        )
    with col2:
        language = st.selectbox(
            "语言",
            ["chinese", "english"],
            index=0,
            key="mode_language",
        )

    st.session_state.config.competition = competition
    st.session_state.config.language = language
    st.session_state.workflow_state["competition"] = competition
    st.session_state.workflow_state["language"] = language

    with st.expander("⚙️ 高级选项（Subagent）"):
        subagents = {}
        subagents["attachment_inventory"] = st.checkbox(
            "附件盘点", value=False,
            help="多附件、复杂PDF/Excel时启用"
        )
        subagents["literature_survey"] = st.checkbox(
            "文献调研", value=False,
            help="按主张簇或模型族搜索文献"
        )
        subagents["algorithm_prototype"] = st.checkbox(
            "算法原型", value=False,
            help="验证候选算法的可实现性"
        )
        subagents["independent_experiment"] = st.checkbox(
            "独立实验", value=False,
            help="并行敏感性、消融或边界实验"
        )
        subagents["dual_language"] = st.checkbox(
            "Python/MATLAB对照", value=False,
            help="双语言实现并验证一致性"
        )
        subagents["terminology_check"] = st.checkbox(
            "术语核验", value=False,
            help="专业术语和英文表达核验"
        )
        st.session_state.config.subagents = subagents
        st.session_state.workflow_state["subagent_config"] = subagents

    trigger = False
    trigger_chat = False
    chat_input = None

    col_btn, col_chat = st.columns([1, 2])
    with col_btn:
        if st.button("▶️ 开始执行", type="primary", use_container_width=True):
            trigger = True

    with col_chat:
        chat_input = st.chat_input("或直接输入问题与 AI 对话...", key="mode_chat")
        if chat_input:
            trigger_chat = True
            st.session_state.messages.append({"role": "user", "content": chat_input})

    mode_map = {
        "🔄 完整流程": "full",
        "🧠 仅建模分析": "modeling",
        "💻 仅编程实现": "coding",
        "📝 仅撰写论文": "writing",
    }

    return {
        "trigger": trigger,
        "mode": mode_map.get(mode, "full"),
        "trigger_chat": trigger_chat,
        "chat_input": chat_input,
        "competition": competition,
        "language": language,
        "subagents": subagents,
    }


def render_results(state: dict, run_log: list):
    if not run_log and not state.get("modeling_report") and not state.get("stage_output"):
        return

    st.markdown("### 📊 执行结果")

    tab1, tab2, tab3 = st.tabs(["📋 运行日志", "📦 交付物", "💬 对话历史"])

    with tab1:
        if run_log:
            for entry in run_log:
                st.text(entry)
        else:
            st.info("暂无运行日志")

        current_stage = state.get("current_stage", "init")
        render_stage_indicator(current_stage)
        render_progress_bar(current_stage, state.get("quality_gates", {}))
        render_quality_gates(state.get("quality_gates", {}))

    with tab2:
        render_deliverables(state)

    with tab3:
        if st.session_state.get("messages"):
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    content = msg["content"]
                    if len(content) > 3000:
                        with st.expander("查看完整内容"):
                            st.markdown(content)
                    else:
                        st.markdown(content)
        else:
            st.info("暂无对话记录")


def render_workflow_panel(state: dict, config: dict):
    st.markdown("### 🔄 工作流状态")
    render_stage_indicator(state.get("current_stage", "init"))
    render_progress_bar(
        state.get("current_stage", "init"),
        state.get("quality_gates", {})
    )
    render_quality_gates(state.get("quality_gates", {}))
    render_deliverables(state)


def render_config_panel():
    st.markdown("### ⚙️ 配置")

    competition = st.selectbox(
        "竞赛类型",
        ["cumcm", "mcm-icm", "other"],
        index=0,
        help="CUMCM=国赛, MCM/ICM=美赛"
    )

    language = st.selectbox(
        "语言",
        ["chinese", "english"],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 🤝 可选 Subagent")

    subagents = {}
    subagents["attachment_inventory"] = st.checkbox(
        "附件盘点", value=False,
        help="多附件、复杂PDF/Excel时启用"
    )
    subagents["literature_survey"] = st.checkbox(
        "文献调研", value=False,
        help="按主张簇或模型族搜索文献"
    )
    subagents["algorithm_prototype"] = st.checkbox(
        "算法原型", value=False,
        help="验证候选算法的可实现性"
    )
    subagents["independent_experiment"] = st.checkbox(
        "独立实验", value=False,
        help="并行敏感性、消融或边界实验"
    )
    subagents["dual_language"] = st.checkbox(
        "Python/MATLAB对照", value=False,
        help="双语言实现并验证一致性"
    )
    subagents["terminology_check"] = st.checkbox(
        "术语核验", value=False,
        help="专业术语和英文表达核验"
    )

    return {
        "competition": competition,
        "language": language,
        "subagents": subagents,
    }


def render_file_uploader():
    st.markdown("### 📁 上传题目文件")
    uploaded_files = st.file_uploader(
        "上传题目PDF、Excel附件等",
        accept_multiple_files=True,
        type=["pdf", "xlsx", "xls", "csv", "txt", "docx"],
        key="file_uploader"
    )
    return uploaded_files


def render_problem_input():
    st.markdown("### 📝 题目描述")
    problem_text = st.text_area(
        "输入题目描述",
        height=150,
        placeholder="在此粘贴题目内容，或描述你的问题...",
        key="problem_input"
    )
    return problem_text


def render_chat_interface():
    st.markdown("### 💬 对话")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("输入你的问题或指令..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        return prompt
    return None