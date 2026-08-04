import streamlit as st
from pathlib import Path
from typing import Dict, Optional


def render_progress_bar(current_stage: str, quality_gates: Dict[str, str]):
    stages = [
        ("M1", "建模分析"),
        ("P1", "最小可运行"),
        ("P2", "编程实现"),
        ("W1", "证据大纲"),
        ("W2", "论文撰写"),
    ]

    stage_to_idx = {
        "init": 0,
        "modeling": 0,
        "m1_check": 0,
        "coding_p1": 1,
        "p1_check": 1,
        "coding_full": 2,
        "p2_check": 2,
        "writing_w1": 3,
        "w1_check": 3,
        "writing_full": 4,
        "w2_check": 4,
        "done": 5,
    }

    current_idx = stage_to_idx.get(current_stage, 0)

    cols = st.columns(5)
    for i, (gate, label) in enumerate(stages):
        with cols[i]:
            status = quality_gates.get(gate, "pending")
            if i < current_idx:
                if status == "PASS":
                    st.markdown(f"✅ **{label}**")
                elif status == "FAIL":
                    st.markdown(f"❌ **{label}**")
                else:
                    st.markdown(f"🟡 **{label}**")
            elif i == current_idx:
                st.markdown(f"🔄 **{label}**")
            else:
                st.markdown(f"⬜ {label}")


def render_stage_indicator(current_stage: str):
    stage_names = {
        "init": "🚀 初始化",
        "modeling": "🧠 建模分析中...",
        "m1_check": "🔍 M1 建模终检中...",
        "coding_p1": "💻 最小实现中...",
        "p1_check": "🔍 P1 最小可运行门禁中...",
        "coding_full": "💻 全量实现中...",
        "p2_check": "🔍 P2 编程终检中...",
        "writing_w1": "📝 证据大纲构建中...",
        "w1_check": "🔍 W1 证据大纲门禁中...",
        "writing_full": "📝 论文撰写中...",
        "w2_check": "🔍 W2 论文终检中...",
        "done": "✅ 完成",
        "error": "❌ 出错",
    }
    name = stage_names.get(current_stage, current_stage)
    st.info(f"**当前阶段：{name}**")


def render_deliverables(state: dict):
    if not state.get("modeling_report") and not state.get("code_files"):
        return

    st.markdown("---")
    st.markdown("### 📦 交付物")

    if state.get("modeling_report"):
        with st.expander("📄 题目分析报告", expanded=False):
            st.markdown(state["modeling_report"][:2000])

    if state.get("terminology_table"):
        with st.expander("📊 术语表格", expanded=False):
            st.markdown(state["terminology_table"][:2000])

    if state.get("code_files"):
        with st.expander("💻 代码文件", expanded=False):
            for fp in state["code_files"]:
                fpath = Path(fp) if isinstance(fp, str) else Path(str(fp))
                try:
                    code_content = fpath.read_text(encoding="utf-8", errors="ignore")
                    st.code(code_content[:2000], language="python")
                except Exception:
                    st.code(str(fp), language="text")

    if state.get("figure_files"):
        st.markdown(f"📈 图表：{len(state['figure_files'])} 张")

    if state.get("word_paper"):
        st.markdown(f"📝 Word 论文：已生成")

    if state.get("pdf_paper"):
        st.markdown(f"📕 PDF 论文：已生成")

    _render_download_buttons()


def _render_download_buttons():
    output_files = st.session_state.get("output_files", {})
    if not output_files:
        return

    st.markdown("---")
    st.markdown("### 📥 下载")

    labels = [
        ("report", "📄 题目分析报告", "题目分析报告.md"),
        ("terminology", "📊 术语表格", "术语表格.md"),
        ("code_p1", "💻 求解代码 (P1)", "solution_p1.py"),
        ("code_full", "💻 求解代码 (完整)", "solution_full.py"),
        ("exec_output", "🖥️ 代码执行结果", "代码执行结果.txt"),
        ("outline", "📋 证据大纲", "证据大纲.md"),
        ("paper", "📝 完整论文", "完整论文.md"),
    ]

    for key, label, filename in labels:
        path = output_files.get(key)
        if path:
            try:
                data = open(path, "rb").read()
                st.download_button(
                    label=label,
                    data=data,
                    file_name=filename,
                    mime="text/markdown" if filename.endswith(".md") else "text/x-python",
                    key=f"dl_{key}",
                )
            except FileNotFoundError:
                pass


def render_quality_gates(quality_gates: Dict[str, str]):
    if not quality_gates:
        return
    st.markdown("---")
    st.markdown("### 🛡️ 质量门禁")
    for gate, status in quality_gates.items():
        if status == "PASS":
            st.markdown(f"- {gate}: ✅ 通过")
        elif status == "FAIL":
            st.markdown(f"- {gate}: ❌ 未通过")
        elif status == "BLOCKED":
            st.markdown(f"- {gate}: 🚫 阻塞")
        else:
            st.markdown(f"- {gate}: ⏳ 待检查")