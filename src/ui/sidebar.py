import streamlit as st
from typing import Optional


def render_sidebar(config: dict) -> dict:
    with st.sidebar:
        st.markdown("## 🎯 数学建模 Agent")

        if st.button("⏹️ 停止思考", type="primary", use_container_width=True, key="stop_btn"):
            st.session_state.stop_requested = True
            st.rerun()

        st.markdown("---")

        st.markdown("### 🔑 API 配置")

        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=config.get("llm_api_key", ""),
            help="输入你的 DeepSeek API Key"
        )

        base_url = st.text_input(
            "API Base URL",
            value=config.get("llm_base_url", "https://api.deepseek.com"),
            help="默认: https://api.deepseek.com，可改为代理地址"
        )

        st.markdown("---")

        st.markdown("### 🧠 模型策略")
        st.caption("生成/质检：`deepseek-chat`（Temperature 0.1）")
        st.caption("修复/修改：`deepseek-reasoner`（Temperature 0.0）")
        st.caption("系统自动切换，无需手动配置")

        st.markdown("---")

        if st.button("🗑️ 清空全部", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_file_paths = []
            st.session_state.workflow_state["problem_description"] = ""
            st.session_state.workflow_state["problem_files"] = []
            st.session_state.workflow_state["modeling_report"] = ""
            st.session_state.workflow_state["stage_output"] = ""
            st.session_state.run_log = []
            st.rerun()

        st.markdown("---")

        st.markdown("### 📖 关于")
        st.markdown("""
        基于 [math-modeling-skill](https://github.com/XiaoMaColtAI/math-modeling-skill) 构建，
        使用 LangChain + LangGraph + Streamlit 实现。
        """)

        st.markdown(f"**Skill 版本**: 1.1.1")

    return {
        "llm_api_key": api_key,
        "llm_base_url": base_url if base_url else None,
    }