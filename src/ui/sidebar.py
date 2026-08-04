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

        provider = st.selectbox(
            "LLM 提供商",
            ["deepseek", "openai"],
            index=0 if config.get("llm_provider", "deepseek") == "deepseek" else 1,
            help="选择大模型提供商"
        )

        if provider == "deepseek":
            default_models = ["deepseek-chat", "deepseek-reasoner"]
            default_base_url = "https://api.deepseek.com"
            api_key_label = "DeepSeek API Key"
        else:
            default_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
            default_base_url = ""
            api_key_label = "OpenAI API Key"

        api_key = st.text_input(
            api_key_label,
            type="password",
            value=config.get("llm_api_key", ""),
            help=f"输入你的 {api_key_label}"
        )

        base_url = st.text_input(
            "API Base URL",
            value=config.get("llm_base_url", default_base_url),
            help="DeepSeek 默认: https://api.deepseek.com"
        )

        model = st.selectbox(
            "模型",
            default_models,
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

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
        "llm_provider": provider,
        "llm_api_key": api_key,
        "llm_base_url": base_url if base_url else None,
        "llm_model": model,
        "temperature": temperature,
    }