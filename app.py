import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.config import AppConfig
from src.state import WorkflowState
from src.graph import create_workflow, create_single_stage_workflow, _extract_code_from_output
from src.agents.base import StopRequested
from src.ui.sidebar import render_sidebar
from src.ui.chat import (
    render_step_upload,
    render_step_problem,
    render_step_mode,
    render_results,
)


st.set_page_config(
    page_title="数学建模 Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎯 数学建模智能助手")
st.caption("基于 LangChain + LangGraph + Streamlit | 三步完成建模 → 编程 → 论文")


if "config" not in st.session_state:
    st.session_state.config = AppConfig.from_env()

if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = {
        "current_stage": "init",
        "stage_history": [],
        "quality_gates": {},
        "project_root": str(st.session_state.config.project_root),
        "skill_root": str(st.session_state.config.skill_root),
        "competition": "cumcm",
        "language": "chinese",
        "problem_description": "",
        "problem_files": [],
        "code_files": [],
        "result_files": [],
        "figure_files": [],
        "subagent_config": {},
        "error": None,
        "retry_counts": {},
        "stage_output": "",
        "modeling_report": "",
        "terminology_table": "",
        "word_paper": "",
        "pdf_paper": "",
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_paths" not in st.session_state:
    st.session_state.uploaded_file_paths = []

if "run_log" not in st.session_state:
    st.session_state.run_log = []

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


sidebar_config = render_sidebar({
    "llm_provider": st.session_state.config.llm_provider,
    "llm_api_key": st.session_state.config.llm_api_key,
    "llm_base_url": st.session_state.config.llm_base_url,
})

if sidebar_config.get("llm_provider"):
    st.session_state.config.llm_provider = sidebar_config["llm_provider"]
if sidebar_config.get("llm_api_key"):
    st.session_state.config.llm_api_key = sidebar_config["llm_api_key"]
if sidebar_config.get("llm_base_url"):
    st.session_state.config.llm_base_url = sidebar_config["llm_base_url"]
st.session_state.config.llm_model = sidebar_config.get("llm_model", "deepseek-chat")
st.session_state.config.temperature = sidebar_config.get("temperature", 0.1)

os.environ["OPENAI_API_KEY"] = st.session_state.config.llm_api_key or ""
if st.session_state.config.llm_base_url:
    os.environ["OPENAI_BASE_URL"] = st.session_state.config.llm_base_url


def _build_initial_state() -> dict:
    return {
        "messages": [],
        "current_stage": "init",
        "stage_history": [],
        "quality_gates": {},
        "project_root": str(st.session_state.config.project_root),
        "skill_root": str(st.session_state.config.skill_root),
        "competition": st.session_state.config.competition,
        "language": st.session_state.config.language,
        "problem_description": st.session_state.workflow_state.get("problem_description", ""),
        "problem_files": st.session_state.workflow_state.get("problem_files", []),
        "attachment_hashes": {},
        "modeling_report": st.session_state.workflow_state.get("modeling_report", ""),
        "terminology_table": st.session_state.workflow_state.get("terminology_table", ""),
        "code_files": [],
        "result_files": [],
        "figure_files": [],
        "reproducibility_manifest": "",
        "code_exec_output": "",
        "evidence_outline": st.session_state.workflow_state.get("evidence_outline", ""),
        "paper_output": st.session_state.workflow_state.get("paper_output", ""),
        "word_paper": "",
        "latex_project": "",
        "pdf_paper": "",
        "subagent_config": st.session_state.config.subagents,
        "error": None,
        "retry_counts": {},
        "stage_output": "",
        "user_input": "",
        "uploaded_files": st.session_state.workflow_state.get("problem_files", []),
    }


def _run_full_workflow():
    workflow = create_workflow(st.session_state.config)
    initial_state = _build_initial_state()

    outputs = {"_stage": "full"}
    config_ = {"configurable": {"thread_id": "main"}}
    for event in workflow.stream(initial_state, config_):
        if st.session_state.get("stop_requested", False):
            st.session_state.run_log.append("⏹️ 用户停止")
            break

        node_name = list(event.keys())[0]
        node_state = event[node_name]
        st.session_state.workflow_state.update(node_state)

        # 保存各阶段产物
        stage_output = node_state.get("stage_output", "")
        if stage_output and node_name not in ("init", "m1_check", "p1_check", "p2_check", "w1_check", "w2_check", "done"):
            outputs[node_name] = stage_output

        # 保存 modeling_report
        if node_state.get("modeling_report"):
            outputs["modeling_report"] = node_state["modeling_report"]
        if node_state.get("terminology_table"):
            outputs["terminology_table"] = node_state["terminology_table"]

        # 保存代码执行输出
        if node_state.get("code_exec_output"):
            outputs["code_exec_output"] = node_state["code_exec_output"]

        # 保存证据大纲和论文（使用专用字段）
        if node_state.get("evidence_outline"):
            outputs["evidence_outline"] = node_state["evidence_outline"]
        if node_state.get("paper_output"):
            outputs["paper_output"] = node_state["paper_output"]

        gates = node_state.get("quality_gates", {})
        log_entry = f"[{node_name}]"
        if gates:
            log_entry += f" 门禁: {gates}"
        if stage_output:
            preview = stage_output[:200].replace("\n", " ")
            log_entry += f" | {preview}..."
        if node_name == "code_exec":
            exec_out = node_state.get("code_exec_output", "")
            if exec_out:
                log_entry += f" | [执行输出: {exec_out[:100]}...]"
        st.session_state.run_log.append(log_entry)

    _save_outputs(outputs)
    st.session_state.stop_requested = False
    st.rerun()


def _run_single_stage(stage: str):
    workflow = create_single_stage_workflow(st.session_state.config, stage)
    initial_state = _build_initial_state()

    outputs = {"_stage": stage}
    config_ = {"configurable": {"thread_id": f"{stage}_only"}}
    for event in workflow.stream(initial_state, config_):
        if st.session_state.get("stop_requested", False):
            st.session_state.run_log.append("⏹️ 用户停止")
            break

        node_name = list(event.keys())[0]
        node_state = event[node_name]
        st.session_state.workflow_state.update(node_state)

        stage_output = node_state.get("stage_output", "")
        if stage_output and node_name not in ("init", "done"):
            outputs[node_name] = stage_output

        # 捕获 modeling_report 和 terminology_table
        if node_state.get("modeling_report"):
            outputs["modeling_report"] = node_state["modeling_report"]
        if node_state.get("terminology_table"):
            outputs["terminology_table"] = node_state["terminology_table"]
        if node_state.get("paper_output"):
            outputs["paper_output"] = node_state["paper_output"]
        if node_state.get("evidence_outline"):
            outputs["evidence_outline"] = node_state["evidence_outline"]
        if node_state.get("code_exec_output"):
            outputs["code_exec_output"] = node_state["code_exec_output"]

        log_entry = f"[{node_name}]"
        if stage_output:
            preview = stage_output[:200].replace("\n", " ")
            log_entry += f" | {preview}..."
        st.session_state.run_log.append(log_entry)

    _save_outputs(outputs)
    st.session_state.stop_requested = False
    st.rerun()


def _save_outputs(outputs: dict):
    output_dir = st.session_state.config.project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.output_files = {}

    # 题目分析报告
    if "modeling_report" in outputs:
        path = output_dir / "题目分析报告.md"
        path.write_text(outputs["modeling_report"], encoding="utf-8")
        st.session_state.output_files["report"] = str(path)
    elif "run" in outputs and outputs.get("_stage") == "modeling":
        path = output_dir / "题目分析报告.md"
        path.write_text(outputs["run"], encoding="utf-8")
        st.session_state.output_files["report"] = str(path)

    # 术语表格 — 优先使用提取后的字段，否则从报告中提取
    if "terminology_table" in outputs and outputs["terminology_table"]:
        tt = outputs["terminology_table"]
        report = outputs.get("modeling_report", "")
        if report:
            tt_ratio = len(tt) / max(len(report), 1)
            if tt_ratio < 0.8:
                path = output_dir / "术语表格.md"
                path.write_text(tt, encoding="utf-8")
                st.session_state.output_files["terminology"] = str(path)
        else:
            path = output_dir / "术语表格.md"
            path.write_text(tt, encoding="utf-8")
            st.session_state.output_files["terminology"] = str(path)
    if "terminology" not in st.session_state.output_files and "modeling_report" in outputs:
        report = outputs["modeling_report"]
        term_start = report.find("术语表格")
        if term_start >= 0:
            term_content = report[term_start:]
            path = output_dir / "术语表格.md"
            path.write_text(term_content, encoding="utf-8")
            st.session_state.output_files["terminology"] = str(path)

    # 代码文件：coding_p1 和 coding_full，以及单阶段 coding
    if "coding_p1" in outputs:
        code = _extract_code_from_output(outputs["coding_p1"])
        if code.strip():
            path = output_dir / "solution_p1.py"
            path.write_text(code, encoding="utf-8")
            st.session_state.output_files["code_p1"] = str(path)

    if "coding_full" in outputs:
        code = _extract_code_from_output(outputs["coding_full"])
        if code.strip():
            path = output_dir / "solution_full.py"
            path.write_text(code, encoding="utf-8")
            st.session_state.output_files["code_full"] = str(path)
    elif "run" in outputs and outputs.get("_stage") == "coding":
        code = _extract_code_from_output(outputs["run"])
        if code.strip():
            path = output_dir / "solution_full.py"
            path.write_text(code, encoding="utf-8")
            st.session_state.output_files["code_full"] = str(path)

    # 论文（writing_full 阶段）
    if "paper_output" in outputs:
        path = output_dir / "完整论文.md"
        path.write_text(outputs["paper_output"], encoding="utf-8")
        st.session_state.output_files["paper"] = str(path)
    elif "writing_full" in outputs:
        path = output_dir / "完整论文.md"
        path.write_text(outputs["writing_full"], encoding="utf-8")
        st.session_state.output_files["paper"] = str(path)
    elif "run" in outputs and outputs.get("_stage") == "writing":
        path = output_dir / "完整论文.md"
        path.write_text(outputs["run"], encoding="utf-8")
        st.session_state.output_files["paper"] = str(path)

    # 证据大纲
    if "evidence_outline" in outputs:
        path = output_dir / "证据大纲.md"
        path.write_text(outputs["evidence_outline"], encoding="utf-8")
        st.session_state.output_files["outline"] = str(path)
    elif "writing_w1" in outputs:
        path = output_dir / "证据大纲.md"
        path.write_text(outputs["writing_w1"], encoding="utf-8")
        st.session_state.output_files["outline"] = str(path)

    # 代码执行输出
    if "code_exec_output" in outputs:
        path = output_dir / "代码执行结果.txt"
        path.write_text(outputs["code_exec_output"], encoding="utf-8")
        st.session_state.output_files["exec_output"] = str(path)

    count = len(st.session_state.output_files)
    st.session_state.run_log.append(f"📁 {count} 个文件已保存到 {output_dir}")


def _run_chat(user_input: str):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"  [Chat] 用户提问中...", file=sys.stderr, flush=True)
    print(f"{'='*60}", file=sys.stderr, flush=True)

    llm = ChatOpenAI(
        model=st.session_state.config.llm_model,
        api_key=st.session_state.config.llm_api_key,
        base_url=st.session_state.config.llm_base_url,
        temperature=st.session_state.config.temperature,
        streaming=True,
    )

    system_prompt = f"""你是数学建模智能助手，基于 math-modeling-skill 项目运行。

当前工作流状态：
- 阶段：{st.session_state.workflow_state.get('current_stage', 'init')}
- 竞赛：{st.session_state.workflow_state.get('competition', 'cumcm')}
- 语言：{st.session_state.workflow_state.get('language', 'chinese')}

你可以帮助用户：
1. 分析数学建模题目
2. 选择合适的模型和算法
3. 编写求解代码
4. 撰写论文

请用中文回答用户的问题。"""

    chat_messages = [SystemMessage(content=system_prompt)]
    for msg in st.session_state.messages[-10:]:
        if msg["role"] == "user":
            chat_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_messages.append(AIMessage(content=msg["content"]))
    chat_messages.append(HumanMessage(content=user_input))

    full_response = []
    for chunk in llm.stream(chat_messages):
        if st.session_state.get("stop_requested", False):
            print(f"\n[用户中断]", file=sys.stderr, flush=True)
            full_response.append("\n\n*[用户中断]*")
            break
        token = chunk.content
        if token:
            print(token, end="", file=sys.stderr, flush=True)
            full_response.append(token)

    response_text = "".join(full_response)
    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"  [Chat] 完成\n", file=sys.stderr, flush=True)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.stop_requested = False
    st.rerun()


st.markdown("---")

step1, step2, step3 = st.columns(3)

with step1:
    st.markdown("### 📁 第一步：上传题目")
    render_step_upload()

with step2:
    st.markdown("### 📝 第二步：描述题目")
    problem_text = render_step_problem()
    if problem_text:
        st.session_state.workflow_state["problem_description"] = problem_text
        st.success(f"✅ 题目已录入（{len(problem_text)} 字）")

with step3:
    st.markdown("### 🚀 第三步：选择模式")
    mode_config = render_step_mode()

    has_files = bool(st.session_state.uploaded_file_paths)
    has_problem = bool(st.session_state.workflow_state.get("problem_description", ""))

    if not has_files and not has_problem:
        st.warning("⚠️ 请先完成第一步和第二步")

    if mode_config["trigger"] and (has_files or has_problem):
        if not st.session_state.config.llm_api_key:
            st.error("❌ 请先在左侧边栏输入 API Key")
        else:
            mode = mode_config["mode"]
            st.session_state.run_log = []

            with st.spinner(f"🚀 正在执行 {mode}..."):
                try:
                    if mode == "full":
                        _run_full_workflow()
                    elif mode == "modeling":
                        _run_single_stage("modeling")
                    elif mode == "coding":
                        _run_single_stage("coding")
                    elif mode == "writing":
                        _run_single_stage("writing")
                except StopRequested:
                    st.warning("⏹️ 已停止")
                    st.session_state.run_log.append("⏹️ 用户中断")
                    st.session_state.stop_requested = False
                except Exception as e:
                    st.error(f"❌ 执行出错：{str(e)}")
                    st.session_state.run_log.append(f"❌ {str(e)}")

    if mode_config["trigger_chat"]:
        user_input = mode_config["chat_input"]
        if user_input and st.session_state.config.llm_api_key:
            _run_chat(user_input)
        elif user_input:
            st.error("❌ 请先在左侧边栏输入 API Key")


st.markdown("---")

render_results(st.session_state.workflow_state, st.session_state.run_log)


st.markdown("---")
st.caption(
    "基于 [math-modeling-skill](https://github.com/XiaoMaColtAI/math-modeling-skill) · "
    "LangChain + LangGraph + Streamlit · "
    "仅供学习参考，生成的论文不作为可直接提交的作品"
)