import os
import shutil
from typing import Literal, Dict, Any
import subprocess
import tempfile
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import WorkflowState
from src.config import AppConfig
from src.agents.modeling import ModelingAgent
from src.agents.coding import CodingAgent
from src.agents.writing import WritingAgent
from src.agents.quality import QualityCheckAgent


def _extract_code_from_output(text: str) -> str:
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                in_code = False
                continue
            else:
                in_code = True
                continue
        if in_code:
            code_lines.append(line)
    if code_lines:
        result = "\n".join(code_lines)
        result = _strip_trailing_non_python(result)
        return result
    return text


def _strip_trailing_non_python(code: str) -> str:
    lines = code.split("\n")
    shell_patterns = [
        "python ", "pip ", "conda ", "apt ", "brew ", "chmod ", "mkdir ",
        "cd ", "ls ", "dir ", "cp ", "mv ", "rm ", "echo ", "export ",
        "set ", "source ", "bash ", "sh ", "cmd ", "powershell ",
    ]
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if last.startswith("#"):
            break
        is_shell = any(
            last.startswith(p) or last.startswith(p.upper())
            for p in shell_patterns
        )
        if is_shell:
            lines.pop()
        else:
            break
    return "\n".join(lines)


def _repair_code_syntax(code: str) -> str:
    import re
    lines = code.split("\n")
    repaired = []
    i = 0
    while i < len(lines):
        line = lines[i]
        single_quotes = line.count("'") - line.count("\\'")
        double_quotes = line.count('"') - line.count('\\"')
        if single_quotes % 2 == 1 or double_quotes % 2 == 1:
            j = i + 1
            while j < len(lines):
                line += lines[j]
                single_quotes = line.count("'") - line.count("\\'")
                double_quotes = line.count('"') - line.count('\\"')
                if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                    i = j
                    break
                j += 1
            repaired.append(line)
        else:
            repaired.append(line)
        i += 1
    result = "\n".join(repaired)

    if _validate_code_syntax(result):
        last_line = repaired[-1].strip() if repaired else ""
        bare_last = last_line.split("#")[0].strip()
        if bare_last and not bare_last.endswith(":") and not bare_last.endswith("\\"):
            open_parens = bare_last.count("(") - bare_last.count(")")
            open_brackets = bare_last.count("[") - bare_last.count("]")
            open_braces = bare_last.count("{") - bare_last.count("}")
            if open_parens > 0 or open_brackets > 0 or open_braces > 0:
                repaired.pop()
                result = "\n".join(repaired)

    err = _validate_code_syntax(result)
    if err and "used prior to global declaration" in err:
        result = _repair_global_declaration(result)

    return result


def _repair_global_declaration(code: str) -> str:
    lines = code.split("\n")

    global_positions = []
    for i, line in enumerate(lines):
        if line.strip().startswith("global "):
            global_positions.append(i)

    if not global_positions:
        return code

    cleaned = [line for i, line in enumerate(lines) if i not in set(global_positions)]

    def get_func_defs(lns):
        funcs = []
        for idx, line in enumerate(lns):
            stripped = line.strip()
            if stripped.startswith("def ") and stripped.endswith(":"):
                funcs.append(idx)
        return funcs

    func_defs = get_func_defs(cleaned)

    all_global_texts = [lines[gp].strip() for gp in global_positions]

    result = []
    for i, line in enumerate(cleaned):
        result.append(line)
        if i in func_defs:
            indent = len(line) - len(line.lstrip())
            func_indent = " " * (indent + 4)
            j = i + 1
            while j < len(cleaned) and (
                cleaned[j].strip() == "" or
                cleaned[j].lstrip().startswith('"""') or
                cleaned[j].lstrip().startswith("'''") or
                cleaned[j].strip().startswith("#")
            ):
                result.append(cleaned[j])
                j += 1
                i = j - 1
            if all_global_texts:
                result.append(func_indent + all_global_texts.pop(0))

    return "\n".join(result)


def _validate_code_syntax(code: str) -> str:
    try:
        compile(code, "<solution>", "exec")
        return ""
    except SyntaxError as e:
        return f"行 {e.lineno}: {e.msg}"


def _extract_terminology_table(report: str) -> str:
    for marker in ["## 术语表格", "### 术语表格", "# 术语表格", "术语表格"]:
        idx = report.find(marker)
        if idx >= 0:
            extracted = report[idx:]
            next_marker = extracted.find("\n## ", len(marker) + 10)
            if next_marker > 0:
                return extracted[:next_marker]
            next_marker = extracted.find("\n# ", len(marker) + 10)
            if next_marker > 0:
                return extracted[:next_marker]
            return extracted
    return ""


def create_workflow(config: AppConfig):
    modeling = ModelingAgent(config)
    coding = CodingAgent(config)
    writing = WritingAgent(config)
    quality = QualityCheckAgent(config)

    workflow = StateGraph(WorkflowState)

    def init_node(state: WorkflowState) -> Dict[str, Any]:
        config.ensure_project_root()
        return {
            "current_stage": "init",
            "stage_history": state.get("stage_history", []) + ["init"],
            "quality_gates": {},
            "error": None,
            "retry_count": 0,
            "project_root": str(config.project_root),
            "skill_root": str(config.skill_root),
            "competition": config.competition,
            "language": config.language,
            "subagent_config": config.subagents,
            "code_files": state.get("code_files", []),
            "result_files": state.get("result_files", []),
            "figure_files": state.get("figure_files", []),
            "stage_output": "✅ 初始化完成，准备开始建模分析。",
        }

    def modeling_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        problem = state.get("problem_description", "")
        files = state.get("problem_files", [])
        project_root = state.get("project_root", str(config.project_root))

        result = modeling.analyze_problem(problem, files, messages, project_root)

        # 尝试从报告中提取术语表格部分
        term_table = _extract_terminology_table(result)

        return {
            "current_stage": "modeling",
            "stage_history": state.get("stage_history", []) + ["modeling"],
            "modeling_report": result,
            "terminology_table": term_table,
            "messages": [AIMessage(content=result)],
            "stage_output": result,
            "error": None,
        }

    def m1_check_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        terminology_table = state.get("terminology_table", "")
        problem = state.get("problem_description", "")
        problem_files = state.get("problem_files", [])

        result = quality.check_m1(modeling_report, terminology_table, problem, messages, problem_files)

        is_pass = "PASS" in result.upper() and "FAIL" not in result.upper()
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["M1"] = "PASS" if is_pass else "FAIL"
        retry_count = state.get("retry_count", 0)
        if not is_pass:
            retry_count += 1
        else:
            retry_count = 0

        return {
            "current_stage": "m1_check",
            "stage_history": state.get("stage_history", []) + ["m1_check"],
            "quality_gates": quality_gates,
            "retry_count": retry_count,
            "messages": [AIMessage(content=f"[M1 建模终检]\n{result}")],
            "stage_output": result,
            "error": None if is_pass else "M1 建模终检未通过",
        }

    def coding_p1_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        terminology_table = state.get("terminology_table", "")
        project_root = state.get("project_root", str(config.project_root))

        result = coding.implement_minimal(modeling_report, terminology_table, messages, project_root)

        return {
            "current_stage": "coding_p1",
            "stage_history": state.get("stage_history", []) + ["coding_p1"],
            "messages": [AIMessage(content=result)],
            "stage_output": result,
            "error": None,
        }

    def p1_check_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        code = state.get("stage_output", "")
        modeling_report = state.get("modeling_report", "")
        problem_files = state.get("problem_files", [])

        result = quality.check_p1(code, "", modeling_report, messages, problem_files)

        is_pass = "PASS" in result.upper() and "FAIL" not in result.upper()
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["P1"] = "PASS" if is_pass else "FAIL"
        retry_count = state.get("retry_count", 0)
        if not is_pass:
            retry_count += 1
        else:
            retry_count = 0

        return {
            "current_stage": "p1_check",
            "stage_history": state.get("stage_history", []) + ["p1_check"],
            "quality_gates": quality_gates,
            "retry_count": retry_count,
            "messages": [AIMessage(content=f"[P1 最小可运行结果门禁]\n{result}")],
            "stage_output": result,
            "error": None if is_pass else "P1 门禁未通过",
        }

    def coding_full_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        terminology_table = state.get("terminology_table", "")
        project_root = state.get("project_root", str(config.project_root))

        result = coding.implement_full(modeling_report, terminology_table, messages, project_root)

        return {
            "current_stage": "coding_full",
            "stage_history": state.get("stage_history", []) + ["coding_full"],
            "messages": [AIMessage(content=result)],
            "stage_output": result,
            "error": None,
        }

    def code_exec_node(state: WorkflowState) -> Dict[str, Any]:
        code = state.get("stage_output", "")

        code_file = _extract_code_from_output(code)
        if not code_file.strip():
            return {
                "current_stage": "code_exec",
                "stage_history": state.get("stage_history", []) + ["code_exec"],
                "code_exec_output": "⚠️ 未能从输出中提取代码",
                "stage_output": state.get("stage_output", ""),
                "error": None,
            }

        syntax_err = _validate_code_syntax(code_file)
        if syntax_err:
            repaired = _repair_code_syntax(code_file)
            syntax_err2 = _validate_code_syntax(repaired)
            if not syntax_err2:
                code_file = repaired
            else:
                return {
                    "current_stage": "code_exec",
                    "stage_history": state.get("stage_history", []) + ["code_exec"],
                    "code_exec_output": f"❌ 代码语法错误（修复失败）:\n  原始错误: {syntax_err}\n  修复后错误: {syntax_err2}\n\n请检查代码中的字符串字面量是否跨行断裂。",
                    "stage_output": state.get("stage_output", ""),
                    "error": "代码语法错误",
                }

        exec_dir = Path(tempfile.mkdtemp(prefix="code_exec_"))
        exec_file = exec_dir / "solution.py"
        exec_file.write_text(code_file, encoding="utf-8")

        try:
            result = subprocess.run(
                ["python", str(exec_file)],
                capture_output=True,
                timeout=config.code_exec_timeout,
                cwd=str(exec_dir),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            output = result.stdout.decode("utf-8", errors="replace")
            if result.stderr:
                output += f"\n\n[stderr]:\n{result.stderr.decode('utf-8', errors='replace')}"
            if result.returncode != 0:
                output += f"\n\n[退出码: {result.returncode}]"
        except subprocess.TimeoutExpired:
            output = f"⏱️ 代码执行超时（{config.code_exec_timeout}秒）"
        except Exception as e:
            output = f"❌ 代码执行失败: {e}"
        finally:
            shutil.rmtree(exec_dir, ignore_errors=True)

        return {
            "current_stage": "code_exec",
            "stage_history": state.get("stage_history", []) + ["code_exec"],
            "code_exec_output": output,
            "stage_output": state.get("stage_output", ""),
            "error": None,
        }

    def p2_check_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        code = state.get("stage_output", "")
        problem_files = state.get("problem_files", [])
        exec_output = state.get("code_exec_output", "")

        result = quality.check_p2(code, exec_output, "", messages, problem_files)

        is_pass = "PASS" in result.upper() and "FAIL" not in result.upper()
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["P2"] = "PASS" if is_pass else "FAIL"
        retry_count = state.get("retry_count", 0)
        if not is_pass:
            retry_count += 1
        else:
            retry_count = 0

        return {
            "current_stage": "p2_check",
            "stage_history": state.get("stage_history", []) + ["p2_check"],
            "quality_gates": quality_gates,
            "retry_count": retry_count,
            "messages": [AIMessage(content=f"[P2 编程终检]\n{result}")],
            "stage_output": result,
            "error": None if is_pass else "P2 门禁未通过",
        }

    def writing_w1_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        code_results = state.get("code_exec_output", state.get("stage_output", ""))
        project_root = state.get("project_root", str(config.project_root))

        result = writing.build_evidence_outline(modeling_report, code_results, "", messages, project_root)

        return {
            "current_stage": "writing_w1",
            "stage_history": state.get("stage_history", []) + ["writing_w1"],
            "messages": [AIMessage(content=result)],
            "stage_output": result,
            "evidence_outline": result,
            "error": None,
        }

    def w1_check_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        evidence = state.get("evidence_outline", state.get("stage_output", ""))
        modeling_report = state.get("modeling_report", "")
        code_results = state.get("code_exec_output", "")
        problem_files = state.get("problem_files", [])

        result = quality.check_w1(evidence, modeling_report, code_results, messages, problem_files)

        is_pass = "PASS" in result.upper() and "FAIL" not in result.upper()
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["W1"] = "PASS" if is_pass else "FAIL"
        retry_count = state.get("retry_count", 0)
        if not is_pass:
            retry_count += 1
        else:
            retry_count = 0

        return {
            "current_stage": "w1_check",
            "stage_history": state.get("stage_history", []) + ["w1_check"],
            "quality_gates": quality_gates,
            "retry_count": retry_count,
            "messages": [AIMessage(content=f"[W1 证据大纲门禁]\n{result}")],
            "stage_output": result,
            "error": None if is_pass else "W1 门禁未通过",
        }

    def writing_full_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        code_results = state.get("code_exec_output", state.get("stage_output", ""))
        evidence = state.get("evidence_outline", "")
        project_root = state.get("project_root", str(config.project_root))

        result = writing.write_paper(modeling_report, code_results, "", evidence, messages, project_root)

        return {
            "current_stage": "writing_full",
            "stage_history": state.get("stage_history", []) + ["writing_full"],
            "word_paper": "完整论文.docx",
            "latex_project": "完整论文-LaTeX/",
            "pdf_paper": "完整论文.pdf",
            "messages": [AIMessage(content=result)],
            "stage_output": result,
            "paper_output": result,
            "error": None,
        }

    def w2_check_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        paper = state.get("paper_output", state.get("stage_output", ""))
        evidence = state.get("evidence_outline", "")
        problem_files = state.get("problem_files", [])

        result = quality.check_w2(paper, evidence, messages, problem_files)

        is_pass = "PASS" in result.upper() and "FAIL" not in result.upper()
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["W2"] = "PASS" if is_pass else "FAIL"
        retry_count = state.get("retry_count", 0)
        if not is_pass:
            retry_count += 1
        else:
            retry_count = 0

        return {
            "current_stage": "w2_check",
            "stage_history": state.get("stage_history", []) + ["w2_check"],
            "quality_gates": quality_gates,
            "retry_count": retry_count,
            "messages": [AIMessage(content=f"[W2 论文终检]\n{result}")],
            "stage_output": result,
            "error": None if is_pass else "W2 门禁未通过",
        }

    def done_node(state: WorkflowState) -> Dict[str, Any]:
        return {
            "current_stage": "done",
            "stage_history": state.get("stage_history", []) + ["done"],
            "messages": [AIMessage(content="🎉 全部流程完成！请查看交付物。")],
            "stage_output": "全部流程完成",
            "error": None,
        }

    def route_after_init(state: WorkflowState) -> Literal["modeling", "done"]:
        files = state.get("problem_files", [])
        problem = state.get("problem_description", "")
        if files or problem:
            return "modeling"
        return "done"

    def route_after_m1(state: WorkflowState) -> Literal["coding_p1", "modeling"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_count", 0)
        if gates.get("M1") == "PASS":
            return "coding_p1"
        if retry < config.max_retries:
            return "modeling"
        return "coding_p1"

    def route_after_p1(state: WorkflowState) -> Literal["coding_full", "coding_p1"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_count", 0)
        if gates.get("P1") == "PASS":
            return "coding_full"
        if retry < config.max_retries:
            return "coding_p1"
        return "coding_full"

    def route_after_p2(state: WorkflowState) -> Literal["writing_w1", "coding_full"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_count", 0)
        if gates.get("P2") == "PASS":
            return "writing_w1"
        if retry < config.max_retries:
            return "coding_full"
        return "writing_w1"

    def route_after_w1(state: WorkflowState) -> Literal["writing_full", "writing_w1"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_count", 0)
        if gates.get("W1") == "PASS":
            return "writing_full"
        if retry < config.max_retries:
            return "writing_w1"
        return "writing_full"

    def route_after_w2(state: WorkflowState) -> Literal["done", "writing_full"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_count", 0)
        if gates.get("W2") == "PASS":
            return "done"
        if retry < config.max_retries:
            return "writing_full"
        return "done"

    workflow.add_node("init", init_node)
    workflow.add_node("modeling", modeling_node)
    workflow.add_node("m1_check", m1_check_node)
    workflow.add_node("coding_p1", coding_p1_node)
    workflow.add_node("p1_check", p1_check_node)
    workflow.add_node("coding_full", coding_full_node)
    workflow.add_node("code_exec", code_exec_node)
    workflow.add_node("p2_check", p2_check_node)
    workflow.add_node("writing_w1", writing_w1_node)
    workflow.add_node("w1_check", w1_check_node)
    workflow.add_node("writing_full", writing_full_node)
    workflow.add_node("w2_check", w2_check_node)
    workflow.add_node("done", done_node)

    workflow.set_entry_point("init")

    workflow.add_conditional_edges("init", route_after_init, {"modeling": "modeling", "done": "done"})
    workflow.add_edge("modeling", "m1_check")
    workflow.add_conditional_edges("m1_check", route_after_m1, {"coding_p1": "coding_p1", "modeling": "modeling"})
    workflow.add_edge("coding_p1", "p1_check")
    workflow.add_conditional_edges("p1_check", route_after_p1, {"coding_full": "coding_full", "coding_p1": "coding_p1"})
    workflow.add_edge("coding_full", "code_exec")
    workflow.add_edge("code_exec", "p2_check")
    workflow.add_conditional_edges("p2_check", route_after_p2, {"writing_w1": "writing_w1", "coding_full": "coding_full"})
    workflow.add_edge("writing_w1", "w1_check")
    workflow.add_conditional_edges("w1_check", route_after_w1, {"writing_full": "writing_full", "writing_w1": "writing_w1"})
    workflow.add_edge("writing_full", "w2_check")
    workflow.add_conditional_edges("w2_check", route_after_w2, {"done": "done", "writing_full": "writing_full"})
    workflow.add_edge("done", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def create_single_stage_workflow(config: AppConfig, stage: str):
    modeling = ModelingAgent(config)
    coding = CodingAgent(config)
    writing = WritingAgent(config)
    quality = QualityCheckAgent(config)

    workflow = StateGraph(WorkflowState)

    def run_single_stage(state: WorkflowState) -> Dict[str, Any]:
        config.ensure_project_root()
        messages = state.get("messages", [])
        problem = state.get("problem_description", "")
        files = state.get("problem_files", [])
        project_root = state.get("project_root", str(config.project_root))
        modeling_report = state.get("modeling_report", "")
        terminology_table = state.get("terminology_table", "")

        if stage == "modeling":
            result = modeling.analyze_problem(problem, files, messages, project_root)
            term_table = _extract_terminology_table(result)
            return {
                "current_stage": "done",
                "modeling_report": result,
                "terminology_table": term_table,
                "messages": [AIMessage(content=result)],
                "stage_output": result,
            }
        elif stage == "coding":
            result = coding.implement_full(modeling_report, terminology_table, messages, project_root)
            return {
                "current_stage": "done",
                "messages": [AIMessage(content=result)],
                "stage_output": result,
            }
        elif stage == "writing":
            evidence_outline = state.get("evidence_outline", "")
            code_results = state.get("code_exec_output", "")
            result = writing.write_paper(modeling_report, code_results, "", evidence_outline, messages, project_root)
            return {
                "current_stage": "done",
                "word_paper": "完整论文.docx",
                "paper_output": result,
                "messages": [AIMessage(content=result)],
                "stage_output": result,
            }
        return {"current_stage": "done"}

    workflow.add_node("run", run_single_stage)
    workflow.set_entry_point("run")
    workflow.add_edge("run", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)