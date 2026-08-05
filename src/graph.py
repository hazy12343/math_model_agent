import os
import re
import shutil
import sys
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
from src.tools.verifier import NumericalVerifier
from src.tools.detector import TrapDetector


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
    _PYTHON_TOKENS = ("=", "(", ")", "[", "]", "{", "}", ":", "import ", "from ", "def ", "class ")
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if last.startswith("#"):
            break
        looks_like_python = any(t in last for t in _PYTHON_TOKENS)
        if looks_like_python:
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
    result = _merge_broken_strings(code)

    result = _fix_last_line_brackets(result)

    result = _iterative_repair(result)

    if not _validate_code_syntax(result):
        result = _ensure_main_block(result)

    return result


def _merge_broken_strings(code: str) -> str:
    lines = code.split("\n")
    repaired = []
    i = 0
    in_triple = None
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_triple:
            repaired.append(line)
            if in_triple in stripped:
                idx = stripped.index(in_triple)
                after = stripped[idx + 3:]
                if not after or in_triple not in after:
                    in_triple = None
            i += 1
            continue

        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_triple = stripped[:3]
            if stripped.count(in_triple) >= 2:
                in_triple = None
            repaired.append(line)
            i += 1
            continue

        if '"""' in stripped or "'''" in stripped:
            for tq in ['"""', "'''"]:
                if tq in stripped:
                    before = stripped[:stripped.index(tq)]
                    single_quotes = before.count("'") - before.count("\\'")
                    double_quotes = before.count('"') - before.count('\\"')
                    if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                        in_triple = tq
                        if stripped.count(tq) >= 2:
                            in_triple = None
                    break
            repaired.append(line)
            i += 1
            continue

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

    if in_triple:
        repaired.append(in_triple)

    return "\n".join(repaired)


def _fix_last_line_brackets(code: str) -> str:
    lines = code.split("\n")
    if not lines:
        return code
    last_line = lines[-1].strip()
    bare_last = last_line.split("#")[0].strip()
    if bare_last and not bare_last.endswith(":") and not bare_last.endswith("\\"):
        open_parens = bare_last.count("(") - bare_last.count(")")
        open_brackets = bare_last.count("[") - bare_last.count("]")
        open_braces = bare_last.count("{") - bare_last.count("}")
        if open_parens > 0 or open_brackets > 0 or open_braces > 0:
            lines.pop()
            return "\n".join(lines)
    return code


def _iterative_repair(code: str) -> str:
    MAX_ITERATIONS = 15
    result = code
    last_err = None
    for _ in range(MAX_ITERATIONS):
        err = _validate_code_syntax(result)
        if not err:
            break

        if err == last_err:
            break
        last_err = err

        if "used prior to global declaration" in err:
            new_result = _repair_global_declaration(result)
            if new_result != result:
                result = new_result
                continue

        if "was never closed" in err or "unterminated" in err or "EOF" in err:
            new_result = _truncate_to_valid(result)
            if new_result != result:
                result = new_result
                continue

        if "expected an indented block" in err:
            new_result = _remove_empty_functions(result)
            if new_result != result:
                result = new_result
                continue
            new_result = _truncate_to_valid(result)
            if new_result != result:
                result = new_result
                continue

        break

    return result


def _truncate_to_valid(code: str) -> str:
    lines = code.split("\n")
    if len(lines) <= 2:
        return code
    original_err = _validate_code_syntax(code)
    if not original_err:
        return code
    max_remove = min(len(lines) - 1, 300)
    for remove_count in range(1, max_remove + 1):
        truncated = "\n".join(lines[:-remove_count])
        err = _validate_code_syntax(truncated)
        if not err:
            return truncated
        if err != original_err:
            return truncated
    return code


def _repair_global_declaration(code: str) -> str:
    lines = code.split("\n")

    global_entries = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("global ") and not stripped.startswith("global #"):
            global_entries.append((i, stripped))

    if not global_entries:
        return code

    func_starts = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            indent = len(line) - len(line.lstrip())
            func_starts.append((i, indent))

    global_to_func = {}
    for g_idx, _ in global_entries:
        enclosing = None
        for f_idx, _ in func_starts:
            if f_idx < g_idx:
                enclosing = f_idx
            else:
                break
        global_to_func[g_idx] = enclosing

    global_line_indices = {g[0] for g in global_entries}
    result = []
    i = 0
    while i < len(lines):
        if i in global_line_indices:
            i += 1
            continue

        result.append(lines[i])

        if i in global_to_func.values():
            func_globals = [g[1] for g in global_entries if global_to_func.get(g[0]) == i]
            if func_globals:
                j = i + 1
                while j < len(lines) and j not in global_line_indices:
                    stripped = lines[j].strip()
                    if stripped == "" or stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith("#"):
                        result.append(lines[j])
                        j += 1
                    else:
                        break
                body_indent = " " * (len(lines[i]) - len(lines[i].lstrip()) + 4)
                for g in func_globals:
                    result.append(body_indent + g)
                i = j - 1

        i += 1

    return "\n".join(result)


def _remove_empty_functions(code: str) -> str:
    lines = code.split("\n")
    result = []
    block_keywords = ("def ", "for ", "if ", "while ", "with ", "try:", "except", "else:", "elif ", "class ")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        is_block = False
        for kw in block_keywords:
            if stripped.startswith(kw) and stripped.endswith(":"):
                is_block = True
                break
        if is_block:
            indent = len(lines[i]) - len(lines[i].lstrip())
            j = i + 1
            has_body = False
            while j < len(lines):
                next_stripped = lines[j].strip()
                if next_stripped == "":
                    j += 1
                    continue
                next_indent = len(lines[j]) - len(lines[j].lstrip())
                if next_indent <= indent:
                    break
                has_body = True
                break
            if not has_body:
                i = j
                continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


def _ensure_main_block(code: str) -> str:
    if 'if __name__' in code:
        return code

    lines = code.split("\n")

    existing_funcs = set()
    func_ranges = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            fname = stripped[4:-1].split("(")[0].strip()
            if fname:
                existing_funcs.add(fname)
                indent = len(line) - len(line.lstrip())
                func_ranges.append((i, indent))

    def _is_top_level_code(line: str, idx: int) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith("#"):
            return False
        if stripped.startswith("import ") or stripped.startswith("from "):
            return False
        if stripped.startswith("def ") or stripped.startswith("class "):
            return False
        if stripped.startswith("@"):
            return False
        indent = len(line) - len(line.lstrip())
        for f_idx, f_indent in func_ranges:
            if f_idx < idx and indent > f_indent:
                return False
        return True

    top_level_stmts = []
    top_level_indices = set()
    for i, line in enumerate(lines):
        if _is_top_level_code(line, i):
            top_level_stmts.append(line)
            top_level_indices.add(i)

    main_func = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def save_results"):
            main_func = "save_results()"
            break
        if stripped.startswith("def main"):
            main_func = "main()"
            break

    if main_func is None:
        for line in lines:
            stripped = line.strip()
            if not (stripped.startswith("def ") and stripped.endswith(":")):
                continue
            func_name = stripped[4:-1].split("(")[0].strip()
            if func_name in ("sensitivity_analysis", "run", "solve", "compute_all", "generate_results"):
                main_func = f"{func_name}()"
                break
        else:
            for line in lines:
                stripped = line.strip()
                if not (stripped.startswith("def ") and stripped.endswith(":")):
                    continue
                signature = stripped[4:-1].strip()
                func_name = signature.split("(")[0].strip()
                if not func_name or func_name.startswith("_"):
                    continue
                params_str = signature[len(func_name):].strip()
                if params_str.startswith("(") and params_str.endswith(")"):
                    params = params_str[1:-1].strip()
                    if not params:
                        main_func = f"{func_name}()"
                        break
                    has_all_defaults = True
                    for param in params.split(","):
                        param = param.strip()
                        if param and "=" not in param and param != "self":
                            has_all_defaults = False
                            break
                    if has_all_defaults:
                        main_func = f"{func_name}()"
                        break

    if main_func or top_level_stmts:
        plot_funcs = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def plot_") and stripped.endswith(":"):
                sig = stripped[4:-1].strip()
                fname = sig.split("(")[0].strip()
                if fname and "(" not in fname:
                    params_str = sig[len(fname):].strip()
                    if params_str == "()":
                        plot_funcs.append(fname)

        has_sensitivity = "sensitivity_analysis" in existing_funcs

        if top_level_stmts:
            filtered = []
            for i, line in enumerate(lines):
                if i not in top_level_indices:
                    filtered.append(line)
            lines = filtered
            lines.append("")

        lines.append("")
        lines.append('if __name__ == "__main__":')
        if top_level_stmts:
            for stmt in top_level_stmts:
                lines.append(f"    {stmt.strip()}")
        if main_func:
            lines.append(f"    {main_func}")
        if has_sensitivity and main_func != "sensitivity_analysis()":
            lines.append("    try:")
            lines.append("        sensitivity_analysis()")
            lines.append("    except Exception as e:")
            lines.append("        print(f'[sensitivity skipped] {e}')")
        for pf in plot_funcs:
            lines.append("    try:")
            lines.append(f"        {pf}()")
            lines.append("    except Exception as e:")
            lines.append(f"        print(f'[{pf} skipped] {{e}}')")

    return "\n".join(lines)


def _inject_chinese_font(code: str) -> str:
    if "font.sans-serif" in code or "SimHei" in code or "Microsoft YaHei" in code or "rcParams['font.sans-serif']" in code or "rcParams['font.sans-serif" in code:
        return code

    _CJK_RANGES = [
        (0x4E00, 0x9FFF),
        (0x3400, 0x4DBF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF),
        (0xF900, 0xFAFF),
        (0x2F800, 0x2FA1F),
    ]
    _CJK_PUNCTUATION = set("，。！？；：""''（）【】《》…—～·、「」『』〔〕〖〗")

    has_chinese = any(
        any(lo <= ord(c) <= hi for lo, hi in _CJK_RANGES)
        or c in _CJK_PUNCTUATION
        for c in code
    )
    if not has_chinese:
        return code

    has_matplotlib = (
        "matplotlib" in code
        or "plt." in code
        or "savefig" in code
        or "subplot" in code
        or "figure" in code.lower()
    )
    if not has_matplotlib:
        return code

    has_matplotlib_import = "import matplotlib" in code or "from matplotlib" in code

    lines = []
    if not has_matplotlib_import:
        lines.append("import matplotlib")
    lines.append("matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']")
    lines.append("matplotlib.rcParams['axes.unicode_minus'] = False")
    lines.append("matplotlib.rcParams['svg.fonttype'] = 'none'")
    font_block = "\n".join(lines)

    code_lines = code.split("\n")
    insert_idx = 0
    for i, line in enumerate(code_lines):
        stripped = line.strip()
        if stripped.startswith("import matplotlib") or stripped.startswith("from matplotlib"):
            insert_idx = i + 1
        elif stripped.startswith("matplotlib.use") and insert_idx < i:
            insert_idx = i + 1

    if insert_idx == 0:
        for i, line in enumerate(code_lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_idx = i + 1

    code_lines.insert(insert_idx, font_block)
    return "\n".join(code_lines)


def _validate_code_syntax(code: str) -> str:
    try:
        compile(code, "<solution>", "exec")
        return ""
    except SyntaxError as e:
        return f"行 {e.lineno}: {e.msg}"


def _parse_quality_status(result: str) -> str:
    text = result.upper()
    m = re.search(r'状态[：:]\s*`?\s*(PASS|FAIL|BLOCKED)', text)
    if m:
        return m.group(1)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "状态" in line or "STATUS" in line:
            if "PASS" in line and "FAIL" not in line:
                return "PASS"
            if "FAIL" in line:
                return "FAIL"
            if "BLOCKED" in line:
                return "BLOCKED"
            if i + 1 < len(lines):
                nl = lines[i + 1].strip()
                if nl in ("PASS", "FAIL", "BLOCKED"):
                    return nl
                for kw in ("PASS", "FAIL", "BLOCKED"):
                    if kw in nl and all(k not in nl for k in ("PASS", "FAIL", "BLOCKED") if k != kw):
                        return kw
    if "PASS" in text and "FAIL" not in text and "BLOCKED" not in text:
        pass_count = text.count("PASS")
        fail_count = text.count("FAIL")
        blocked_count = text.count("BLOCKED")
        if pass_count > max(fail_count, blocked_count, 0):
            return "PASS"
    if re.search(r'\bFAIL\b', text):
        return "FAIL"
    if re.search(r'\bBLOCKED\b', text):
        return "BLOCKED"
    if re.search(r'\bPASS\b', text):
        return "PASS"
    if "通过" in result or "✅" in result:
        return "PASS"
    if "未通过" in result or "失败" in result or "❌" in result:
        return "FAIL"
    return "UNKNOWN"


def _extract_terminology_table(report: str) -> str:
    for marker in ["## 术语表格", "### 术语表格", "# 术语表格", "术语表格"]:
        idx = report.find(marker)
        if idx >= 0:
            extracted = report[idx:]
            search_start = len(marker)
            next_marker = extracted.find("\n## ", search_start)
            if next_marker > 0:
                return extracted[:next_marker]
            next_marker = extracted.find("\n# ", search_start)
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
            "retry_counts": {},
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
        retry_counts = state.get("retry_counts", {})
        m1_retry = retry_counts.get("M1", 0)

        trap_detector = TrapDetector(project_root)
        trap_note = ""
        if problem:
            attachment_content = ""
            for f in files:
                try:
                    content = Path(f).read_text(encoding="utf-8", errors="replace")
                    attachment_content += content[:3000]
                except Exception:
                    pass
            trap_result = trap_detector.detect_data_anomalies(problem, attachment_content)
            if trap_result["anomaly_count"] > 0:
                trap_note = "\n\n# ⚠️ 题目陷阱检测\n" + "\n".join(
                    f"- {f}" for f in trap_result["findings"] if "P0" in f or "P1" in f
                )
                problem = problem + trap_note

        if m1_retry > 0:
            feedback = state.get("stage_output", "")
            if not feedback:
                for msg in reversed(messages):
                    if hasattr(msg, "content") and "M1" in str(msg.content):
                        feedback = str(msg.content)
                        break
            if feedback:
                result = modeling.fix_model(feedback, messages, project_root)
            else:
                result = modeling.analyze_problem(problem, files, messages, project_root)
        else:
            result = modeling.analyze_problem(problem, files, messages, project_root)

        model_risks = trap_detector.detect_model_risks(result)
        if model_risks["risk_count"] > 0:
            result += "\n\n# ⚠️ 模型风险提示\n" + "\n".join(
                f"- {f}" for f in model_risks["findings"] if "P0" in f or "P1" in f
            )

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

        status = _parse_quality_status(result)
        is_pass = status == "PASS"
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["M1"] = status if status != "UNKNOWN" else ("PASS" if is_pass else "FAIL")
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["M1"] = retry_counts.get("M1", 0) + 1
        else:
            retry_counts["M1"] = 0

        return {
            "current_stage": "m1_check",
            "stage_history": state.get("stage_history", []) + ["m1_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
            "messages": [AIMessage(content=f"[M1 建模终检]\n{result}")],
            "stage_output": result,
            "error": None if is_pass else "M1 建模终检未通过",
        }

    def coding_p1_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        terminology_table = state.get("terminology_table", "")
        project_root = state.get("project_root", str(config.project_root))
        retry_counts = state.get("retry_counts", {})
        p1_retry = retry_counts.get("P1", 0)

        if p1_retry > 0:
            feedback = state.get("stage_output", "")
            if not feedback:
                for msg in reversed(messages):
                    if hasattr(msg, "content") and "P1" in str(msg.content):
                        feedback = str(msg.content)
                        break
            if feedback:
                result = coding.fix_code(feedback, messages, project_root)
            else:
                result = coding.implement_minimal(modeling_report, terminology_table, messages, project_root)
        else:
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

        code_file = _extract_code_from_output(code)
        if code_file.strip():
            code_file = _inject_chinese_font(code_file)
            syntax_err = _validate_code_syntax(code_file)
            if syntax_err:
                repaired = _repair_code_syntax(code_file)
                if not _validate_code_syntax(repaired):
                    code_file = repaired
                else:
                    lines = code_file.split("\n")
                    for remove_n in range(1, max(len(lines) // 2, 10) + 1):
                        candidate = "\n".join(lines[:-remove_n])
                        if not _validate_code_syntax(candidate):
                            code_file = candidate
                            break

            exec_dir = Path(tempfile.mkdtemp(prefix="p1_exec_"))
            exec_file = exec_dir / "solution.py"
            exec_file.write_text(code_file, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(exec_file)],
                    capture_output=True,
                    timeout=config.code_exec_timeout,
                    cwd=str(exec_dir),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                exec_output = proc.stdout.decode("utf-8", errors="replace")
                if proc.stderr:
                    exec_output += f"\n[stderr]:\n{proc.stderr.decode('utf-8', errors='replace')}"
                if proc.returncode != 0:
                    exec_output += f"\n[退出码: {proc.returncode}]"
            except subprocess.TimeoutExpired:
                exec_output = f"⏱️ 代码执行超时（{config.code_exec_timeout}秒）"
            except Exception as e:
                exec_output = f"❌ 代码执行失败: {e}"
            finally:
                shutil.rmtree(exec_dir, ignore_errors=True)
        else:
            exec_output = "（未能从输出中提取代码）"

        result = quality.check_p1(code, exec_output, modeling_report, messages, problem_files)

        status = _parse_quality_status(result)
        is_pass = status == "PASS"
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["P1"] = status if status != "UNKNOWN" else ("PASS" if is_pass else "FAIL")
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["P1"] = retry_counts.get("P1", 0) + 1
        else:
            retry_counts["P1"] = 0

        return {
            "current_stage": "p1_check",
            "stage_history": state.get("stage_history", []) + ["p1_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
            "messages": [AIMessage(content=f"[P1 最小可运行结果门禁]\n{result}")],
            "stage_output": result,
            "error": None if is_pass else "P1 门禁未通过",
        }

    def coding_full_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        terminology_table = state.get("terminology_table", "")
        project_root = state.get("project_root", str(config.project_root))
        retry_counts = state.get("retry_counts", {})
        p2_retry = retry_counts.get("P2", 0)

        if p2_retry > 0:
            feedback = state.get("stage_output", "")
            if not feedback:
                for msg in reversed(messages):
                    if hasattr(msg, "content") and "P2" in str(msg.content):
                        feedback = str(msg.content)
                        break
            if feedback:
                result = coding.fix_code(feedback, messages, project_root)
            else:
                result = coding.implement_full(modeling_report, terminology_table, messages, project_root)
        else:
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

        code_file = _inject_chinese_font(code_file)

        syntax_err = _validate_code_syntax(code_file)
        syntax_note = ""
        if syntax_err:
            original_lines = len(code_file.split("\n"))
            repaired = _repair_code_syntax(code_file)
            repaired_lines = len(repaired.split("\n"))
            syntax_err2 = _validate_code_syntax(repaired)
            if not syntax_err2:
                code_file = repaired
                removed = original_lines - repaired_lines
                if removed > 0:
                    syntax_note = f"\n[代码修复: 自动修复语法错误，从 {original_lines} 行调整为 {repaired_lines} 行（移除 {removed} 行）]"
                else:
                    syntax_note = f"\n[代码修复: 自动修复语法错误]"
            else:
                # 更激进的回退：逐步删除末尾行直到语法正确，最多删除 50% 行
                lines = code_file.split("\n")
                max_aggressive = max(len(lines) // 2, 10)
                aggressive_fixed = None
                for remove_n in range(1, max_aggressive + 1):
                    candidate = "\n".join(lines[:-remove_n])
                    err = _validate_code_syntax(candidate)
                    if not err:
                        aggressive_fixed = candidate
                        break
                if aggressive_fixed:
                    code_file = aggressive_fixed
                    removed = original_lines - len(aggressive_fixed.split("\n"))
                    syntax_note = f"\n[代码修复: 激进截断修复，从 {original_lines} 行调整为 {original_lines - removed} 行（移除 {removed} 行）]"
                else:
                    return {
                        "current_stage": "code_exec",
                        "stage_history": state.get("stage_history", []) + ["code_exec"],
                        "code_exec_output": (
                            f"❌ 代码语法错误（修复失败）:\n"
                            f"  原始错误: {syntax_err}\n"
                            f"  修复后错误: {syntax_err2}\n"
                            f"  原始行数: {original_lines}，修复后行数: {repaired_lines}\n\n"
                            f"可能原因：LLM 输出被截断，代码末尾不完整。请检查 max_tokens 配置。"
                        ),
                        "stage_output": state.get("stage_output", ""),
                        "error": "代码语法错误",
                    }

        exec_dir = Path(tempfile.mkdtemp(prefix="code_exec_"))
        exec_file = exec_dir / "solution.py"
        exec_file.write_text(code_file, encoding="utf-8")

        figure_files = []
        result_files = []
        code_files = []

        try:
            result = subprocess.run(
                [sys.executable, str(exec_file)],
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

            project_root = Path(state.get("project_root", str(config.project_root)))
            figures_dir = project_root / "figures"
            results_dir = project_root / "results"
            figures_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

            img_extensions = {".png", ".svg", ".jpg", ".jpeg", ".pdf", ".eps"}
            result_extensions = {".csv", ".xlsx", ".xls", ".json", ".txt", ".npz", ".pkl", ".pickle"}

            for f in exec_dir.rglob("*"):
                if f.is_file() and f.name != "solution.py":
                    suffix = f.suffix.lower()
                    if suffix in img_extensions:
                        dest = figures_dir / f.name
                        shutil.copy2(f, dest)
                        figure_files.append(str(dest))
                    elif suffix in result_extensions:
                        dest = results_dir / f.name
                        shutil.copy2(f, dest)
                        result_files.append(str(dest))
                    elif suffix == ".py":
                        dest = project_root / f.name
                        shutil.copy2(f, dest)
                        code_files.append(str(dest))

            if figure_files:
                output += f"\n\n[生成图表: {len(figure_files)} 个文件]"
                for ff in figure_files:
                    output += f"\n  - {Path(ff).name}"
            if result_files:
                output += f"\n\n[生成结果文件: {len(result_files)} 个]"
        except subprocess.TimeoutExpired:
            output = f"⏱️ 代码执行超时（{config.code_exec_timeout}秒）"
        except Exception as e:
            output = f"❌ 代码执行失败: {e}"
        finally:
            shutil.rmtree(exec_dir, ignore_errors=True)

        return {
            "current_stage": "code_exec",
            "stage_history": state.get("stage_history", []) + ["code_exec"],
            "code_exec_output": output + syntax_note,
            "stage_output": state.get("stage_output", ""),
            "figure_files": figure_files,
            "result_files": result_files,
            "code_files": code_files,
            "error": None,
        }

    def _run_figure_audit(figures_dir: str, questions: list = None, skill_root: str = None) -> str:
        if skill_root:
            script = Path(skill_root) / "references" / "roles" / "编程手" / "scripts" / "figure_audit.py"
        else:
            script = Path(__file__).parent.parent / "math-modeling-skill" / "references" / "roles" / "编程手" / "scripts" / "figure_audit.py"
        if not script.exists():
            return "[图表审计脚本不存在]"
        try:
            cmd = [sys.executable, str(script), figures_dir, "--min-dpi", "150", "--no-category-check"]
            if questions:
                cmd += ["--questions"] + questions
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout + ("\n" + result.stderr if result.stderr else "")
        except Exception as e:
            return f"[图表审计失败: {e}]"

    def verify_node(state: WorkflowState) -> Dict[str, Any]:
        """数值验证节点：计算验证结果正确性"""
        exec_output = state.get("code_exec_output", "")
        verify_text = ""

        if not config.enable_verification:
            return {
                "current_stage": "verify",
                "stage_history": state.get("stage_history", []) + ["verify"],
                "code_exec_output": exec_output,
                "verification_output": "[数值验证已禁用]",
                "stage_output": state.get("stage_output", ""),
                "error": None,
            }

        code = state.get("stage_output", "")
        problem = state.get("problem_description", "")
        project_root = state.get("project_root", str(config.project_root))
        figure_files = state.get("figure_files", [])

        verifier = NumericalVerifier(project_root)
        trap_detector = TrapDetector(project_root)

        code_file = _extract_code_from_output(code)
        verify_results = []

        cv = verifier.cross_validation(code_file, exec_output)
        verify_results.append(f"[交叉验证] 状态: {cv['status']}")
        for f in cv["findings"]:
            verify_results.append(f"  {f}")

        da = verifier.dimensional_analysis(code_file)
        verify_results.append(f"[量纲分析] 状态: {da['status']}")
        for f in da["findings"]:
            verify_results.append(f"  {f}")

        if problem:
            bc = verifier.boundary_condition_check(exec_output, problem)
            verify_results.append(f"[边界条件] 状态: {bc['status']}")
            for f in bc["findings"]:
                verify_results.append(f"  {f}")

        sc = verifier.sensitivity_check(exec_output)
        verify_results.append(f"[敏感性分析] 状态: {sc['status']}")
        for f in sc["findings"]:
            verify_results.append(f"  {f}")

        if figure_files:
            figures_dir = str(Path(project_root) / "figures")
            fv = verifier.format_verification(figures_dir)
            verify_results.append(f"[图表格式] 状态: {fv['status']}")
            for f in fv["findings"]:
                verify_results.append(f"  {f}")

        num_traps = trap_detector.detect_numerical_traps(code_file)
        if num_traps["trap_count"] > 0:
            verify_results.append(f"[数值陷阱] 检测到 {num_traps['trap_count']} 个陷阱")
            for f in num_traps["findings"]:
                if "P0" in f or "P1" in f:
                    verify_results.append(f"  {f}")

        verify_text = "\n".join(verify_results)
        has_p0 = any("P0" in f for f in verify_results)

        return {
            "current_stage": "verify",
            "stage_history": state.get("stage_history", []) + ["verify"],
            "code_exec_output": exec_output + f"\n\n[数值验证]\n{verify_text}",
            "verification_output": verify_text,
            "stage_output": state.get("stage_output", ""),
            "error": None if not has_p0 else "数值验证发现P0级错误",
        }

    def p2_check_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        code = state.get("stage_output", "")
        problem_files = state.get("problem_files", [])
        exec_output = state.get("code_exec_output", "")
        figure_files = state.get("figure_files", [])
        project_root = state.get("project_root", str(config.project_root))

        figure_list = ""
        figure_audit_result = ""
        if figure_files:
            figure_list = "\n".join(f"  - {Path(f).name}" for f in figure_files)
            figures_dir = str(Path(project_root) / "figures")
            figure_audit_result = _run_figure_audit(figures_dir, skill_root=config.skill_root)
            figure_list = f"共 {len(figure_files)} 个图表文件:\n{figure_list}\n\n图表审计结果:\n{figure_audit_result}"
        else:
            figure_list = "（未检测到图表文件，请确认代码是否生成了图表）"

        result = quality.check_p2(code, exec_output, figure_list, messages, problem_files)

        status = _parse_quality_status(result)
        is_pass = status == "PASS"
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["P2"] = status if status != "UNKNOWN" else ("PASS" if is_pass else "FAIL")
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["P2"] = retry_counts.get("P2", 0) + 1
        else:
            retry_counts["P2"] = 0

        return {
            "current_stage": "p2_check",
            "stage_history": state.get("stage_history", []) + ["p2_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
            "messages": [AIMessage(content=f"[P2 编程终检]\n{result}")],
            "stage_output": result,
            "error": None if is_pass else "P2 门禁未通过",
        }

    def writing_w1_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        code_results = state.get("code_exec_output", state.get("stage_output", ""))
        project_root = state.get("project_root", str(config.project_root))
        figure_files = state.get("figure_files", [])
        retry_counts = state.get("retry_counts", {})
        w1_retry = retry_counts.get("W1", 0)

        figure_list = ""
        if figure_files:
            figure_list = "\n".join(f"  - {Path(f).name}" for f in figure_files)

        if w1_retry > 0:
            feedback = state.get("stage_output", "")
            if not feedback:
                for msg in reversed(messages):
                    if hasattr(msg, "content") and "W1" in str(msg.content):
                        feedback = str(msg.content)
                        break
            if feedback:
                result = writing.fix_paper(feedback, messages, project_root)
            else:
                result = writing.build_evidence_outline(modeling_report, code_results, figure_list, messages, project_root)
        else:
            result = writing.build_evidence_outline(modeling_report, code_results, figure_list, messages, project_root)

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

        status = _parse_quality_status(result)
        is_pass = status == "PASS"
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["W1"] = status if status != "UNKNOWN" else ("PASS" if is_pass else "FAIL")
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["W1"] = retry_counts.get("W1", 0) + 1
        else:
            retry_counts["W1"] = 0

        return {
            "current_stage": "w1_check",
            "stage_history": state.get("stage_history", []) + ["w1_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
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
        figure_files = state.get("figure_files", [])
        retry_counts = state.get("retry_counts", {})
        w2_retry = retry_counts.get("W2", 0)

        figure_list = ""
        if figure_files:
            figure_list = "\n".join(f"  - {Path(f).name}" for f in figure_files)

        if w2_retry > 0:
            feedback = state.get("stage_output", "")
            if not feedback:
                for msg in reversed(messages):
                    if hasattr(msg, "content") and "W2" in str(msg.content):
                        feedback = str(msg.content)
                        break
            if feedback:
                result = writing.fix_paper(feedback, messages, project_root)
            else:
                result = writing.write_paper(modeling_report, code_results, figure_list, evidence, messages, project_root)
        else:
            result = writing.write_paper(modeling_report, code_results, figure_list, evidence, messages, project_root)

        return {
            "current_stage": "writing_full",
            "stage_history": state.get("stage_history", []) + ["writing_full"],
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

        status = _parse_quality_status(result)
        is_pass = status == "PASS"
        quality_gates = dict(state.get("quality_gates", {}))
        quality_gates["W2"] = status if status != "UNKNOWN" else ("PASS" if is_pass else "FAIL")
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["W2"] = retry_counts.get("W2", 0) + 1
        else:
            retry_counts["W2"] = 0

        return {
            "current_stage": "w2_check",
            "stage_history": state.get("stage_history", []) + ["w2_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
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
        retry = state.get("retry_counts", {}).get("M1", 0)
        if gates.get("M1") == "PASS":
            return "coding_p1"
        if retry < config.max_retries:
            return "modeling"
        return "coding_p1"

    def route_after_p1(state: WorkflowState) -> Literal["coding_full", "coding_p1"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_counts", {}).get("P1", 0)
        if gates.get("P1") == "PASS":
            return "coding_full"
        if retry < config.max_retries:
            return "coding_p1"
        return "coding_full"

    def route_after_p2(state: WorkflowState) -> Literal["writing_w1", "coding_full"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_counts", {}).get("P2", 0)
        if gates.get("P2") == "PASS":
            return "writing_w1"
        if retry < config.max_retries:
            return "coding_full"
        return "writing_w1"

    def route_after_w1(state: WorkflowState) -> Literal["writing_full", "writing_w1"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_counts", {}).get("W1", 0)
        if gates.get("W1") == "PASS":
            return "writing_full"
        if retry < config.max_retries:
            return "writing_w1"
        return "writing_full"

    def route_after_w2(state: WorkflowState) -> Literal["done", "writing_full"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_counts", {}).get("W2", 0)
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
    workflow.add_node("verify", verify_node)
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
    workflow.add_edge("code_exec", "verify")
    workflow.add_edge("verify", "p2_check")
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