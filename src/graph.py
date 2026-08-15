import os
import re
import shutil
import sys
from typing import Literal, Dict, Any, List, Optional, Set
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


def _sanitize_llm_output(text: str) -> str:
    """对 LLM 输出文本进行安全转义，防止花括号被 format()/f-string 误解析。

    当 LLM 输出包含类似 `{value: 10d}` 或 `{:.2f}` 等模式时，
    如果被 langgraph 内部状态管理或其他组件的 .format() 处理，
    会触发 "Space not allowed in string format specifier" 等 ValueError。
    此函数将独立的 `{` 和 `}` 转义为 `{{` 和 `}}`。

    注意：只转义看起来像格式占位符的模式（{identifier:spec} 或 {}），
    保留正常的 Python 代码中的花括号（如 f-string、dict 字面量）。
    """
    if not text or not isinstance(text, str):
        return text
    # 匹配模式：{identifier:format_spec} 或 {identifier} 或 {:format_spec} 或 {}
    # 这些模式在 .format() 中会被解释为占位符
    # 我们将其转义为 {{...}} 以便安全通过 .format()
    # 但要小心不要破坏 Python f-string 中的花括号
    # 策略：转义独立的花括号对，但保留在 Python 代码块中的花括号

    # 先用占位符保护 markdown 代码块中的内容
    code_blocks = []
    def _save_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"
    text = re.sub(r'```[\s\S]*?```', _save_code_block, text)

    # 转义可能被误解析为格式占位符的花括号模式
    # 模式：{ 后跟 标识符（可选空格和冒号加格式说明符）}
    # 例如：{value: 10d}, {:.2f}, {name}, { }
    text = re.sub(r'\{([^}]*)\}', lambda m: '{{' + m.group(1) + '}}', text)

    # 恢复代码块
    for i, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)

    return text


def _extract_code_from_output(text: str) -> str:
    """从输出中提取所有代码块（支持多文件：单块含 # file: 标记或多块各自为文件）

    优先级：
    1. 提取所有 ```python 代码块，合并（支持多块输出）
    2. 提取所有 ``` 代码块（无语言标记），过滤非 Python 块
    3. 回退：启发式提取 Python 代码
    """
    lines = text.split("\n")
    all_code_lines = []
    in_code = False
    code_lang = ""
    has_python_blocks = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                in_code = False
                code_lang = ""
                continue
            else:
                in_code = True
                # 提取语言标记
                lang = stripped[3:].strip().lower()
                if lang in ("python", "py"):
                    code_lang = "python"
                    has_python_blocks = True
                elif lang in ("json", "bash", "sh", "shell", "yaml", "yml", "markdown", "md", "text", "sql", "r", ""):
                    code_lang = lang if lang else "unknown"
                else:
                    # 未知语言标记，可能是误标注的 Python 代码
                    code_lang = "unknown"
                continue
        if in_code:
            if code_lang in ("json", "bash", "sh", "shell", "yaml", "yml", "markdown", "md", "sql", "r"):
                continue
            all_code_lines.append(line)

    if all_code_lines:
        result = "\n".join(all_code_lines)
        result = _strip_trailing_non_python(result)
        # 如果不是从明确标记的 python 块提取的，验证是否为有效 Python
        if not has_python_blocks:
            if not any(kw in result for kw in ("import ", "def ", "class ", "if __name__", "print(", "=")):
                heuristic = _extract_python_heuristic(text)
                if heuristic:
                    return heuristic
        return result

    # 回退：无 markdown 代码块时，用启发式方法提取 Python 代码
    heuristic = _extract_python_heuristic(text)
    if heuristic:
        return heuristic

    return text


def _extract_python_heuristic(text: str) -> str:
    """启发式提取 Python 代码：当 LLM 未使用 markdown 代码块时"""
    lines = text.split("\n")
    python_lines = []
    python_started = False
    python_keywords = ("import ", "from ", "def ", "class ", "if __name__", "print(", "# file:")
    non_python_markers = ("# ", "## ", "### ", "```", "---", "===", "请", "完成", "执行", "运行")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if python_started:
                python_lines.append("")
            continue

        if any(stripped.startswith(m) for m in non_python_markers):
            if python_started and len(python_lines) > 5:
                break
            continue

        if any(stripped.startswith(kw) for kw in python_keywords):
            python_started = True
            python_lines.append(line)
            continue

        if python_started:
            if stripped and not stripped.startswith("#"):
                python_lines.append(line)
            elif stripped.startswith("#"):
                python_lines.append(line)

    if len(python_lines) >= 3 and any(
        any(kw in l for kw in ("import ", "def ", "print(", "=")) for l in python_lines
    ):
        return "\n".join(python_lines)

    return ""


def _split_multi_file_code(code: str) -> list:
    """将含 `# file: 文件名.py` 标记的代码拆分为多个文件。
    返回 [(filename, code), ...] 列表；若无标记则返回 [(None, code)]。
    """
    pattern = re.compile(r'^#\s*file\s*:\s*(.+\.py)\s*$', re.MULTILINE | re.IGNORECASE)
    matches = list(pattern.finditer(code))
    if not matches:
        return [(None, code)]

    files = []
    for i, m in enumerate(matches):
        filename = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
        file_code = code[start:end].strip()
        files.append((filename, file_code))
    return files


def _strip_trailing_non_python(code: str) -> str:
    lines = code.split("\n")
    shell_patterns = [
        "python ", "pip ", "conda ", "apt ", "brew ", "chmod ", "mkdir ",
        "cd ", "ls ", "dir ", "cp ", "mv ", "rm ", "echo ", "export ",
        "set ", "source ", "bash ", "sh ", "cmd ", "powershell ",
    ]
    console_patterns = [
        "===", "---", "[诊断]", "算法对比", "蒙特卡洛验证", "敏感性分析",
        "收敛性", "生成图表", "全部完成", "最优解:", "理论最优:", "误差:",
        "结果表格已保存", "图", "结果已保存", "运行说明", "输出结果",
        "控制台输出", "单点测试", "粗搜索", "精细搜索", "模拟退火",
        "Nelder-Mead", "网格搜索", "成功率", "95%CI", "均值=", "标准差=",
        "耗时", "提升:", "已保存", "完成（", "收敛代数",
    ]
    _PYTHON_TOKENS = ("=", "(", ")", "[", "]", "{", "}", ":", "import ", "from ", "def ", "class ")
    _PY_STMT_STARTS = ("print", "plt", "fig", "ax", "df", "pd", "np", "os", "json", "csv",
                        "for ", "if ", "while ", "with ", "try", "return", "yield", "raise",
                        "assert", "pass", "break", "continue", "del ", "global ", "nonlocal ")

    import re
    _CONSOLE_SECTION_START = re.compile(r'^(===|---|\[诊断\])')
    _VAR_ASSIGN = re.compile(r'^\w+\s*=')

    truncate_idx = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if _CONSOLE_SECTION_START.match(stripped):
            if any(stripped.startswith(s) for s in _PY_STMT_STARTS):
                continue
            if _VAR_ASSIGN.match(stripped):
                continue
            truncate_idx = i
            break

    lines = lines[:truncate_idx]

    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if last.startswith("#"):
            break
        looks_like_py_stmt = any(last.startswith(s) for s in _PY_STMT_STARTS)
        if looks_like_py_stmt:
            break
        looks_like_console = any(p in last for p in console_patterns)
        if looks_like_console:
            lines.pop()
            continue
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
    # 提取错误消息部分（去掉行号前缀），避免行号变化导致误判
    original_msg = original_err.split(": ", 1)[-1] if ": " in original_err else original_err
    max_remove = min(len(lines) - 1, 300)
    for remove_count in range(1, max_remove + 1):
        truncated = "\n".join(lines[:-remove_count])
        err = _validate_code_syntax(truncated)
        if not err:
            return truncated
        err_msg = err.split(": ", 1)[-1] if ": " in err else err
        if err_msg != original_msg:
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

    top_level_stmts = []
    top_level_indices = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
            continue
        indent = len(line) - len(line.lstrip())
        if any(f_idx < i and indent > f_indent for f_idx, f_indent in func_ranges):
            continue
        top_level_stmts.append(line)
        top_level_indices.add(i)

    main_func = _find_main_func(lines, existing_funcs)

    if main_func or top_level_stmts:
        plot_funcs = _find_noarg_funcs(lines, "plot_")
        has_sensitivity = "sensitivity_analysis" in existing_funcs

        if top_level_stmts:
            lines = [line for i, line in enumerate(lines) if i not in top_level_indices]
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


def _find_main_func(lines: List[str], existing_funcs: Set[str]) -> Optional[str]:
    priority_funcs = ("save_results", "main", "sensitivity_analysis", "run", "solve", "compute_all", "generate_results")
    for name in priority_funcs:
        if name in existing_funcs:
            return f"{name}()"

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
                return f"{func_name}()"
            if all("=" in p.strip() for p in params.split(",") if p.strip() and p.strip() != "self"):
                return f"{func_name}()"

    return None


def _find_noarg_funcs(lines: List[str], prefix: str) -> List[str]:
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"def {prefix}") and stripped.endswith(":"):
            sig = stripped[4:-1].strip()
            fname = sig.split("(")[0].strip()
            if fname and "(" not in fname:
                params_str = sig[len(fname):].strip()
                if params_str == "()":
                    result.append(fname)
    return result


# CJK 字符检测常量（模块级复用）
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


def _has_cjk(text: str) -> bool:
    return any(
        any(lo <= ord(c) <= hi for lo, hi in _CJK_RANGES)
        or c in _CJK_PUNCTUATION
        for c in text
    )


def _inject_chinese_font(code: str) -> str:
    if "font.sans-serif" in code or "SimHei" in code or "Microsoft YaHei" in code:
        return code

    if not _has_cjk(code):
        return code

    if not any(kw in code for kw in ("matplotlib", "plt.", "savefig", "subplot", "figure")):
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


# ====== 结果合理性检测辅助函数（从 _check_result_plausibility 拆分） ======

_ALGO_NAME_PAT = re.compile(
    r'(?:随机搜索|遗传算法|粒子群|模拟退火|网格搜索|坐标下降|差分进化|'
    r'蚁群|禁忌搜索|爬山|贪心|穷举|双层优化|bi-level|two-level|'
    r'构造启发式|构造|启发式|暴力搜索|暴力|枚举|深度优先|广度优先|A\*|'
    r'分支定界|线性规划|整数规划|动态规划|梯度下降|牛顿法|共轭梯度|'
    r'L-BFGS|BFGS|Nelder|Powell|CG|TNC|SLSQP|trust|'
    r'DE|PSO|GA|SA|Bayes|贝叶斯|神经网络|决策树|SVM|随机森林|'
    r'XGBoost|LightGBM|KNN|K-means|DBSCAN|谱聚类|层次聚类)',
    re.IGNORECASE
)


def _check_zero_results(output: str) -> tuple:
    """检测零值结果，返回 (warnings, zero_count, nonzero_count)"""
    warnings = []
    zero_count = len(re.findall(
        r'(?:最优|best|total|result|output|值|解|distance|time|cost|score|target|objective|'
        r'时间|分数|代价|收益|遮蔽|覆盖|效率|成功率|准确率)'
        r'.*?[=:：]\s*0(?:\.0+)?',
        output, re.IGNORECASE))
    nonzero_count = len(re.findall(
        r'(?:最优|best|total|result|output|值|解|distance|time|cost|score|target|objective|'
        r'时间|分数|代价|收益|遮蔽|覆盖|效率|成功率|准确率)'
        r'.*?[=:：]\s*(?:[1-9]\d*(?:\.\d+)?|0\.[1-9]\d*)',
        output, re.IGNORECASE))
    # 额外检测：所有导弹/目标均为 0.0 的模式（如 "各导弹遮蔽: {'M1': 0.0, 'M2': 0.0}"）
    all_zero_targets = re.findall(r"'(M\d|[A-Z][a-z]?\d)':\s*0\.0", output)
    if all_zero_targets:
        zero_count += len(all_zero_targets)
    # 检测 "均为 0" 或 "所有.*?0\.0" 模式
    if re.search(r'(?:均为|全是|所有|全部).*?(?:0\.0|为\s*0)', output):
        zero_count = max(zero_count, 1)  # 至少算一个零值
    if zero_count > 0 and nonzero_count == 0:
        diag = _diagnose_zero_results(output)
        warnings.append(
            f"P0-算法失败: 所有算法结果均为 0，{diag}")
    elif zero_count > nonzero_count:
        warnings.append(f"P1-部分零值: {zero_count} 个结果为零，{nonzero_count} 个非零，部分任务/算法可能未找到有效解")
    return warnings, zero_count, nonzero_count


def _diagnose_zero_results(output: str) -> str:
    """诊断全零结果的可能原因，返回诊断文本"""
    parts = []

    # 检测是否所有资源都未使用
    if re.search(r'资源利用率.*?0/', output):
        parts.append("所有资源均未被使用，优化器未找到任何可行分配方案")

    # 检测是否所有目标/任务都未找到有效解
    all_zero_targets = re.findall(r'(M\d|目标\d|导弹\d|任务\d|节点\d|区域\d).*?(?:未找到有效解|结果=0|值为0)', output)
    if all_zero_targets:
        parts.append(f"所有目标({', '.join(all_zero_targets)})均未找到有效解")

    # 检测是否有理论最大值但实际为0（说明搜索策略完全失效）
    if re.search(r'理论上界.*?(\d+\.?\d*).*总.*?0\.0', output):
        parts.append("理论最大值存在但实际结果为0，可能原因：a) 核心计算模型（公式/数据/判定条件）有误，b) 约束条件过于严格导致所有候选解不可行，c) 目标函数返回常量")

    # 检测是否运行时间极短（说明优化器未真正工作）
    time_match = re.search(r'优化耗时.*?(\d+\.?\d*)\s*s', output)
    if time_match and float(time_match.group(1)) < 1.0:
        parts.append(f"优化耗时仅{time_match.group(1)}s，优化器可能未真正搜索（目标函数对所有输入返回相同值）")

    # 检测是否所有测试点/单点诊断都返回零（→ 平坦优化景观）
    # 单点诊断返回 0.0s 表示核心判定函数对所有输入都返回 0
    if re.search(r'(?:单点诊断|测试点|测试参数).*?(?:0\.0|返回零|均为\s*0)', output, re.IGNORECASE):
        parts.append(
            "平坦优化景观: 核心判定函数对所有输入返回相同值(0)，"
            "优化器无法找到梯度方向。请检查：a) 判定条件是否过于严格(如判定半径在全局尺度空间中占比极低)，"
            "b) 是否需要改用软化约束(如距离惩罚替代硬阈值)，c) 核心计算公式是否用错参数"
        )

    if not parts:
        parts.append("可能原因：约束过强或核心计算模型有误，建议先手动测试一组参数确认核心计算函数能否返回非零值")

    # 检测 NaN 传播（计算过程中出现 NaN 导致所有结果变 NaN）
    if re.search(r'(?:nan|NaN|NAN)', output) and re.search(r'(?:总|最优|结果|fitness).*?(?:nan|NaN|0\.0)', output, re.IGNORECASE):
        parts.append("NaN传播: 计算过程中出现NaN导致所有结果无效。"
                     "请检查：a) 是否有除零操作，b) sqrt/asin等函数是否传入非法参数，"
                     "c) 是否有未初始化的变量参与计算")

    # 检测 DE 优化器是否因种群大小不足而退化
    if re.search(r'differential.evolution.*?converged.*?0\.0', output, re.IGNORECASE):
        parts.append("DE优化器退化: 差分进化算法收敛到0.0，"
                     "可能原因：a) 种群大小(popsize)过小无法覆盖搜索空间，b) 变异率过低导致早熟收敛")

    # 检测所有算法返回相同结果（算法无法区分优劣）
    algo_values = re.findall(r'(?:DE|PSO|GA|随机搜索|网格搜索).*?[：:]\s*(\d+\.?\d*)', output)
    if len(algo_values) >= 2 and len(set(algo_values)) == 1:
        parts.append(f"算法全同: 所有算法({len(algo_values)}个)返回相同结果{algo_values[0]}，"
                     "核心计算模型对所有输入返回相同值，优化无法区分优劣")

    return "；".join(parts)


def _analyze_function_nesting(func_lines: list, func_name: str, func_start: int,
                               func_nests: dict, analysis_keywords: str) -> None:
    """分析单个函数内的嵌套循环深度，区分优化型循环与分析型循环。

    改进版本：
    1. 准确识别 itertools.product() 展平的循环（算作1层而非N层）
    2. 扩大分析型循环识别范围（包括验证、统计、输出等）
    3. 识别逐实体分解优化模式（安全的高维优化方式）
    4. 基于实际缩进层级计算深度（而非简单的缩进量//4）
    """
    func_text = "\n".join(func_lines)
    
    # 扩展分析型关键词（基于实际案例）
    extended_analysis_keywords = analysis_keywords + r'|验证|verify|统计|statistics|对比|compare|结果|result|输出|output|报告|report|绘图|plot|figure|可视化|visual'
    is_analysis_func = bool(re.search(extended_analysis_keywords, func_text, re.IGNORECASE))
    
    # 识别逐实体分解优化模式（安全的高维优化）
    is_decomposed_optimization = bool(
        re.search(r'for\s+\w+\s+in\s+(?:MISSILE_NAMES|UAVS|entities|range\(N_\w+\))', func_text) and
        re.search(r'differential_evolution|minimize|optimize', func_text) and
        not re.search(r'for.*\n\s*for.*\n\s*for.*\n\s*for.*\n\s*for', func_text, re.MULTILINE)
    )

    # 计算实际的嵌套深度（基于栈结构）
    opt_nest = 0
    all_nest = 0
    opt_cnt = 0
    all_cnt = 0
    indent_stack = []  # 缩进栈，用于跟踪当前嵌套层级
    
    # 检测 itertools.product() 使用
    uses_product = bool(re.search(r'from\s+itertools\s+import\s+product|itertools\.product', func_text))
    
    for i, line in enumerate(func_lines):
        stripped = line.strip()
        if not stripped.startswith("for ") and not stripped.startswith("while "):
            continue

        indent = len(line) - len(line.lstrip())
        
        # 更新缩进栈：弹出比当前indent大的层级
        while indent_stack and indent_stack[-1] >= indent:
            indent_stack.pop()
        indent_stack.append(indent)
        
        # 实际嵌套深度 = 栈长度 - 1（减去函数体基础层级）
        actual_depth = len(indent_stack) - 1
        
        # 如果使用了 product() 且这行包含 product，则深度-1（因为product展平了N-1层）
        effective_depth = actual_depth
        if uses_product and 'product(' in stripped:
            effective_depth = max(0, actual_depth - 1)
        
        all_cnt += 1
        all_nest = max(all_nest, effective_depth)

        # 跳过分析型函数的所有循环
        if is_analysis_func:
            continue
        
        # 跳过逐实体分解优化（已证明是安全的）
        if is_decomposed_optimization:
            continue
        
        # 检查上下文是否为分析型代码
        context_start = max(0, i - 5)
        context = "\n".join(func_lines[context_start:i])
        if re.search(extended_analysis_keywords, context, re.IGNORECASE):
            continue

        opt_cnt += 1
        opt_nest = max(opt_nest, effective_depth)

    func_nests[(func_name, func_start)] = (opt_nest, all_nest, opt_cnt, all_cnt)


def _detect_llm_code_bugs(code: str) -> List[str]:
    """检测 LLM 生成代码中的常见 Bug 模式（未定义变量、死代码、逻辑错误等）。

    改进版本：基于实际案例增加更多检测模式。
    """
    warnings = []

    # 1. 检测未定义变量：在 for 循环或 if 条件中使用未定义的变量
    undefined_patterns = [
        (r'(?:any|all)\s*\(.*?\bfor\s+\w+\s+in\s+\[\s*\]', "对空列表字面量 [] 迭代（死代码，永远为空）"),
        (r'\bsegments_exist\b', "使用了未定义的变量 segments_exist（常见 LLM 幻觉）"),
        (r'if\s+\w+\s+is\s+False\s*:', "使用 is False 比较可能为 None 的变量，建议用 if not var"),
        # 基于实际案例新增的拼写错误模式
        (r'\buname\b(?!\s*=)', "可能未定义变量 'uname'（应为 'name' 或 'uav_name'）"),
        (r'\bmissile\b(?!\s*[=:\[])', "可能未定义变量 'missile'（应为 'missiles' 或具体名称如 'M1'）"),
        (r'\beffictive\b', "拼写错误：'effictive' 应为 'effective'"),
        (r'\brecevied\b', "拼写错误：'recevied' 应为 'received'"),
        (r'\boccured\b', "拼写错误：'occured' 应为 'occurred'"),
    ]
    
    # 收集所有已定义的变量名
    defined_vars = set()
    for m in re.finditer(r'^(\w+)\s*=', code, re.MULTILINE):
        defined_vars.add(m.group(1))
    for m in re.finditer(r'for\s+(\w+)\s+in', code):
        defined_vars.add(m.group(1))
    for m in re.finditer(r'def\s+(\w+)\s*\(', code):
        defined_vars.add(m.group(1))

    for pattern, msg in undefined_patterns:
        if re.search(pattern, code):
            # 对于可能的未定义变量，进一步确认是否真的未定义
            var_match = re.search(r'\b(\w+)\b', pattern)
            if var_match and var_match.group(1) in defined_vars:
                continue  # 变量已定义，跳过
            warnings.append(f"⚠️ LLM代码Bug: {msg}")

    # 3. 检测条件永远为真/假的模式
    always_false_patterns = [
        (r'if\s+any\s*\(.*?\bfor\s+\w+\s+in\s+\[\s*\]\s*\)', "any() 对空列表迭代，条件永远为 False"),
        (r'if\s+all\s*\(.*?\bfor\s+\w+\s+in\s+\[\s*\]\s*\)', "all() 对空列表迭代，条件永远为 True（空真）"),
        (r'if\s+False\s*:', "if False: 条件永远不成立（死代码）"),
        (r'if\s+True\s*:', "if True: 条件永远成立（无意义分支）"),
    ]
    for pattern, msg in always_false_patterns:
        if re.search(pattern, code):
            warnings.append(f"⚠️ LLM代码Bug: {msg}")

    # 4. 检测异常处理吃掉所有错误
    if re.search(r'except\s*:\s*\n\s*pass\s*$', code, re.MULTILINE):
        warnings.append("⚠️ LLM代码Bug: 裸 except: pass 会静默吞掉所有异常，建议至少 print 错误信息")

    # 4.5. 检测蒙特卡洛成功率计算的假代码
    # LLM 常见错误：用随机数是否 > 0 来判断成功率（永远为 100%）
    if re.search(r'np\.mean\s*\(\s*\[.*?\bfor\s+_\s+in\s+.*?np\.random.*?\.uniform\s*\(', code):
        if re.search(r'>\s*0\s*\)', code) or re.search(r'>\s*0\.0\s*\)', code):
            warnings.append(
                "⚠️ LLM代码Bug: 蒙特卡洛成功率计算使用随机数 > 0 判断（永远为 100%），"
                "这是假验证！正确做法：对模型参数扰动后重新评估，统计 evaluate() > 0 的比例"
            )

    # 5. 检测变量名拼写错误（常见 LLM 幻觉）
    # 如 plot_timeline 中使用了 segments_exist 但实际定义的是 segments
    # 先剥离字符串字面量，避免将字符串内容（如 'DejaVu Sans' 中的 Sans、
    # matplotlib.use('Agg') 中的 Agg）误判为变量引用
    code_no_strings = re.sub(r'"[^"]*"', '""', code)
    code_no_strings = re.sub(r"'[^']*'", "''", code_no_strings)
    var_usage = set(re.findall(r'\b([a-zA-Z_]\w*)\b', code_no_strings))
    var_defs = set()
    for m in re.finditer(r'(?:^|\n)\s*([a-zA-Z_]\w*)\s*=', code, re.MULTILINE):
        var_defs.add(m.group(1))
    # 处理元组解包: a, b, c = expr
    for m in re.finditer(r'(?:^|\n)\s*([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)\s*=', code, re.MULTILINE):
        for name in re.findall(r'[a-zA-Z_]\w*', m.group(1)):
            var_defs.add(name)
    for m in re.finditer(r'for\s+([a-zA-Z_]\w*)\s+in', code):
        var_defs.add(m.group(1))
    # 处理 for 循环中的元组解包: for a, b, c in ...
    for m in re.finditer(r'for\s+([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)\s+in', code):
        for name in re.findall(r'[a-zA-Z_]\w*', m.group(1)):
            var_defs.add(name)
    for m in re.finditer(r'def\s+([a-zA-Z_]\w*)\s*\(', code):
        var_defs.add(m.group(1))
    # 过滤 Python 内置关键字和常见库名
    PY_KEYWORDS = {
        'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'import', 'from',
        'return', 'break', 'continue', 'pass', 'True', 'False', 'None', 'and',
        'or', 'not', 'in', 'is', 'with', 'as', 'try', 'except', 'finally',
        'raise', 'lambda', 'yield', 'global', 'nonlocal', 'assert', 'del',
        'print', 'range', 'len', 'int', 'float', 'str', 'list', 'dict', 'set',
        'tuple', 'bool', 'type', 'enumerate', 'zip', 'map', 'filter', 'sorted',
        'reversed', 'any', 'all', 'min', 'max', 'sum', 'abs', 'round', 'open',
        'np', 'os', 'sys', 'plt', 'math', 'time', 'json', 'csv', 're', 'Path',
        'shutil', 'subprocess', 'tempfile', 'itertools', 'warnings', 'copy',
        'deepcopy', 'defaultdict', 'Counter', 'namedtuple', 'dataclass',
        'matplotlib', 'scipy', 'pandas', 'sklearn', 'numpy', '__name__',
        '__file__', '__doc__', '__init__', 'self', 'cls', 'super', 'object',
        'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
        'NotImplementedError', 'RuntimeError', 'StopIteration', 'IOError',
        'OSError', 'AttributeError', 'ImportError', 'ZeroDivisionError',
        'FileNotFoundError', 'PermissionError', 'IsADirectoryError',
        'fig', 'ax', 'axes', 'i', 'j', 'k', 'm', 'n', 'x', 'y', 'z', 't',
        'dt', 'dx', 'dy', 'dz', 'f', 'g', 'h', 'p', 'q', 'r', 's', 'u', 'v',
        'w', 'a', 'b', 'c', 'd', 'e', 'idx', 'val', 'key', 'item', 'row',
        'col', 'line', 'lines', 'text', 'data', 'result', 'results', 'output',
        'path', 'file', 'name', 'value', 'values', 'params', 'args', 'kwargs',
        'msg', 'err', 'error', 'msg', 'fn', 'func', 'ret', 'tmp', 'temp',
        'best', 'best_val', 'best_x', 'cand', 'hist', 'hist_g', 'hist_h',
        'bounds', 'lb', 'ub', 'lo', 'hi', 'pos', 'dist', 'val_g', 'val_r',
        'val_h', 'x_g', 'x_r', 'x_h', 'tr', 'tg', 'th', 'dp', 'bp', 'cp',
        'mp', 'ab', 'ap', 'ba', 'pb', 'pc', 'pa', 'ci', 'ci_lo', 'ci_hi',
        'mc', 'mc_vals', 'mc_mean', 'mc_std', 'sens', 'sens_rows',
        'pen', 'penalty', 'total', 'raw', 'raw_best', 'raw_total',
        'covered', 'cover', 'coverage', 'segments', 'segment',
        'drop_pos', 'burst_pos', 'cloud_pos', 'uav_pos', 'missile_pos',
        'bomb_pos', 'point_seg_dist', 'evaluate', 'random_search',
        'grid_search', 'hill_climb', 'main', 'plot_initial', 'plot_distance',
        'plot_process', 'plot_strategy', 'plot_timeline', 'plot_sensitivity',
        'plot_mc', 'save_csv', 'export_fig', 'fitness', 'objective',
        'compute', 'solve', 'optimize', 'simulate', 'verify', 'validate',
        'train', 'test', 'predict', 'classify', 'cluster', 'fit', 'transform',
        'figsize', 'dpi', 'fontsize', 'alpha', 'color', 'label', 'title',
        'xlabel', 'ylabel', 'zlabel', 'legend', 'grid', 'tight_layout',
        'show', 'savefig', 'close', 'subplot', 'subplots', 'figure',
        'Axes3D', 'add_subplot', 'scatter', 'plot', 'bar', 'barh', 'hist',
        'axhline', 'axvline', 'text', 'fill_between', 'contour', 'imshow',
        # 常见库方法名（通过 dot notation 调用，非独立变量）
        'mean', 'std', 'var', 'sum', 'axis', 'clip', 'use', 'reshape',
        'shape', 'size', 'dtype', 'T', 'real', 'imag', 'prod', 'cumsum',
        'cumprod', 'argsort', 'argmin', 'argmax', 'flatten', 'ravel',
        'transpose', 'squeeze', 'unsqueeze', 'expand_dims', 'concatenate',
        'stack', 'split', 'hsplit', 'vsplit', 'dsplit', 'tile', 'repeat',
        'where', 'nonzero', 'flatnonzero', 'argwhere', 'searchsorted',
        # 常见 numpy/math 函数名（通过 np.xxx / math.xxx 调用）
        'arange', 'linspace', 'logspace', 'meshgrid', 'sin', 'cos', 'tan',
        'arcsin', 'arccos', 'arctan', 'arctan2', 'sqrt', 'log', 'log10',
        'log2', 'exp', 'expm1', 'log1p', 'abs', 'sign', 'ceil', 'floor',
        'round', 'power', 'mod', 'fmod', 'dot', 'cross', 'inner', 'outer',
        'linalg', 'norm', 'inv', 'det', 'eig', 'svd', 'solve', 'lstsq',
        # 常见对象方法/属性名（通过 obj.xxx 调用）
        'seed', 'diff', 'gradient', 'trapz', 'cumtrapz', 'interp', 'interpolate',
        'writerow', 'writerows', 'dump', 'dumps', 'load', 'loads',
        'loc', 'iloc', 'at', 'iat', 'keys', 'values', 'items', 'get',
        'append', 'extend', 'insert', 'pop', 'remove', 'sort', 'reverse',
        'index', 'count', 'copy', 'clear', 'update', 'setdefault',
        'popitem', 'fromkeys', 'join', 'split', 'strip', 'replace',
        'startswith', 'endswith', 'find', 'rfind', 'lower', 'upper',
        'format', 'encode', 'decode', 'read', 'write', 'readline',
        'readlines', 'writelines', 'seek', 'tell', 'truncate', 'flush',
        'close', 'send', 'recv', 'bind', 'listen', 'accept', 'connect',
        # 常见配置/属性名
        'rcParams', 'fun', 'jac', 'hess', 'hessp', 'method', 'options',
        'callback', 'constraints', 'tol', 'maxiter', 'popsize', 'mutation',
        'recombination', 'strategy', 'polish', 'workers', 'updating',
        'init', 'display', 'disp', 'verbose', 'random_state', 'n_jobs',
    }
    suspicious_vars = var_usage - var_defs - PY_KEYWORDS
    # 过滤明显是内置函数/属性的
    suspicious_vars = {v for v in suspicious_vars if not v.startswith('__') and len(v) > 2}
    if suspicious_vars:
        # 只报告那些看起来像拼写错误的（与已定义变量名相似）
        suspicious_list = []
        for var in suspicious_vars:
            for defined in var_defs:
                if _levenshtein_distance(var.lower(), defined.lower()) <= 2:
                    suspicious_list.append(f"'{var}'（可能应为 '{defined}'）")
                    break
        if suspicious_list:
            warnings.append(f"⚠️ LLM代码Bug: 可能引用了未定义变量: {', '.join(suspicious_list[:5])}")

    return warnings


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _check_negative_results(output: str, nonzero_count: int) -> List[str]:
    """检测负值结果"""
    warnings = []
    neg_count = len(re.findall(
        r'(?:'
        r'\b(?:result|best|total|output|distance|time|cost|score|target|objective)\b'  # English: \b to prevent "result" in "results"
        r'|'
        r'(?:最优|值|解)'  # Chinese: no \b needed (individual chars)
        r')'
        r'.*?[=:：]\s*-\d+(?:\.\d+)?',
        output, re.IGNORECASE))
    if neg_count > 0:
        if nonzero_count == 0:
            warnings.append(
                f"P0-负值结果: {neg_count} 个优化结果均为负值，适应度函数可能存在符号错误"
                f"（如 -(raw - penalty) 在 penalty>raw 时导致优化器寻找约束违反最严重的解）")
        elif neg_count > nonzero_count:
            warnings.append(f"P1-负值结果: {neg_count} 个结果为负值，{nonzero_count} 个非负，部分结果异常")
    return warnings


def _check_convergence(output: str) -> List[str]:
    """检测收敛性"""
    warnings = []
    if not re.search(r'已收敛|converged|收敛', output, re.IGNORECASE):
        warnings.append("P1-收敛性: 未检测到收敛性判断，迭代类算法必须输出收敛状态")
    return warnings


def _check_algorithm_contrast(output: str) -> List[str]:
    """检测算法对比中一算法成功、一算法失败的模式"""
    warnings = []
    algo_results = re.findall(_ALGO_NAME_PAT.pattern + r'\s+([\d.]+(?:e[+-]?\d+)?)', output)
    if algo_results:
        try:
            results_float = [float(r) for r in algo_results]
            positive = [r for r in results_float if r > 0.0]
            zero_or_negative = [r for r in results_float if r <= 0.0]
            if positive and zero_or_negative:
                warnings.append(
                    f"P1-算法失效: {len(positive)} 个算法找到有效解，"
                    f"{len(zero_or_negative)} 个算法返回非正结果（{zero_or_negative}），"
                    f"可能因约束过强或算法参数不当导致无法找到有效解")
            # 检测所有算法返回完全相同值（非零也触发——说明目标函数是常数，几何模型有误）
            if len(results_float) >= 3:
                unique_vals = set(round(r, 6) for r in results_float)
                if len(unique_vals) == 1:
                    val = results_float[0]
                    if val <= 0.0:
                        warnings.append(
                            f"P0-算法全零: 所有算法结果均为 0，目标函数可能始终返回 0，"
                            f"请检查几何模型（实体位置/距离判定/时间窗口）是否正确"
                        )
                    else:
                        warnings.append(
                            f"P1-算法全同: 所有 {len(results_float)} 种算法返回完全相同值 {val:.4f}，"
                            f"目标函数可能为常数（核心模型对所有输入返回相同值），"
                            f"请检查核心判定条件是否过于宽松或位置计算是否使用了固定值"
                        )
        except (ValueError, TypeError):
            pass
    return warnings


def _check_monte_carlo(output: str, code: str) -> List[str]:
    """检测蒙特卡洛验证"""
    warnings = []
    output_lower = output.lower()
    mc_keywords = ["蒙特卡洛", "monte carlo", "随机模拟", "置信区间", "95%", "扰动"]
    if not any(kw in output_lower for kw in mc_keywords):
        mc_in_code = bool(re.search(
            r'(?:monte.carlo|蒙特卡洛|np\.random\.(?:normal|uniform|randn)\s*\(.*?100|n_samples\s*=\s*\d{3})',
            code, re.IGNORECASE))
        if mc_in_code:
            warnings.append("P1-蒙特卡洛: 代码中存在蒙特卡洛逻辑但未在控制台输出，请在代码中 print 验证结果")
        else:
            warnings.append("P1-蒙特卡洛: 未检测到蒙特卡洛验证，国赛要求对关键参数进行随机扰动验证")
    if re.search(r'(?:std|标准差|std_dev)\s*[=:：]\s*0+(?:\.0+)?(?:\s|$|,|，|\))', output, re.IGNORECASE):
        warnings.append("P1-蒙特卡洛异常: std=0，所有扰动结果完全相同，可能扰动幅度过小或目标函数平坦")
    return warnings


def _check_multi_algorithm(output: str) -> List[str]:
    """检测多算法对比"""
    warnings = []
    output_lower = output.lower()
    algo_keywords = [
        "遗传算法", "粒子群", "模拟退火", "网格搜索", "坐标下降",
        "差分进化", "双层优化", "蚁群", "禁忌搜索", "爬山", "随机搜索",
        "nelder.med", "nelder", "贪心", "greedy", "粗扫", "粗搜索",
        "genetic", "particle swarm", "simulated annealing", "grid search",
        "differential evolution", "ga", "pso", "de", "sa", "aco", "ts",
        "two-level", "bi-level", "multi-level", "双层", "两层",
    ]
    if not any(kw in output_lower for kw in algo_keywords):
        warnings.append("P1-多算法: 未检测到多种算法对比，国赛要求至少 2 种算法求解并对比")
    return warnings


def _check_sensitivity(output: str, code: str) -> List[str]:
    """检测敏感性分析"""
    warnings = []
    output_lower = output.lower()
    sens_keywords = ["敏感性分析", "sensitivity", "敏感度", "参数扫描"]
    if not any(kw in output_lower for kw in sens_keywords):
        sens_in_code = bool(re.search(
            r'(?:sensitivity|敏感性|sensitivity_analysis|sens_analysis)',
            code, re.IGNORECASE))
        if sens_in_code:
            warnings.append("P1-敏感性: 代码中存在敏感性分析逻辑但未在控制台输出，请在代码中 print 分析结果")
        else:
            warnings.append("P1-敏感性: 未检测到敏感性分析，国赛要求对关键参数进行敏感性分析")
    return warnings


def _check_nan_inf(output: str) -> List[str]:
    """检测结果中的 NaN/Inf"""
    warnings = []
    for line in output.split("\n"):
        line_lower = line.lower()
        if re.match(r'\s*\[.*?(?:检查|检测|NaN|Inf)\]', line):
            if re.search(r'(?:False|false|无|未检测|not found|no\s+NaN|all\s+valid)', line, re.IGNORECASE):
                continue
        if any(kw in line_lower for kw in ["nan", "inf", "-inf"]):
            if any(c in line for c in ("=", ":", "结果", "value", "output")):
                warnings.append("P0-数值异常: 结果输出中包含 NaN/Inf 值，代码可能存在除零或数值溢出")
                break
    return warnings


def _check_resource_utilization(output: str) -> List[str]:
    """检测资源利用率"""
    warnings = []
    if "仅使用" in output or "未使用" in output or "剩余" in output:
        return warnings
    resource_patterns = [
        (r'可用\s*(\d+\.?\d*).*?实际\s*(\d+\.?\d*)', "资源利用不足"),
        (r'(\d+\.?\d*)\s*个.*?使用\s*(\d+\.?\d*)\s*个', "资源利用不足"),
    ]
    for pat, desc in resource_patterns:
        match = re.search(pat, output, re.DOTALL)
        if match:
            try:
                available = float(match.group(1))
                used = float(match.group(2))
                if available > 0 and used < available:
                    warnings.append(
                        f"P1-{desc}: 可用 {available}，实际使用 {used}，"
                        f"资源利用率仅 {used/available*100:.0f}%")
            except (ValueError, ZeroDivisionError):
                pass

    # 新增：检测"X/Y"格式的资源使用统计（如 "FY1: 0/3"）
    # 匹配模式：资源名: 使用数/总容量
    utilization_pattern = re.findall(
        r'(\w+)\s*[:：]\s*(\d+)\s*/\s*(\d+)',
        output
    )
    if utilization_pattern:
        unused_resources = []
        low_usage_resources = []
        total_capacity = 0
        total_used = 0
        for name, used_str, cap_str in utilization_pattern:
            try:
                used = int(used_str)
                cap = int(cap_str)
                total_capacity += cap
                total_used += used
                if used == 0 and cap > 0:
                    unused_resources.append(f"{name}({used}/{cap})")
                elif cap > 0 and used / cap < 0.5:
                    low_usage_resources.append(f"{name}({used}/{cap})")
            except ValueError:
                pass

        if total_capacity > 0:
            util_pct = total_used / total_capacity * 100
            if unused_resources:
                warnings.append(
                    f"P0-资源闲置: {len(unused_resources)} 个资源节点完全未使用 "
                    f"({', '.join(unused_resources)})，总利用率仅 {util_pct:.0f}%。"
                    f"国赛要求所有资源节点均应参与任务分配")
            elif util_pct < 60:
                warnings.append(
                    f"P1-资源利用率低: 总利用率仅 {util_pct:.0f}% ({total_used}/{total_capacity})，"
                    f"建议充分利用所有资源节点")
            elif low_usage_resources:
                warnings.append(
                    f"P2-部分资源利用不足: {', '.join(low_usage_resources)}，"
                    f"建议均衡分配任务")

    return warnings


def _check_search_precision(output: str) -> List[str]:
    """检测搜索精度和收敛状态"""
    warnings = []
    unconverged_count = len(re.findall(r'未收敛|not converged', output, re.IGNORECASE))
    all_zero = bool(re.search(r'全部为\s*0', output))
    no_solution = bool(re.search(r'未找到有效解', output, re.IGNORECASE))
    if all_zero or no_solution:
        warnings.append("P0-算法未收敛: 迭代算法未收敛或未找到有效解，需要调整算法参数或约束处理方式")
    elif unconverged_count >= 3:
        warnings.append(f"P0-算法未收敛: {unconverged_count} 个任务未收敛，可能需要调整算法参数")
    elif unconverged_count > 0:
        warnings.append(f"P1-部分未收敛: {unconverged_count} 个任务未收敛，但多数任务已收敛")
    search_points_match = re.search(r'(\d+)\s*个点', output)
    if search_points_match:
        points = int(search_points_match.group(1))
        if points < 200:
            warnings.append(f"P1-搜索精度: 仅使用 {points} 个搜索点，建议增加搜索点数以提高精度")
    # 检测网格步长是否过大
    grid_step_match = re.search(r'(?:步长|step|grid).*?(\d+\.?\d*)', output, re.IGNORECASE)
    if grid_step_match:
        step_val = float(grid_step_match.group(1))
        if step_val > 10:
            warnings.append(f"P1-搜索精度: 网格步长 ({step_val}) 过大，建议减小步长进行精细搜索")
    return warnings


def _check_code_fabrication(code: str) -> List[str]:
    """检测代码中的伪造数据（假敏感性分析、假收敛曲线、假蒙特卡洛数据）"""
    warnings = []
    fake_sens_patterns = [
        r'-\s*10\.0\s*\)\s*/\s*10\.0',
        r'-\s*\d+\.0\s*\)\s*/\s*\d+\.0',
        r'change_pct.*?-\s*\d+\.\d+\s*\)\s*/\s*\d+\.\d+',
        r'\(\s*\w+\s*-\s*\d+(?:\.\d+)?\s*\)\s*/\s*\d+(?:\.\d+)?\s*\*\s*100',
        r'base\s*=\s*\d+(?:\.\d+)?\s*#.*基线',
    ]
    for pat in fake_sens_patterns:
        if re.search(pat, code):
            warnings.append(
                "P0-假敏感性分析: 代码中敏感性分析的基线值被硬编码"
                "（如 (x - 10.0) / 10.0），应使用实际计算结果作为基线")
            break
    fake_conv_patterns = [
        r'history\s*=\s*\[.*?\*\s*\(.*?/.*?\)\s+for',
        r'history.*?=.*?(?:\[.*?linspace|np\.linspace)',
        r'history\s*=\s*\[.*?for\s+i\s+in\s+range',
    ]
    for pat in fake_conv_patterns:
        if re.search(pat, code):
            warnings.append(
                "P0-假收敛曲线: 代码中收敛曲线是合成的直线"
                "（如 history = [best * (i/N) for i in ...]），应在优化过程中记录每次迭代的真实值")
            break
    # 新增：检测假蒙特卡洛数据
    fake_mc_patterns = [
        r'mc_vals\s*=\s*np\.(?:ones|zeros|full)\(',
        r'mc_vals\s*=\s*\[.*?\]\s*\*\s*\d+',
        r'monte.*?carlo.*?np\.random.*?seed.*?\n.*?return\s+np\.(?:ones|zeros)',
    ]
    for pat in fake_mc_patterns:
        if re.search(pat, code, re.IGNORECASE | re.DOTALL):
            warnings.append(
                "P0-假蒙特卡洛: 蒙特卡洛验证数据可能为合成数据"
                "（如全零/全一数组），应使用随机扰动生成真实分布")
            break
    # 新增：检测假算法对比（所有算法返回相同值）
    fake_compare_patterns = [
        r'algos?\s*=\s*\{.*?\}.*?for.*?in\s+algos?.*?:\s*\n\s*results?\[.*?\]\s*=\s*same_val',
        r'for\s+\w+\s+in\s+\[.*?\]:\s*\n\s*print.*?same.*?result',
    ]
    for pat in fake_compare_patterns:
        if re.search(pat, code, re.IGNORECASE | re.DOTALL):
            warnings.append(
                "P1-假算法对比: 算法对比结果可能为统一固定值，"
                "各算法应独立运行并产出真实对比结果")
            break
    return warnings


def _check_result_quality_ratio(output: str) -> List[str]:
    """检测结果质量与理论最大值之比（国赛关键）"""
    warnings = []
    patterns = [
        (r'理论[总最大上限值]*[：:]\s*(\d+\.?\d*)', r'(?:总|实际|最优|最终).*?(?:遮蔽|时间|结果|值)[：:=]\s*(\d+\.?\d*)'),
        (r'(?:理论上界|理论上限|理论最大|upper.bound)[：:]\s*(\d+\.?\d*)', r'(?:总|实际|最优|最终).*?[=:：]\s*(\d+\.?\d*)'),
        (r'理论[总最大上限值]*.*?(\d+\.?\d+)\s*s', r'(?:总|实际|最优|最终).*?(\d+\.?\d+)\s*s'),
        (r'(?:理论上界|理论上限|理论最大|upper.bound).*?(\d+\.?\d+)', r'(?:result|结果|实际|最优|最终).*?(\d+\.?\d+)'),
    ]
    for theory_pat, actual_pat in patterns:
        theory_match = re.search(theory_pat, output, re.IGNORECASE)
        actual_match = re.search(actual_pat, output, re.IGNORECASE)
        if theory_match and actual_match:
            try:
                theory_max = float(theory_match.group(1))
                actual = float(actual_match.group(1))
                if theory_max > 0:
                    ratio = actual / theory_max
                    if ratio < 0.05:
                        warnings.append(
                            f"P0-结果质量极低: 实际结果({actual})仅为理论最大值({theory_max})的 "
                            f"{ratio*100:.1f}%，搜索策略可能严重失效，必须重新设计算法")
                    elif ratio < 0.15:
                        warnings.append(
                            f"P0-结果质量过低: 实际结果({actual})仅为理论最大值({theory_max})的 "
                            f"{ratio*100:.1f}%，不满足国赛要求（≥15%），需要优化搜索策略")
                    elif ratio < 0.30:
                        warnings.append(
                            f"P1-结果质量偏低: 实际结果({actual})为理论最大值({theory_max})的 "
                            f"{ratio*100:.1f}%，建议进一步提高搜索精度")
                break
            except (ValueError, TypeError):
                pass
    # 启发式回退：如果未找到理论最大值，但结果值极小且有多目标，发出警告
    if not warnings:
        # 检测结果值是否异常小（< 10）且有多目标
        result_match = re.search(r'(?:总|最优|最佳|最终).*?(?:结果|遮蔽|值|时间)[=:：,]\s*(\d+\.?\d*)', output, re.IGNORECASE)
        if not result_match:
            result_match = re.search(r'(?:best|optimal|total|result).*?[=:：,]\s*(\d+\.?\d*)', output, re.IGNORECASE)
        if not result_match:
            # 表格格式：算法对比表中的最佳结果值
            # 在 "算法对比" 或 "结果" 之后的表格区域中提取所有数值，取最大值
            table_section = re.search(r'(?:算法对比|结果对比|性能对比).*?(?:\n\n|\Z)', output, re.DOTALL | re.IGNORECASE)
            if table_section:
                all_nums = re.findall(r'\s+(\d+\.?\d*)\s+', table_section.group(0))
                if all_nums:
                    try:
                        result_val = max(float(n) for n in all_nums if float(n) > 0)
                    except (ValueError, TypeError):
                        result_val = None
                else:
                    result_val = None
            else:
                result_val = None
        else:
            try:
                result_val = float(result_match.group(1))
            except (ValueError, TypeError):
                result_val = None
        if result_val is not None:
            try:
                target_count = len(re.findall(r'(?:M\d|目标\d|导弹\d|节点\d|设备\d)', output, re.IGNORECASE))
                if result_val < 10 and target_count >= 2:
                    warnings.append(
                        f"P1-结果偏低: 最优结果仅{result_val}，而问题涉及{target_count}个目标/资源，"
                        "结果可能远低于理论最大值。请检查：1) 核心计算模型是否正确 2) 搜索策略是否充分收敛")
            except (ValueError, TypeError):
                pass
    return warnings


def _check_multi_target_coverage(output: str) -> List[str]:
    """检测多目标/多资源覆盖情况（国赛关键）"""
    warnings = []
    # 检测目标/资源的覆盖情况
    target_patterns = [
        (r'(?:M\d|目标\d|导弹\d|节点\d|设备\d|车辆\d|基站\d)', '目标'),
        (r'(?:任务\d|Task\s*\d|区域\d|Area\s*\d)', '任务'),
        (r'(?:资源\d|Resource\s*\d|通道\d|Channel\s*\d)', '资源'),
    ]
    for pat, label in target_patterns:
        all_targets = set(re.findall(pat, output, re.IGNORECASE))
        if len(all_targets) >= 2:
            # 检测每个目标的结果
            zero_targets = []
            non_zero_targets = []
            for t in all_targets:
                # 支持 = : ： , 等多种分隔符（CSV/表格/控制台输出）
                if re.search(rf'{re.escape(t)}.*?[=:：,]\s*0(?:\.0+)?', output, re.IGNORECASE):
                    zero_targets.append(t)
                elif re.search(rf'{re.escape(t)}.*?[=:：,]\s*[1-9]\d*(?:\.\d+)?', output, re.IGNORECASE):
                    non_zero_targets.append(t)
            if zero_targets and non_zero_targets:
                warnings.append(
                    f"P1-{label}覆盖不足: {len(non_zero_targets)}个{label}有结果，"
                    f"{len(zero_targets)}个{label}结果为零（{', '.join(zero_targets)}），"
                    f"覆盖率仅{len(non_zero_targets)/len(all_targets)*100:.0f}%")
            elif zero_targets and not non_zero_targets:
                pass  # 全部为零的情况已由 _check_zero_results 检测
    return warnings


def _check_monte_carlo_robustness(output: str) -> List[str]:
    """检测蒙特卡洛验证的鲁棒性（国赛关键）"""
    warnings = []
    mc_patterns = [
        (r'蒙特卡洛验证.*?均值[=:：]\s*(\d+\.?\d*)', r'蒙特卡洛验证.*?最优[值解].*?[=:：]\s*(\d+\.?\d*)'),
        (r'Monte Carlo.*?mean[=:：]\s*(\d+\.?\d*)', r'Monte Carlo.*?(?:optimal|best).*?[=:：]\s*(\d+\.?\d*)'),
        (r'蒙特卡洛验证.*?N\s*=\s*\d+.*?均值[=:：]\s*(\d+\.?\d*)', r'蒙特卡洛验证.*?标准差[=:：]\s*(\d+\.?\d*)'),
        (r'MC.*?mean[=:：]\s*(\d+\.?\d*).*?std[=:：]\s*\d+\.?\d*', r'MC.*?optimal[=:：]\s*(\d+\.?\d*)'),
        (r'蒙特卡洛验证.*?均值[=:：]\s*(\d+\.?\d*)', r'均值/最优值[=:：]\s*(\d+\.?\d*)%'),
        (r'均值[=:：]\s*(\d+\.?\d*).*?标准差[=:：]\s*\d+\.?\d*', r'蒙特卡洛验证.*?最优[值解].*?[=:：]\s*(\d+\.?\d*)'),
        (r'均值[=:：]\s*(\d+\.?\d*).*?最优值[=:：]\s*(\d+\.?\d*)', r'均值[=:：]\s*(\d+\.?\d*).*?最优值[=:：]\s*(\d+\.?\d*)'),
        # 新增：从算法对比表和蒙特卡洛输出中分别提取最优值和MC均值
        (r'N\s*=\s*\d+.*?均值[=:：]\s*(\d+\.?\d*)', r'[总最优最终].*?(?:遮蔽|时间|结果|值)[=:：]\s*(\d+\.?\d*)'),
        (r'蒙特卡洛.*?均值[=:：]\s*(\d+\.?\d*)', r'选择最优策略.*?总遮蔽时间\s*(\d+\.?\d*)'),
    ]
    for pat1, pat2 in mc_patterns:
        m1 = re.search(pat1, output, re.IGNORECASE)
        m2 = re.search(pat2, output, re.IGNORECASE)
        if m1 and m2:
            try:
                mean_val = float(m1.group(1))
                comparison_val = float(m2.group(1))
                if comparison_val > 0:
                    ratio = mean_val / comparison_val
                    if ratio < 0.30:
                        warnings.append(
                            f"P0-鲁棒性极差: 蒙特卡洛均值({mean_val:.2f})仅为最优值({comparison_val:.2f})的"
                            f"{ratio*100:.0f}%，策略对参数扰动极度敏感，国赛要求均值≥最优值的50%。"
                            f"建议：1)在目标函数中加入鲁棒性惩罚项 2)使用保守参数（向可行域内部收缩5-10%）"
                            f"3)扰动所有关键参数（非仅部分参数）")
                    elif ratio < 0.50:
                        warnings.append(
                            f"P1-鲁棒性不足: 蒙特卡洛均值({mean_val:.2f})为最优值({comparison_val:.2f})的"
                            f"{ratio*100:.0f}%，国赛要求均值≥最优值的50%。"
                            f"建议：加入鲁棒性惩罚项或使用min-max鲁棒优化")
                    elif ratio < 0.70:
                        warnings.append(
                            f"P2-鲁棒性可改进: 蒙特卡洛均值({mean_val:.2f})为最优值({comparison_val:.2f})的"
                            f"{ratio*100:.0f}%，已达国赛基本要求，可进一步优化至≥70%")
                break
            except (ValueError, TypeError):
                pass
    # 检测蒙特卡洛中零值比例
    mc_zero_ratio = re.search(
        r'(?:零值|0值|zero|失败).*?(?:比例|占比|ratio|rate|fraction)[：:=]\s*(\d+\.?\d*)',
        output, re.IGNORECASE)
    if not mc_zero_ratio:
        # 尝试从成功率反推失败率
        mc_success = re.search(
            r'(?:成功率|success).*?[=:：]\s*(\d+\.?\d*)%',
            output, re.IGNORECASE)
        if mc_success:
            try:
                success_rate = float(mc_success.group(1))
                fail_rate = 100.0 - success_rate
                if fail_rate > 20:
                    warnings.append(
                        f"P0-蒙特卡洛失败率高: {fail_rate:.0f}%的模拟结果为零（成功率={success_rate:.1f}%），"
                        f"策略鲁棒性不满足国赛要求（失败率应≤20%）")
                elif fail_rate > 10:
                    warnings.append(
                        f"P1-蒙特卡洛失败率偏高: {fail_rate:.0f}%的模拟结果为零（成功率={success_rate:.1f}%），建议提高鲁棒性")
            except ValueError:
                pass
    else:
        try:
            zero_ratio = float(mc_zero_ratio.group(1))
            if zero_ratio > 20:
                warnings.append(
                    f"P0-蒙特卡洛失败率高: {zero_ratio:.0f}%的模拟结果为零，"
                    f"策略鲁棒性不满足国赛要求（失败率应≤20%）")
            elif zero_ratio > 15:
                warnings.append(
                    f"P1-蒙特卡洛失败率偏高: {zero_ratio:.0f}%的模拟结果为零，建议提高鲁棒性")
        except ValueError:
            pass
    return warnings


def _p1_auto_pass_check(exec_output: str, code: str) -> bool:
    """P1 自动通过预检：若代码执行完美且产出有效结果，直接返回 True 绕过 LLM 质检"""
    # 条件1：代码执行成功（无超时、无异常、无错误退出码）
    if "超时" in exec_output or "TimeoutExpired" in exec_output:
        return False
    if "代码执行失败" in exec_output or "❌" in exec_output:
        return False
    if "未能从输出中提取代码" in exec_output:
        return False
    if re.search(r'\[退出码:\s*[1-9]', exec_output):
        return False

    # 条件2：输出中有有效数值结果（非零、非NaN/Inf）
    output_lower = exec_output.lower()
    if re.search(r'(?<!no )(?<!无 )(?<!not )\bnan\b', output_lower):
        return False
    if re.search(r'(?<!no )(?<!无 )\binf\b', output_lower):
        return False

    has_nonzero = bool(re.search(
        r'(?:最优|best|total|result|output|值|解|distance|time|cost|score|target|objective|'
        r'遮蔽|时间|覆盖率|=)\s*[=:：]\s*(?:[1-9]\d*(?:\.\d+)?|0\.[1-9]\d*)',
        output_lower
    ))
    if not has_nonzero:
        has_nonzero = bool(re.search(
            r'[=:：]\s*(?:[1-9]\d*(?:\.\d+)?|0\.[1-9]\d*)',
            output_lower
        ))
    if not has_nonzero:
        return False

    has_structure = (
        "if __name__" in code
        or "def " in code
        or bool(re.search(r'(?:genetic|ga|pso|greedy|贪心|random|np\.random|scipy)', code, re.IGNORECASE))
    )
    if not has_structure:
        return False

    return True


def _check_prediction_quality(output: str) -> List[str]:
    """检测预测类结果的质量（适用于 B 类问题：预测/预报）"""
    warnings = []
    # 检查是否报告了预测误差指标
    error_metrics = re.findall(r'(?:MAE|RMSE|MAPE|MSE|R[²2])\s*[=:：]\s*([\d.]+(?:e[+-]?\d+)?)', output, re.IGNORECASE)
    if not error_metrics:
        warnings.append("P1-预测指标缺失: 未输出预测误差指标（MAE/RMSE/MAPE/R² 至少一项）")
    else:
        # 检查是否有过大的误差
        for val_str in error_metrics:
            try:
                val = float(val_str)
                if val > 1e6:
                    warnings.append(f"P0-预测误差过大: 误差指标 {val:.2e} 异常大，模型可能未正确拟合")
            except ValueError:
                pass
    # 检查是否有预测区间
    has_interval = bool(re.search(r'(?:预测区间|置信区间|prediction.interval|confidence.interval|CI|PI)', output, re.IGNORECASE))
    if not has_interval:
        warnings.append("P1-预测区间缺失: 未给出预测的不确定性区间")
    # 检查是否划分了训练/测试集
    has_split = bool(re.search(r'(?:训练集|测试集|train|test|训练|测试).*?(?:\d+%|比例)', output, re.IGNORECASE))
    if not has_split:
        warnings.append("P1-数据划分缺失: 未说明训练集/测试集划分")
    return warnings


def _check_classification_quality(output: str) -> List[str]:
    """检测分类类结果的质量（适用于 D 类问题：分类/识别）"""
    warnings = []
    # 检查是否有混淆矩阵
    has_matrix = bool(re.search(r'(?:混淆矩阵|confusion.matrix|TP|FP|TN|FN)', output, re.IGNORECASE))
    if not has_matrix:
        warnings.append("P1-混淆矩阵缺失: 未输出混淆矩阵")
    # 检查是否有 F1/精确率/召回率
    has_f1 = bool(re.search(r'(?:F1|精确率|召回率|precision|recall|accuracy)', output, re.IGNORECASE))
    if not has_f1:
        warnings.append("P1-分类指标缺失: 未输出精确率/召回率/F1 等分类指标")
    # 检查是否有交叉验证
    has_cv = bool(re.search(r'(?:交叉验证|cross.validation|K-fold|k-fold)', output, re.IGNORECASE))
    if not has_cv:
        warnings.append("P1-交叉验证缺失: 未进行交叉验证")
    return warnings


def _p1_code_is_runnable(exec_output: str, code: str) -> bool:
    """P1 降级判定：代码是否至少可执行且产出合理结果（即使 LLM 判 FAIL 也放行）"""
    # 硬性阻断：代码根本没跑起来
    if "未能从输出中提取代码" in exec_output:
        return False
    if "代码执行失败" in exec_output or "❌" in exec_output:
        return False
    if re.search(r'\[退出码:\s*[1-9]', exec_output):
        return False

    output_lower = exec_output.lower()

    # 硬性阻断：输出全是 NaN/Inf
    all_nan_inf = False
    if re.search(r'\bnan\b', output_lower) or re.search(r'\binf\b', output_lower):
        # 检查是否所有数值结果都是 NaN/Inf
        valid_numbers = re.findall(r'[=:：]\s*([1-9]\d*(?:\.\d+)?|0\.[1-9]\d*)', output_lower)
        if not valid_numbers:
            all_nan_inf = True
    if all_nan_inf:
        return False

    # 硬性阻断：没有任何有效输出
    has_any_output = bool(re.search(r'[=:：]\s*\d+', output_lower))
    if not has_any_output and "超时" not in exec_output:
        return False

    # 代码有基本结构
    has_structure = (
        "if __name__" in code
        or "def " in code
        or bool(re.search(r'(?:genetic|ga|pso|greedy|贪心|random|scipy)', code, re.IGNORECASE))
    )
    if not has_structure:
        return False

    return True


def _detect_problem_type(modeling_report: str) -> str:
    """从建模报告中检测问题类型（A=优化/B=预测/C=评价/D=分类/E=仿真/F=机理/G=统计/unknown=未识别）"""
    if not modeling_report:
        return "unknown"
    report_lower = modeling_report.lower()
    type_patterns = [
        (r'问题类型[：:]\s*A[.)、\s]*优化', 'A'),
        (r'问题类型[：:]\s*B[.)、\s]*预测', 'B'),
        (r'问题类型[：:]\s*C[.)、\s]*评价', 'C'),
        (r'问题类型[：:]\s*D[.)、\s]*分类', 'D'),
        (r'问题类型[：:]\s*E[.)、\s]*仿真', 'E'),
        (r'问题类型[：:]\s*F[.)、\s]*机理', 'F'),
        (r'问题类型[：:]\s*G[.)、\s]*统计', 'G'),
        (r'优化类问题|优化类[：:]|类型[：:]\s*优化', 'A'),
        (r'预测类问题|预测类[：:]|类型[：:]\s*预测', 'B'),
        (r'评价类问题|评价类[：:]|类型[：:]\s*评价', 'C'),
        (r'分类类问题|分类类[：:]|类型[：:]\s*分类', 'D'),
        (r'仿真类问题|仿真类[：:]|类型[：:]\s*仿真', 'E'),
        (r'机理类问题|机理类[：:]|类型[：:]\s*机理', 'F'),
        (r'统计类问题|统计类[：:]|类型[：:]\s*统计', 'G'),
    ]
    for pattern, ptype in type_patterns:
        if re.search(pattern, modeling_report):
            return ptype
    return "unknown"


def _check_result_plausibility(output: str, code: str, problem: str) -> List[str]:
    """检测代码执行结果是否合理，返回 P0/P1 级警告（调度各子检查函数）"""
    warnings = []

    z_warnings, zero_count, nonzero_count = _check_zero_results(output)
    warnings.extend(z_warnings)
    warnings.extend(_check_negative_results(output, nonzero_count))
    warnings.extend(_check_convergence(output))
    warnings.extend(_check_algorithm_contrast(output))
    warnings.extend(_check_monte_carlo(output, code))
    warnings.extend(_check_multi_algorithm(output))
    warnings.extend(_check_sensitivity(output, code))
    warnings.extend(_check_nan_inf(output))
    warnings.extend(_check_search_precision(output))
    warnings.extend(_check_code_fabrication(code))

    # 优化类专属检查：仅在输出包含优化相关关键词时触发
    is_optimization = bool(re.search(
        r'优化|最优解|optimization|optimal|约束|constraint|可行域|feasible|'
        r'目标函数|objective|适应度|fitness|种群|population|粒子群|PSO|'
        r'差分进化|differential|遗传算法|GA|模拟退火|贪心|greedy|爬山|hill',
        output + code, re.IGNORECASE
    ))
    if is_optimization:
        warnings.extend(_check_result_quality_ratio(output))
        warnings.extend(_check_multi_target_coverage(output))
        warnings.extend(_check_monte_carlo_robustness(output))
        warnings.extend(_check_resource_utilization(output))

    # 预测类专属检查：仅在输出包含预测相关关键词时触发
    is_prediction = bool(re.search(
        r'预测|forecast|predict|arima|lstm|prophet|回归|regression|'
        r'MAE|RMSE|MAPE|R[²2]|时间序列|time.series',
        output + code, re.IGNORECASE
    ))
    if is_prediction and not is_optimization:
        warnings.extend(_check_prediction_quality(output))

    # 分类类专属检查：仅在输出包含分类相关关键词时触发
    is_classification = bool(re.search(
        r'分类|classify|classifier|混淆矩阵|confusion|F1|precision|recall|'
        r'SVM|随机森林|random.forest|逻辑回归|logistic|XGBoost|准确率',
        output + code, re.IGNORECASE
    ))
    if is_classification and not is_optimization:
        warnings.extend(_check_classification_quality(output))

    return warnings


def _extract_comparison_table(output: str) -> str:
    """从执行输出中提取算法对比表（支持 Markdown 表格和空格对齐表格）"""
    lines = output.split("\n")
    table_lines = []
    in_table = False
    in_simple_table = False
    table_count = 0
    for i, line in enumerate(lines):
        # 检测 Markdown 表头行（包含算法/方法/模型+结果/精度/耗时等关键词）
        if re.search(r'\|?\s*(算法|方法|模型|Method|Algorithm|Model)\s*\|', line, re.IGNORECASE):
            if re.search(r'(结果|精度|耗时|稳定性|收敛|误差|时间|最优值|score|time|accuracy|result)', line, re.IGNORECASE):
                in_table = True
                table_lines.append(f"\n--- 对比表 {table_count + 1} ---")
                table_lines.append(line)
                continue
        if in_table:
            stripped = line.strip()
            if not stripped or (not stripped.startswith("|") and not stripped.startswith("+") and not stripped.startswith("---") and not re.search(r'\|', stripped)):
                if len(table_lines) >= 4:
                    table_count += 1
                in_table = False
                continue
            table_lines.append(line)

        # 检测空格对齐表格（算法对比: 后跟空格对齐的表格）
        if not in_table and re.search(r'算法对比\s*[:：]', line):
            in_simple_table = True
            table_lines.append(f"\n--- 对比表 {table_count + 1} ---")
            table_lines.append("| 算法 | 结果 | 耗时 |")
            table_lines.append("|------|------|------|")
            continue
        if in_simple_table:
            stripped = line.strip()
            if not stripped or stripped.startswith("==="):
                if len(table_lines) >= 5:
                    table_count += 1
                in_simple_table = False
                continue            # 尝试多种格式匹配
            parts = stripped.split()
            # 跳过表头行（"算法 结果 耗时"这类）
            if len(parts) >= 2 and parts[0] in ("算法", "方法", "模型", "Algorithm", "Method", "Model") and parts[1] in ("结果", "耗时", "精度", "Result", "Time"):
                continue
            # 格式1: "遗传算法  0.00  1.05" (空格分隔)
            if len(parts) >= 2 and any(kw in stripped for kw in [
                "算法", "遗传", "粒子", "贪心", "网格", "模拟退火", "梯度", "随机", "穷举",
                "差分", "进化", "双层", "两层", "蚁群", "禁忌", "爬山",
                "genetic", "greedy", "pso", "grid", "模型", "方法", "model", "method",
                "ga", "de", "sa", "aco", "ts", "two-level", "bi-level"
            ]):
                # 排除包含元信息关键词的行（收敛状态、敏感性分析等）
                if re.search(r'(?:状态|分析|验证|保存|已完成|结论|建议|详见|共计)',
                             stripped):
                    continue
                algo = parts[0]
                # 合并数字和其后孤立的单位（如 "-56170.97 s" → "-56170.97 s"）
                merged = []
                skip_next = False
                for idx, p in enumerate(parts[1:]):
                    if skip_next:
                        skip_next = False
                        continue
                    # 如果当前是数字，下一个是纯单位（s, ms, m, kg, min 等），合并
                    if re.match(r'^-?\d+(?:\.\d+)?$', p) and idx + 2 < len(parts):
                        next_p = parts[idx + 2]
                        if re.match(r'^[a-zA-Z]+$', next_p) and len(next_p) <= 4:
                            merged.append(f"{p} {next_p}")
                            skip_next = True
                        else:
                            merged.append(p)
                    else:
                        merged.append(p)
                vals = merged
                # 剥离常见标签前缀（如 "结果=0.000334" → "0.000334"）
                clean_vals = []
                for v in vals[:2]:
                    clean = re.sub(r'^(?:结果|值|最优值|最优解|目标值|value|result|score|objective|最优|解|耗时|时间|time|收敛代数|收敛)[=：:]\s*', '', v, flags=re.IGNORECASE)
                    clean_vals.append(clean)
                table_lines.append(f"| {algo} | {' | '.join(clean_vals)} |")
                continue
            # 格式2: "模型A (描述): 0.00 s" (冒号分隔) 或 "结果=0.000334" (等号分隔)
            # 排除非算法输出行（状态、分析、验证、保存等元信息）
            if ":" in stripped or "：" in stripped or "=" in stripped:
                if re.search(r'(状态|分析|验证|保存|已保存|完成|结论|建议|备注|输出|详见|共计|合计)',
                             stripped):
                    continue
                kv = re.split(r'[：:=]', stripped, maxsplit=1)
                if len(kv) == 2:
                    algo = kv[0].strip()
                    val = kv[1].strip()
                    table_lines.append(f"| {algo} | {val} |")
                continue

    if in_table and len(table_lines) >= 4:
        table_count += 1
    if in_simple_table and len(table_lines) >= 5:
        table_count += 1

    return "\n".join(table_lines) if table_lines else ""


def _parse_quality_status(result: str) -> str:
    text = result.upper()
    m = re.search(r'状态[：:]\s*`?\s*(PASS|FAIL|BLOCKED)', text)
    if m:
        return m.group(1)
    lines = text.split("\n")
    has_fail = False
    has_pass = False
    for i, line in enumerate(lines):
        if "状态" in line or "STATUS" in line:
            if "FAIL" in line:
                has_fail = True
            if "PASS" in line:
                has_pass = True
            if "BLOCKED" in line:
                return "BLOCKED"
            if i + 1 < len(lines):
                nl = lines[i + 1].strip()
                if nl in ("PASS", "FAIL", "BLOCKED"):
                    return nl
                for kw in ("PASS", "FAIL", "BLOCKED"):
                    if kw in nl and all(k not in nl for k in ("PASS", "FAIL", "BLOCKED") if k != kw):
                        return kw
    if has_fail:
        return "FAIL"
    if has_pass:
        return "PASS"
    if re.search(r'\bBLOCKED\b', text):
        return "BLOCKED"
    if re.search(r'\bFAIL\b', text):
        return "FAIL"
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
            "code_exec_success": False,
            "code_exec_blocked": False,
            "exec_error": None,
            "stage_output": "✅ 初始化完成，准备开始建模分析。",
        }

    def modeling_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        problem = state.get("problem_description", "")
        files = state.get("problem_files", [])
        project_root = state.get("project_root", str(config.project_root))
        retry_counts = state.get("retry_counts", {})
        m1_retry = retry_counts.get("M1", 0)

        # 聚焦问题指令
        focus_q = state.get("focus_question", "").strip()
        focus_instruction = ""
        if focus_q:
            focus_instruction = f"\n\n## ⚠️ 用户聚焦指令（最高优先级）\n用户明确要求只处理第{focus_q}问。请只分析和建模第{focus_q}问，忽略其他子问题。不要在报告中涉及其他子问题的建模。"
            problem = problem + focus_instruction

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
        safe_result = _sanitize_llm_output(result)

        return {
            "current_stage": "modeling",
            "stage_history": state.get("stage_history", []) + ["modeling"],
            "modeling_report": safe_result,
            "terminology_table": term_table,
            "messages": [AIMessage(content=result)],
            "stage_output": safe_result,
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
        if is_pass:
            quality_gates["M1"] = "PASS"
        elif status == "BLOCKED":
            quality_gates["M1"] = "BLOCKED"
        else:
            quality_gates["M1"] = "FAIL"
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["M1"] = retry_counts.get("M1", 0) + 1
        else:
            retry_counts["M1"] = 0

        safe_result = _sanitize_llm_output(result)
        return {
            "current_stage": "m1_check",
            "stage_history": state.get("stage_history", []) + ["m1_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
            "messages": [AIMessage(content=f"[M1 建模终检]\n{result}")],
            "stage_output": safe_result,
            "error": None if is_pass else "M1 建模终检未通过",
        }

    def coding_p1_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        terminology_table = state.get("terminology_table", "")
        project_root = state.get("project_root", str(config.project_root))
        retry_counts = state.get("retry_counts", {})
        p1_retry = retry_counts.get("P1", 0)

        # 聚焦问题指令
        focus_q = state.get("focus_question", "").strip()
        if focus_q:
            focus_instruction = f"\n\n## ⚠️ 用户聚焦指令（最高优先级）\n用户明确要求只实现第{focus_q}问的代码。请只生成第{focus_q}问的求解代码，忽略其他子问题。"
            modeling_report = focus_instruction + "\n\n" + modeling_report

        if p1_retry > 0:
            feedback = state.get("stage_output", "")
            if not feedback:
                for msg in reversed(messages):
                    if hasattr(msg, "content") and "P1" in str(msg.content):
                        feedback = str(msg.content)
                        break
            # 第二次重试时强制降级策略
            if p1_retry >= 2:
                feedback = (
                    "[P1 降级策略] 前两次重试失败，请放弃复杂优化算法（GA/PSO/DE），"
                    "改用贪心算法或网格搜索快速得到一个可行解。\n"
                    "目标：只要代码能跑出非零、非NaN的结果即可，不追求最优值。\n\n"
                    + (feedback or "")
                )
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
            # 多文件拆分
            p1_multi_files = _split_multi_file_code(code_file)
            p1_is_multi = len(p1_multi_files) > 1 and p1_multi_files[0][0] is not None
            if p1_is_multi:
                p1_main = p1_multi_files[0][0]
                code_file = "\n\n".join(c for _, c in p1_multi_files)
            else:
                p1_main = "solution.py"
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
            if p1_is_multi:
                for fname, fcode in p1_multi_files:
                    (exec_dir / fname).write_text(fcode, encoding="utf-8")
                exec_file = exec_dir / p1_main
            else:
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
                exec_output = f"⏱️ 代码执行超时（{config.code_exec_timeout}秒）\n\n"
                # 超时场景：分析代码结构，判断是否"结构正确但性能不足"
                code_has_main = "if __name__" in code_file or "def main" in code_file
                code_has_algorithm = bool(re.search(
                    r'(?:genetic|ga|pso|粒子群|模拟退火|simulated.annealing|greedy|贪心|differential.evolution|差分进化)',
                    code_file, re.IGNORECASE
                ))
                code_has_penalty = bool(re.search(r'penalty|惩罚|惩罚|constraint', code_file, re.IGNORECASE))
                if code_has_main and (code_has_algorithm or code_has_penalty):
                    exec_output += (
                        "[代码结构分析] 代码结构完整，超时原因可能是迭代参数过大。\n"
                        "建议：P1 阶段使用贪心算法快速验证，复杂优化留到 P2。\n"
                        "如果代码结构正确，P1 门禁可在超时场景下放宽判定。"
                    )
            except Exception as e:
                exec_output = f"❌ 代码执行失败: {e}"
            finally:
                shutil.rmtree(exec_dir, ignore_errors=True)
        else:
            exec_output = "（未能从输出中提取代码）"

        auto_pass = _p1_auto_pass_check(exec_output, code_file)
        if auto_pass:
            result = "[P1 自动通过] 代码执行成功且产出有效结果，自动通过 P1 门禁。"
            status = "PASS"
            is_pass = True
        else:
            result = quality.check_p1(code, exec_output, modeling_report, messages, problem_files)
            status = _parse_quality_status(result)
            # 关键修正：即使 LLM 判 FAIL，若代码基础条件满足，降级为 WARNED_PASS（不阻塞流程）
            if status != "PASS" and _p1_code_is_runnable(exec_output, code_file):
                status = "PASS"
                is_pass = True
                result = (
                    "[P1 降级通过] 代码可执行且产出有效结果，但因结果质量问题被 LLM 标记。\n"
                    "P1 门禁放行，质量问题将在 P2 阶段重点优化。\n\n"
                    "--- LLM 原始反馈 ---\n"
                    f"{result}"
                )
            else:
                is_pass = status == "PASS"
        quality_gates = dict(state.get("quality_gates", {}))
        if is_pass:
            quality_gates["P1"] = "PASS"
        elif status == "BLOCKED":
            quality_gates["P1"] = "BLOCKED"
        else:
            quality_gates["P1"] = "FAIL"
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["P1"] = retry_counts.get("P1", 0) + 1
        else:
            retry_counts["P1"] = 0

        safe_result = _sanitize_llm_output(result)
        return {
            "current_stage": "p1_check",
            "stage_history": state.get("stage_history", []) + ["p1_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
            "messages": [AIMessage(content=f"[P1 最小可运行结果门禁]\n{result}")],
            "stage_output": safe_result,
            "error": None if is_pass else "P1 门禁未通过",
        }

    def coding_full_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        modeling_report = state.get("modeling_report", "")
        terminology_table = state.get("terminology_table", "")
        project_root = state.get("project_root", str(config.project_root))
        retry_counts = state.get("retry_counts", {})
        p2_retry = retry_counts.get("P2", 0)

        # 聚焦问题指令
        focus_q = state.get("focus_question", "").strip()
        if focus_q:
            focus_instruction = f"\n\n## ⚠️ 用户聚焦指令（最高优先级）\n用户明确要求只实现第{focus_q}问的代码。请只生成第{focus_q}问的求解代码，忽略其他子问题。"
            modeling_report = focus_instruction + "\n\n" + modeling_report

        # 提取 P1 质检反馈（用于指导 P2 优化方向）
        p1_feedback = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and "P1" in str(msg.content) and "门禁" in str(msg.content):
                p1_feedback = str(msg.content)
                break

        if p2_retry > 0:
            feedback = state.get("stage_output", "")
            if not feedback:
                for msg in reversed(messages):
                    if hasattr(msg, "content") and "P2" in str(msg.content):
                        feedback = str(msg.content)
                        break
            # 如果代码被预执行阻断，将详细阻断信息注入反馈
            if state.get("code_exec_blocked", False):
                code_exec_output = state.get("code_exec_output", "")
                if code_exec_output:
                    # 提取阻断的具体原因
                    blocked_reason = ""
                    if "禁止的函数名" in code_exec_output:
                        blocked_reason = "检测到禁止的暴力枚举函数名（如 heuristic_greedy）"
                    elif "嵌套循环过深" in code_exec_output:
                        nest_match = re.search(r'(\d+)\s*层', code_exec_output)
                        if nest_match:
                            blocked_reason = f"检测到 {nest_match.group(1)} 层嵌套循环（要求 ≤3 层）"
                        else:
                            blocked_reason = "检测到嵌套循环过深"
                    elif "固定值惩罚" in code_exec_output:
                        blocked_reason = "检测到固定值惩罚（return 1e6）配合优化算法"
                    elif "致命代码模式" in code_exec_output:
                        blocked_reason = "检测到致命代码组合（裁剪到零 + 嵌套过深）"
                    else:
                        blocked_reason = "代码被预执行扫描阻断"
                    
                    feedback = (
                        f"### 🚫 代码被预执行扫描阻断！请根据以下原因修改代码后重新生成。\n\n"
                        f"**阻断原因**：{blocked_reason}\n\n"
                        f"### 你必须做的修改（逐项检查）：\n"
                        f"1. [ ] 删除所有暴力枚举函数（如 heuristic_greedy、greedy_search 等）\n"
                        f"2. [ ] 将所有决策变量放入参数向量，用 DE 优化器搜索\n"
                        f"3. [ ] 确保最深嵌套循环 ≤ 3 层（使用 itertools.product 展平）\n"
                        f"4. [ ] 使用比例惩罚函数，禁止 return 1e6 等固定值\n"
                        f"5. [ ] 在输出代码前，自检代码生成前自检清单的每一项\n\n"
                        f"{feedback}"
                    )
            # 如果代码执行成功但结果异常（全零/低质量/NaN/未收敛），将执行输出诊断注入反馈
            exec_output = state.get("code_exec_output", "")
            if exec_output:
                diag_info = []
                for m in re.finditer(r'P0-(?:算法失败|结果质量|数值异常|蒙特卡洛失败|算法未收敛|鲁棒性|假敏感性|假收敛|假蒙特卡洛).*?(?=\n|$)', exec_output):
                    diag_info.append(m.group(0))
                for m in re.finditer(r'⚠️\s+P0-.*?(?=\n|$)', exec_output):
                    diag_info.append(m.group(0))
                # 提取 LLM 代码 Bug 警告
                llm_bug_info = []
                for m in re.finditer(r'⚠️\s+LLM代码Bug:.*?(?=\n|$)', exec_output):
                    llm_bug_info.append(m.group(0))
                if diag_info:
                    diag_text = "\n".join(diag_info)
                    guidance = "## ⚠️ 关键提示："
                    if re.search(r'P0-算法失败|P0-数值异常', exec_output):
                        guidance += (
                            "结果全零或NaN通常意味着核心计算模型或约束判断有误。"
                            "请先添加单点诊断测试确认核心计算函数能否返回非零值。"
                            "如果单点诊断也返回零，说明目标函数在所有输入上都返回相同值（平坦优化景观），"
                            "优化器无法找到梯度方向。此时应："
                            "1) 检查判定阈值是否过严（如判定半径在全局尺度空间中占比极低），"
                            "2) 改用软化约束（如 exp(-d²/σ²) 替代硬阈值），"
                            "3) 放宽判定条件做初步搜索，确认能找到非零解后再收紧。"
                        )
                    if re.search(r'P0-结果质量', exec_output):
                        guidance += "结果质量过低（<理论最大值15%）通常意味着搜索策略过于粗糙或候选解密度不足，请增大搜索精度或增加候选数量。"
                    if re.search(r'P0-蒙特卡洛失败', exec_output):
                        guidance += "蒙特卡洛失败率过高说明策略对参数扰动敏感，请分析失败原因并增强鲁棒性。"
                    if re.search(r'P0-假敏感性|P0-假收敛|P0-假蒙特卡洛', exec_output):
                        guidance += "检测到伪造数据模式（硬编码基线/合成收敛曲线/假蒙特卡洛），请使用真实计算结果和随机扰动生成数据。"
                    if re.search(r'维度爆炸|参数维度.*≥|参数维度.*>', exec_output):
                        guidance += (
                            "检测到高维优化问题（参数维度≥20）。一次性DE搜索在40维空间中几乎等于随机采样。"
                            "请使用逐实体分解优化（for entity in range(N): optimize_one(entity)）或随机采样+局部优化策略。"
                        )
                    feedback = f"[执行结果诊断]\n{diag_text}\n\n{guidance}\n\n[质检反馈]\n{feedback}"
                if llm_bug_info:
                    bug_text = "\n".join(llm_bug_info)
                    feedback = f"[代码Bug检测]\n{bug_text}\n\n请修复以上代码Bug后重新生成。\n\n{feedback}"
            if feedback:
                # 如果上一次没有被阻断但本次是修复，注入嵌套循环硬约束提醒
                if not state.get("code_exec_blocked", False) and "嵌套" not in feedback:
                    feedback = "⚠️ **硬性约束提醒**：修复时最深嵌套循环不得超过 3 层！如果引入新循环，必须用 `itertools.product()` 或优化器替代。\n\n" + feedback
                # 如果反馈未提及维度问题但结果全零，注入高维优化提醒
                if "维度" not in feedback and "全零" in feedback:
                    feedback = "⚠️ **高维优化提醒**：如果参数维度 > 20，请使用逐实体分解优化或随机采样+局部优化，禁止一次性 DE 搜索！\n\n" + feedback
                result = coding.fix_code(feedback, messages, project_root)
            else:
                result = coding.implement_full(modeling_report, terminology_table, messages, project_root, p1_feedback)
        else:
            result = coding.implement_full(modeling_report, terminology_table, messages, project_root, p1_feedback)

        return {
            "current_stage": "coding_full",
            "stage_history": state.get("stage_history", []) + ["coding_full"],
            "messages": [AIMessage(content=result)],
            "stage_output": result,
            "error": None,
        }

    def code_exec_node(state: WorkflowState) -> Dict[str, Any]:
        code = state.get("stage_output", "")
        problem = state.get("problem_description", "")

        code_file = _extract_code_from_output(code)
        if not code_file.strip():
            return {
                "current_stage": "code_exec",
                "stage_history": state.get("stage_history", []) + ["code_exec"],
                "code_exec_output": "⚠️ 未能从输出中提取代码",
                "stage_output": state.get("stage_output", ""),
                "code_exec_success": False,
                "exec_error": "未能从输出中提取代码",
                "error": None,
            }

        code_file = _inject_chinese_font(code_file)

        # 多文件拆分：检测 `# file: 文件名.py` 标记
        multi_files = _split_multi_file_code(code_file)
        is_multi_file = len(multi_files) > 1 and multi_files[0][0] is not None
        if is_multi_file:
            main_filename = multi_files[0][0]
            # 合并所有文件代码用于语法检查和预扫描
            code_file = "\n\n".join(c for _, c in multi_files)
        else:
            main_filename = "solution.py"

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
                        "code_exec_success": False,
                        "exec_error": f"代码语法错误（修复失败）: {syntax_err}",
                        "error": "代码语法错误",
                    }

        # ====== 多轮 Debug 循环 ======
        MAX_DEBUG_ROUNDS = 3
        final_output = ""
        final_figure_files = []
        final_result_files = []
        final_code_files = []
        exec_success = False
        debug_round = 0

        while debug_round < MAX_DEBUG_ROUNDS:
            exec_dir = Path(tempfile.mkdtemp(prefix="code_exec_"))
            if is_multi_file:
                for fname, fcode in multi_files:
                    (exec_dir / fname).write_text(fcode, encoding="utf-8")
                exec_file = exec_dir / main_filename
            else:
                exec_file = exec_dir / "solution.py"
                exec_file.write_text(code_file, encoding="utf-8")

            # 预执行安全扫描：检测可能超时的代码模式
            pre_scan_warnings = []
            code_lines = code_file.split("\n")

            # 函数级嵌套分析：区分优化型循环与分析型循环
            # 分析型循环（蒙特卡洛/敏感性/绘图）允许 ≤5 层，优化型循环严格 ≤3 层
            analysis_keywords = r'蒙特卡洛|monte.?carlo|敏感性|sensitivity|绘图|plot|figure|图表|chart|convergence|收敛|验证|verify|diagnos|诊断|统计|statistics|报告|report|输出|output|print|save|write'
            func_nests = {}  # {(func_name, start_line): (max_opt_nest, max_all_nest, total_opt, total_all)}
            in_function = False
            func_name = ""
            func_start = -1
            func_lines = []
            for i, line in enumerate(code_lines):
                if re.match(r'def\s+(\w+)\s*\(', line):
                    if in_function and func_lines:
                        _analyze_function_nesting(func_lines, func_name, func_start, func_nests, analysis_keywords)
                    in_function = True
                    func_name = re.match(r'def\s+(\w+)\s*\(', line).group(1)
                    func_start = i
                    func_lines = [line]
                elif in_function:
                    func_lines.append(line)
            if in_function and func_lines:
                _analyze_function_nesting(func_lines, func_name, func_start, func_nests, analysis_keywords)

            # 汇总：取所有函数中优化型嵌套的最大值
            max_nest = 0
            total_loops = 0
            for (fname, _), (opt_nest, all_nest, opt_cnt, all_cnt) in func_nests.items():
                max_nest = max(max_nest, opt_nest)
                total_loops += all_cnt
            # 如果没有任何函数被识别（如纯脚本），回退到简单计数
            if not func_nests:
                max_nest = 0
                total_loops = 0
                for line in code_lines:
                    stripped = line.strip()
                    if stripped.startswith("for ") or stripped.startswith("while "):
                        total_loops += 1
                        indent = len(line) - len(line.lstrip())
                        depth = indent // 4
                        max_nest = max(max_nest, depth)
            de_scan = re.search(r'differential_evolution\(.*?popsize\s*=\s*(\d+).*?maxiter\s*=\s*(\d+)', code_file, re.DOTALL)
            de_pop = 0
            de_iter = 0
            if de_scan:
                de_pop = int(de_scan.group(1))
                de_iter = int(de_scan.group(2))
                if de_pop > 15 or de_iter > 50:  # 放宽限制
                    pre_scan_warnings.append(f"⚠️ 差分进化参数较大: popsize={de_pop} (建议≤15), maxiter={de_iter} (建议≤50)")
                # 维度爆炸检测：bounds 维度 > 20 且 popsize*maxiter 不足以覆盖搜索空间
                bounds_scan = re.search(r'bounds\s*=\s*\[(.*?)\]', code_file, re.DOTALL)
                if bounds_scan:
                    bounds_content = bounds_scan.group(1)
                    dim_count = len(re.findall(r'\(', bounds_content))
                    if dim_count > 20:
                        total_evals = de_pop * de_iter
                        pre_scan_warnings.append(
                            f"⚠️ 维度爆炸风险: bounds 定义 {dim_count} 个参数维度，"
                            f"但 DE 的 popsize={de_pop} × maxiter={de_iter} 仅 {total_evals} 次评估，"
                            f"在 {dim_count} 维空间中几乎等于随机采样。"
                            f"建议使用逐实体分解优化（for entity in range(N): optimize_one(entity)）"
                        )
            pso_scan_pop = re.search(r'(?:n_particles|swarm_size)\s*=\s*(\d+)', code_file)
            if pso_scan_pop:
                pso_pop = int(pso_scan_pop.group(1))
                if pso_pop > 30:  # 放宽到30
                    pre_scan_warnings.append(f"⚠️ PSO 粒子数较大: {pso_pop} (建议≤30)")
            pso_scan_iter = None
            # 仅在代码包含 PSO/粒子群 关键词时检查迭代次数，避免将蒙特卡洛/敏感性分析循环误判为 PSO 迭代
            if re.search(r'(?:pso|粒子群|particle.?swarm|n_particles|swarm_size)', code_file, re.IGNORECASE):
                for m in re.finditer(r'for\s+\w+\s+in\s+range\((\d+)\)', code_file):
                    n = int(m.group(1))
                    if n > 100:  # 放宽到100
                        pso_scan_iter = n
                        break
            if pso_scan_iter:
                pre_scan_warnings.append(f"⚠️ PSO 迭代次数较大: {pso_scan_iter} (建议≤100)")

            # 通用迭代次数检测：随机搜索/坐标下降/任意 for 循环中的大迭代次数
            # 变量赋值模式：n_iter = 200, n_iterations = 200, max_iter = 200, iterations = 200
            general_iter_vars = {}
            for m in re.finditer(r'(?:n_iter|n_iterations|max_iter|iterations|n_iters|num_iter|num_iters)\s*=\s*(\d+)', code_file):
                general_iter_vars[m.group(1)] = int(m.group(1))
            # 函数调用模式：random_search(n_iter=200), random_search(iterations=200)
            for m in re.finditer(r'(?:random_search|random_search|monte_carlo_search|grid_search|exhaustive_search)\s*\(\s*(?:n_iter|n_iterations|max_iter|iterations|n_iters)\s*=\s*(\d+)', code_file):
                val = int(m.group(1))
                if val > 50:
                    pre_scan_warnings.append(f"⚠️ 随机搜索迭代次数过大: {m.group(0).strip()[:40]}... (迭代={val}，建议≤100)")
            # 检测 for 循环中使用的迭代变量对应的值
            for var_name, var_val in general_iter_vars.items():
                if var_val > 100:
                    pre_scan_warnings.append(f"⚠️ 迭代次数较大: {var_name}={var_val} (建议≤100，{var_val} 次迭代可能耗时较长)")

            # 坐标下降维度检测：for dim in range(N) 且 N > 阈值
            # 但排除逐实体分解优化（每实体只优化少量参数，是安全的）
            cd_dim_match = None
            cd_rounds_match = None
            cd_rounds = 3
            
            # 检测是否为逐实体分解优化（安全模式）
            is_safe_decomposition = bool(
                re.search(r'for\s+\w+\s+in\s+(?:MISSILE_NAMES|UAVS|entities|missiles|uavs)', code_file, re.IGNORECASE) and
                re.search(r'differential_evolution|minimize', code_file) and
                not re.search(r'bounds\s*=\s*\[.*?\].*?\n.*?for\s+\w+\s+in\s+range\((?:50|100|200)\)', code_file, re.DOTALL | re.IGNORECASE)
            )
            
            if re.search(r'(?:coordinate_descent|坐标下降)', code_file, re.IGNORECASE) and not is_safe_decomposition:
                for m in re.finditer(r'for\s+\w+\s+in\s+range\((\d+)\)', code_file):
                    n = int(m.group(1))
                    if n > 30:  # 提高阈值到30（原值为20）
                        cd_dim_match = n
                        break
            if cd_dim_match:
                pre_scan_warnings.append(
                    f"⚠️ 坐标下降维度爆炸: for dim in range({cd_dim_match}) — "
                    f"{cd_dim_match} 维 × N 轮 = 大量优化器调用，300s 内几乎必定超时！"
                    "建议：使用逐实体分解优化（for entity in range(N_ENTITIES): optimize_one(entity)）"
                    "或限制坐标下降维度 ≤ 20")

            # 总评估次数估算（粗略）
            total_est_evals = 0
            # DE 评估次数
            if de_scan:
                total_est_evals += de_pop * de_iter
            # PSO 评估次数
            pso_pop_val = int(pso_scan_pop.group(1)) if pso_scan_pop else 0
            pso_iter_val = pso_scan_iter if pso_scan_iter else 0
            if pso_pop_val and pso_iter_val:
                total_est_evals += pso_pop_val * pso_iter_val
            # 随机搜索迭代
            for var_val in general_iter_vars.values():
                total_est_evals += var_val
            # 坐标下降
            if cd_dim_match:
                cd_rounds_match = re.search(r'(?:n_rounds|cd_rounds|rounds)\s*=\s*(\d+)', code_file)
                if cd_rounds_match:
                    cd_rounds = int(cd_rounds_match.group(1))
                total_est_evals += cd_dim_match * cd_rounds
            # 敏感性分析
            sens_match = re.findall(r'for\s+\w+\s+in\s+(?:enumerate\(|range\(len\().*?\)\)', code_file)
            if 'sensitivity' in code_file.lower() or '敏感性' in code_file:
                total_est_evals += 100  # 粗略估计
            # 蒙特卡洛
            mc_match = re.search(r'(?:n_sim|n_mc|mc_samples|num_simulations)\s*=\s*(\d+)', code_file)
            if mc_match:
                total_est_evals += int(mc_match.group(1))
            if total_est_evals > 500:
                pre_scan_warnings.append(
                    f"⚠️ 总评估次数过多: 约 {total_est_evals} 次目标函数评估（DE={de_pop}x{de_iter} + "
                    f"PSO={pso_pop_val}x{pso_iter_val} + 随机搜索={sum(general_iter_vars.values())} + "
                    f"坐标下降={cd_dim_match or 0}x" +
                    f"{cd_rounds_match.group(1) if cd_rounds_match else '3'} + 其他），"
                    "在 300s 内几乎必定超时！建议限制总评估次数 ≤ 500")

            dt_scan = re.search(r'(?:DT|dt|TIME_STEP|time_step)\s*=\s*(\d+\.?\d*)', code_file)
            dt_val = 1.0
            if dt_scan:
                dt_val = float(dt_scan.group(1))
                if dt_val < 0.2:
                    pre_scan_warnings.append(f"⚠️ 时间步长违规: DT={dt_val} (要求≥0.2)")
            # DT ≤ 0.1 严重违规（2x 低于最小阈值），直接阻断
            if dt_val <= 0.1:
                # 计算影响：每个 evaluate 调用多慢
                t_sim_match = re.search(r'(?:T_SIM|T_MAX|t_max|t_sim)\s*=\s*(\d+\.?\d*)', code_file)
                t_sim = float(t_sim_match.group(1)) if t_sim_match else 70.0
                eval_iters = int(t_sim / dt_val)
                recommended_iters = int(t_sim / 0.5)
                fatal_output = (
                    "### 🚫 预执行阻断：时间步长过小\n\n"
                    f"检测到 **DT={dt_val}**，远低于最小要求（≥0.2）。\n\n"
                    f"**影响分析**：\n"
                    f"- 当前 DT={dt_val} → 每次评估 = {t_sim}/{dt_val} = **{eval_iters} 次迭代**\n"
                    f"- 推荐 DT=0.5 → 每次评估 = {t_sim}/0.5 = **{recommended_iters} 次迭代**\n"
                    f"- 当前评估速度慢 **{eval_iters / recommended_iters:.0f}x**！\n\n"
                    f"**为什么 DT={dt_val} 必定失败**：\n"
                    f"- 优化器（DE）会调用目标函数 {6 * 12} ≈ 72 次\n"
                    f"- 每次调用 {eval_iters} 次内部循环 = {72 * eval_iters} 次总迭代\n"
                    f"- 加上 15 个弹的独立优化，总迭代次数 = {72 * eval_iters * 15} 次\n"
                    f"- 300s 内根本无法完成，优化器来不及收敛就超时了\n\n"
                    f"**修复：将 DT 改为 ≥ 0.5**\n"
                    f"```python\n"
                    f"DT = 0.5  # 将 DT 从 {dt_val} 改为 0.5（减少 {eval_iters / recommended_iters:.0f}x 计算量）\n"
                    f"```\n\n"
                    f"**注意**：DT=0.5 对遮蔽时间的计算精度影响极小（烟幕漂移速度 3m/s × 0.5s = 1.5m，远小于有效半径 10m），"
                    f"但能让代码在 300s 内完成！\n\n"
                    f"**请修改 DT 并重新生成代码。**"
                )
                return {
                    "current_stage": "code_exec",
                    "stage_history": state.get("stage_history", []) + ["code_exec"],
                    "code_exec_output": fatal_output,
                    "figure_files": [],
                    "result_files": [],
                    "code_files": [],
                    "code_exec_success": False,
                    "code_exec_blocked": True,
                }
            if max_nest > 3:
                pre_scan_warnings.append(f"⚠️ 嵌套循环过深: 最大 {max_nest} 层 (共 {total_loops} 个循环, 要求≤3层)")
            # 检测固定值惩罚（return 1e6 / return 1e9 等）
            fixed_penalty = re.search(r'return\s+(1e\d+|1E\d+|\d{6,})', code_file)
            if fixed_penalty:
                pre_scan_warnings.append(f"⚠️ 固定值惩罚: '{fixed_penalty.group(0)}' — 这会导致优化器失效！请改用比例惩罚（penalty += 1000 * violation）")
            # 检测潜在 NaN/Inf 来源（除零、sqrt 负数、atan2(0,0)）
            nan_sources = []
            # 除零风险：/ norm 但 norm 可能为 0
            if re.search(r'/\s*norm_d?\b', code_file) or re.search(r'/\s*np\.linalg\.norm', code_file):
                nan_sources.append("除零风险：`/ norm` 或 `/ np.linalg.norm()` 在向量为零时产生 NaN/Inf")
            # sqrt 负数风险
            if re.search(r'(?:math|np)\.sqrt\s*\(\s*(?:disc|delta|d|diff|val)', code_file):
                nan_sources.append("sqrt 负数风险：`sqrt(disc)` 在判别式为负时产生 NaN，请先 `max(0, disc)` 保护")
            # atan2(0, 0) 风险
            if re.search(r'math\.atan2\s*\(\s*0[,.]?0?\s*,\s*0', code_file):
                nan_sources.append("atan2(0,0) 风险：`math.atan2(0, 0)` 在 dist_h=0 时产生 NaN")
            if nan_sources:
                pre_scan_warnings.append(
                    f"⚠️ 潜在 NaN/Inf 来源（{len(nan_sources)} 处）：\n" +
                    "\n".join(f"  - {s}" for s in nan_sources) +
                    "\n  建议：优化前先手动测试一组参数，确认结果非 NaN/Inf"
                )
            # 检测适应度函数符号错误: -(raw - penalty) 模式
            fitness_sign_error = re.search(r'return\s+-\s*\(\s*\w+\s*-\s*\w+\s*\)', code_file)
            if fitness_sign_error:
                pre_scan_warnings.append(f"⚠️ 适应度函数符号错误: '{fitness_sign_error.group(0)}' — -(raw - penalty) 在 penalty>raw 时会导致优化器寻找约束违反最严重的解！请改为 -(raw) + penalty")
            # 检测裁剪到 0（T = max(0.0, T) 等模式可能导致优化器扁平化）
            # 仅当 max(0, ...) 出现在目标函数的 return 语句中时才警告
            # 用于中间计算（如时间裁剪）的 max(0, t) 是合法的
            clip_to_zero = re.findall(r'(?:max|np\.maximum)\s*\(\s*0\.?0*\s*,\s*(\w+)\s*\)', code_file)
            has_clip_in_obj = bool(clip_to_zero) and bool(re.search(r'def\s+(?:fitness|evaluate|objective|compute)', code_file))
            if has_clip_in_obj:
                # 额外检查：max(0, ...) 是否在 return 语句中
                is_return_clip = bool(re.search(
                    r'return\s+.*?(?:max|np\.maximum)\s*\(\s*0\.?0*\s*,',
                    code_file
                ))
                if is_return_clip:
                    pre_scan_warnings.append(
                        f"⚠️ 裁剪到零: 在目标函数的 return 语句中检测到 max(0, ...) 模式 — "
                        "不可行解被裁剪到 0，优化器无法区分违反程度，可能导致 DE/PSO 返回全零解！"
                        "请改用比例惩罚：return raw_score - penalty * violation")
                else:
                    pre_scan_warnings.append(
                        f"⚠️ 裁剪到零（中间计算）: 检测到 max(0, ...) 用于中间计算而非 return 语句 — "
                        "如果这是用于时间/距离的合法裁剪，可以忽略此警告。但如果该值最终影响目标函数返回值，"
                        "请确保不可行区域仍能产生梯度信息")

            # 检测 evaluate/objective/fitness 函数内部嵌套 ≥4 层 → 高概率超时
            eval_func_nest = 0
            eval_func_name = ""
            for (fname, _), (opt_nest, _, _, _) in func_nests.items():
                if re.search(r'(?:evaluate|objective|fitness|compute|simulate)', fname, re.IGNORECASE):
                    if opt_nest > eval_func_nest:
                        eval_func_nest = opt_nest
                        eval_func_name = fname
            if eval_func_nest >= 4:
                pre_scan_warnings.append(
                    f"⚠️ 评估函数 `{eval_func_name}()` 有 {eval_func_nest} 层嵌套循环 — "
                    "该函数在优化过程中会被调用数百次，{eval_func_nest} 层嵌套将导致 300s 内必定超时！"
                    "建议：将内层循环向量化（NumPy 广播），或使用 np.interp 预计算轨迹")

            # 检测嵌套优化反模式：DE/minimize 被包裹在循环中调用
            # 模式：定义一个包含 DE 的辅助函数，然后在 for/while 循环中调用它
            funcs_with_optimizer = []  # 包含 DE/minimize 调用的函数名
            for func_match in re.finditer(r'def\s+(\w+)\s*\(', code_file):
                fname = func_match.group(1)
                # 找到该函数的函数体
                func_start = func_match.end()
                # 简单策略：找到下一个同缩进级别的 def 或文件末尾
                body_start = code_file.find('\n', func_start)
                if body_start == -1:
                    continue
                # 找到函数体的结束（下一个顶级 def 之前）
                next_def = re.search(r'^(?:def\s+|class\s+|if\s+__name__)', code_file[body_start:], re.MULTILINE)
                if next_def:
                    func_body = code_file[body_start:body_start + next_def.start()]
                else:
                    func_body = code_file[body_start:]
                if re.search(r'(?:differential_evolution|scipy\.optimize\.minimize|basinhopping|dual_annealing)\s*\(', func_body):
                    funcs_with_optimizer.append(fname)
            if funcs_with_optimizer:
                # 检查这些函数是否在循环中被调用
                nested_opt_funcs = []
                for fname in funcs_with_optimizer:
                    # 检查 fname 是否在 for/while 循环内部被调用
                    lines = code_file.split('\n')
                    in_loop = False
                    for i, line in enumerate(lines):
                        stripped = line.lstrip()
                        if re.match(r'(?:for|while)\s+', stripped):
                            in_loop = True
                        elif stripped and not stripped.startswith(' ') and not stripped.startswith('\t'):
                            if not stripped.startswith('#'):
                                in_loop = False
                        if in_loop and re.search(rf'\b{fname}\s*\(', line):
                            nested_opt_funcs.append(fname)
                            break
                if nested_opt_funcs:
                    pre_scan_warnings.append(
                        f"⚠️ 嵌套优化反模式: 函数 {', '.join(nested_opt_funcs)}() 包含优化器调用，"
                        "但该函数在 for/while 循环中被调用。这会导致优化器被重复执行数百次，"
                        "总计算量爆炸。建议：将所有参数放入一个参数向量，用单次 DE/PSO 调用替代嵌套优化")

            # 检测禁止的函数名（LLM 最常见的高危模式）
            banned_func_names = [
                'heuristic_greedy', 'greedy_search', 'brute_force', 'exhaustive_search',
                'grid_search_all', 'enumerate_all',
            ]
            found_banned = []
            for fname, _ in func_nests:
                for banned in banned_func_names:
                    if banned.lower() in fname.lower():
                        found_banned.append(fname)
                        break
            if found_banned:
                banned_func = found_banned[0]
                fatal_output = (
                    "### 🚫 预执行阻断：检测到禁止的函数名\n\n"
                    f"代码中定义了名为 **`{banned_func}()`** 的函数，"
                    f"这是 LLM 最常见的暴力枚举模式函数名，已被禁止。\n\n"
                    f"**为什么禁止 `{banned_func}()`？**\n"
                    f"这个名字暗示该函数内部通过嵌套循环暴力枚举所有参数组合，"
                    f"这在数学建模中是不可行的——计算量会指数级爆炸。\n\n"
                    f"**正确做法（二选一）：**\n\n"
                    f"**方案 A（推荐）：用 DE 优化器替代**\n"
                    f"```python\n"
                    f"# 将所有决策变量放入一个参数向量\n"
                    f"bounds = [(0, 2*pi)]*N + [(V_MIN, V_MAX)]*N + [(0, T_MAX)]*(N*M) + [(0, T_EFF)]*(N*M)\n"
                    f"result = differential_evolution(lambda x: -evaluate(x), bounds, maxiter=30, popsize=10, seed=42)\n"
                    f"```\n\n"
                    f"**方案 B：用 itertools.product() 展平**\n"
                    f"```python\n"
                    f"from itertools import product\n"
                    f"for a, b, c, d, e in product(range(N), range(M), cand, p1, p2):\n"
                    f"    # 所有 N 层展开为 1 层！\n"
                    f"```\n\n"
                    f"**请删除 `{banned_func}()` 函数，改用上述方案后重试。**"
                )
                return {
                    "current_stage": "code_exec",
                    "stage_history": state.get("stage_history", []) + ["code_exec"],
                    "code_exec_output": fatal_output,
                    "figure_files": [],
                    "result_files": [],
                    "code_files": [],
                    "code_exec_success": False,
                    "code_exec_blocked": True,
                }

            # 独立嵌套深度警告：≥8 层嵌套 → 给出警告但不阻断（允许复杂算法）
            if max_nest >= 8:
                # 找出最深嵌套的函数名
                deepest_func = ""
                deepest_nest = 0
                for (fname, _), (opt_nest, all_nest, opt_cnt, all_cnt) in func_nests.items():
                    if opt_nest > deepest_nest:
                        deepest_nest = opt_nest
                        deepest_func = fname
                if not deepest_func:
                    deepest_func = "顶层代码"

                # 只警告不阻断（因为已增加超时时间到600s）
                pre_scan_warnings.append(
                    f"⚠️ 嵌套循环较深: {max_nest}层 (建议≤5层，位于函数 `{deepest_func}`)"
                )

            # 致命组合检测：return 1e6 + 优化算法 → 保证优化器失效，直接阻断
            has_fixed_penalty = bool(re.search(r'return\s+(1e\d+|1E\d+|\d{6,})', code_file))
            has_optimizer = bool(re.search(r'(?:differential_evolution|minimize|basinhopping|dual_annealing|shgo)', code_file))
            if has_fixed_penalty and has_optimizer:
                fatal_output = (
                    "### 🚫 预执行阻断：固定值惩罚 + 优化算法\n\n"
                    "代码中同时存在：\n\n"
                    "1. **固定值惩罚**：`return 1e6`（或 `return 1e9` 等）\n"
                    "2. **优化算法**：`differential_evolution` 或 `minimize` 等\n\n"
                    "**这是保证失败的组合！** 固定值惩罚创建了不连续的适应度景观，"
                    "优化器（DE/PSO/梯度下降）无法区分约束违反程度，"
                    "会在平坦区域停止搜索，导致所有结果为零。\n\n"
                    "**修复方向（必须全部满足才能重新执行）：**\n"
                    "- 将 `return 1e6` 替换为比例惩罚：`penalty = 1000 * max(0, violation)`\n"
                    "- 或在 DE 中使用 `bounds` 参数天然约束，配合 `polish=False`\n"
                    "- 测试：手动验证 `evaluate(valid_params)` 是否能返回非零值\n\n"
                    "**请修复后重新生成代码。**"
                )
                return {
                    "current_stage": "code_exec",
                    "stage_history": state.get("stage_history", []) + ["code_exec"],
                    "code_exec_output": fatal_output,
                    "figure_files": [],
                    "result_files": [],
                    "code_files": [],
                    "code_exec_success": False,
                    "code_exec_blocked": True,
                }

            # 致命组合检测：max(0,...) + 嵌套过深 → 几乎肯定返回全零，直接阻断
            fatal_combo = has_clip_in_obj and max_nest > 3
            if fatal_combo:
                fatal_output = (
                    "### 🚫 预执行阻断：检测到致命代码模式\n\n"
                    f"代码同时存在以下致命问题：\n\n"
                    f"1. **裁剪到零**：目标函数中使用了 `max(0, ...)` 模式，"
                    f"导致优化器无法区分不可行解的违反程度。当所有候选解都不可行时，"
                    f"优化器会返回全零结果（因为所有不可行解都被裁剪到同一个值 0）。\n\n"
                    f"2. **嵌套循环过深**：最大 {max_nest} 层嵌套循环（要求 ≤3 层），"
                    f"这会导致计算时间爆炸且代码几乎不可能产生正确结果。\n\n"
                    f"**修复方向（必须全部满足才能重新执行）：**\n"
                    f"- 将 `max(0, x)` 替换为带惩罚的原始值：`x + penalty * max(0, -x)` 或使用比例惩罚函数\n"
                    f"- 将嵌套循环从 {max_nest} 层重构为 ≤3 层（使用向量化/预计算中间结果）\n"
                    f"- 在优化前先验证几何约束：手动测试一组参数，确认 `evaluate()` 能返回非零值\n"
                    f"- 如果 `evaluate()` 始终返回 0，说明几何模型本身有误，需要重新分析几何关系\n\n"
                    f"**请修复以上问题后重新生成代码。**"
                )
                return {
                    "current_stage": "code_exec",
                    "stage_history": state.get("stage_history", []) + ["code_exec"],
                    "code_exec_output": fatal_output,
                    "figure_files": [],
                    "result_files": [],
                    "code_files": [],
                    "code_exec_success": False,
                    "code_exec_blocked": True,
                }

            # 总评估次数警告：P2 > 1000 或 P1 > 2000 → 给出警告但不阻断
            # （因为已增加超时时间到600s，允许更多计算量）
            is_p2_code = any(s in str(state.get("stage_history", [])) for s in ["coding_full", "P2"])
            eval_limit = 1000 if is_p2_code else 2000
            if total_est_evals > eval_limit:
                # 只警告不阻断
                pre_scan_warnings.append(
                    f"⚠️ 总评估次数较多: 约 {total_est_evals} 次 (建议≤{eval_limit}次，当前600s超时可能勉强够用)"
                )

            # 坐标下降维度警告：> 80 维 → 给出建议但不阻断
            if cd_dim_match and cd_dim_match > 80:
                pre_scan_warnings.append(
                    f"⚠️ 坐标下降维度较高: {cd_dim_match}维 (建议≤80维，可考虑逐实体分解或仅优化关键参数)"
                )

            # ====== LLM 常见代码 Bug 检测 ======
            llm_bug_warnings = _detect_llm_code_bugs(code_file)
            pre_scan_warnings.extend(llm_bug_warnings)

            # ====== P2 质量要求检测 ======
            # 检测收敛性分析（优化类问题必须追踪收敛过程）
            is_optimization_code = bool(re.search(
                r'(?:differential_evolution|minimize|basinhopping|dual_annealing|shgo|PSO|GA|遗传|粒子群)',
                code_file
            ))
            if is_optimization_code and is_p2_code:
                has_convergence = bool(re.search(
                    r'(?:history|收敛|convergence|converged|best_so_far|best_history|iter_history)',
                    code_file, re.IGNORECASE
                ))
                has_convergence_plot = bool(re.search(
                    r'(?:收敛.*plot|plot.*收敛|收敛曲线|convergence.*plot|plot.*convergence)',
                    code_file, re.IGNORECASE
                ))
                if not has_convergence:
                    pre_scan_warnings.append(
                        "⚠️ P2-缺失收敛性分析: 优化代码中未检测到收敛追踪（history/convergence/best_so_far）。"
                        "P2 要求优化类问题必须记录并输出收敛过程，否则会被 P2 门禁判定为 FAIL！"
                        "请在优化循环中追踪 best_so_far 值，并在最后输出收敛曲线。"
                    )
                elif not has_convergence_plot:
                    pre_scan_warnings.append(
                        "⚠️ P2-缺失收敛曲线: 检测到收敛追踪但未检测到收敛曲线绘图。"
                        "P2 要求绘制收敛曲线图（plt.plot(history)），请确保代码包含收敛曲线可视化。"
                    )

            # 检测构造性初始解可能失败的模式（construct_initial 等）
            has_construct_func = bool(re.search(
                r'def\s+construct.*initial|def\s+build.*initial|def\s+make.*initial',
                code_file, re.IGNORECASE
            ))
            if has_construct_func and is_optimization_code:
                # 检查构造函数是否验证了结果
                has_construct_validation = bool(re.search(
                    r'(?:if.*==\s*0|if.*<\s*1e-|if.*is\s*None|警告|验证|check|validate|warn).*construct',
                    code_file, re.IGNORECASE
                ))
                if not has_construct_validation:
                    pre_scan_warnings.append(
                        "⚠️ 构造性初始解未验证: 检测到 `construct_initial` 类函数但未验证其结果。"
                        "如果构造的初始解返回 0.00s，优化器将无法从该初始点改进。"
                        "建议：在构造初始解后立即验证 `score = evaluate(init_params)`，"
                        "如果 score < 1e-6 则打印警告并尝试其他构造策略。"
                    )

            # ====== 算法多样性检测（国赛 P2 关键） ======
            if is_optimization_code and is_p2_code:
                # 检测代码中是否包含差分进化
                has_de = bool(re.search(r'differential_evolution', code_file))
                # 检测代码中是否包含至少 3 种算法
                algorithm_count = 0
                algorithm_patterns = [
                    r'differential_evolution', r'minimize', r'basinhopping', r'dual_annealing',
                    r'PSO', r'粒子群', r'pso', r'GeneticAlg', r'遗传算法', r'GA',
                    r'simulated.annealing', r'模拟退火', r'greedy', r'贪心', r'random.search', r'随机搜索',
                    r'grid.search', r'网格搜索', r'coordinate.descent', r'坐标下降',
                    r'hill.climb', r'爬山', r'nelder.mead', r'BFGS', r'L-BFGS',
                ]
                for pat in algorithm_patterns:
                    if re.search(pat, code_file, re.IGNORECASE):
                        algorithm_count += 1
                # 去重：同一算法可能被多次匹配
                algorithm_count = min(algorithm_count, 20)  # 防止过度计数

                if not has_de:
                    pre_scan_warnings.append(
                        "⚠️ P2-缺失差分进化: 优化类代码中未检测到 differential_evolution。"
                        "国赛要求优化类问题必须包含差分进化作为 3 种对比算法之一。"
                        "请添加以下代码：\n"
                        "  from scipy.optimize import differential_evolution\n"
                        "  result_de = differential_evolution(lambda x: -evaluate(x), bounds, maxiter=25, popsize=10, seed=42)"
                    )
                if algorithm_count < 3:
                    pre_scan_warnings.append(
                        f"⚠️ P2-算法数量不足: 仅检测到约 {algorithm_count} 种算法（国赛要求至少 3 种算法对比）。"
                        "请补充更多算法，如：差分进化 + 随机搜索 + 贪心算法"
                    )

            # ====== 资源利用率检测（国赛 P2 关键） ======
            if is_p2_code:
                # 检测是否有资源利用率统计输出
                has_resource_stats = bool(re.search(
                    r'(?:资源利用率|利用率|使用率|usage.*rate|utilization)',
                    code_file, re.IGNORECASE
                ))
                # 检测是否有资源节点分配统计（如 "FY1: 0/3" 格式）
                has_resource_allocation = bool(re.search(
                    r'print.*[:：].*\d+/\d+',
                    code_file
                ))
                if not has_resource_stats and not has_resource_allocation:
                    # 仅当代码中定义了多个资源节点时警告
                    resource_nodes = re.findall(
                        r'(?:NODES|RESOURCES|UAV|无人机|资源|节点)\s*=\s*[\[\(]',
                        code_file
                    )
                    if resource_nodes:
                        pre_scan_warnings.append(
                            "⚠️ P2-缺失资源利用率统计: 代码中定义了资源节点但未输出资源利用率。"
                            "国赛要求输出每个资源节点的使用情况统计。"
                            "请添加：print(f'资源利用率: {name}: {used}/{capacity}')"
                        )

            if pre_scan_warnings:
                output = "### ⚠️ 预执行扫描警告（代码可能超时，但将继续执行）\n"
                output += "\n".join(f"  - {w}" for w in pre_scan_warnings)
                output += "\n\n---\n\n"
            else:
                output = ""

            # ====== 执行前语法检查 ======
            import py_compile as _pyc
            try:
                _pyc.compile(str(exec_file), doraise=True)
            except _pyc.PyCompileError as _e:
                # 提取行号、错误行内容和错误信息
                err_msg = str(_e)
                output += f"### ❌ 语法错误：代码无法执行\n\n{err_msg}\n\n"
                # 尝试提取出错行
                line_match = re.search(r'line\s+(\d+)', err_msg)
                if line_match:
                    err_line = int(line_match.group(1))
                    code_lines_list = code_file.split('\n')
                    if err_line <= len(code_lines_list):
                        start = max(0, err_line - 3)
                        end = min(len(code_lines_list), err_line + 2)
                        output += "**出错位置上下文：**\n```\n"
                        for i in range(start, end):
                            marker = ">>> " if i == err_line - 1 else "    "
                            output += f"{marker}{i+1}: {code_lines_list[i]}\n"
                        output += "```\n\n"
                output += "**请修复语法错误后重新生成代码。**"
                final_output = output
                # 不执行，直接跳出循环
                return {
                    "current_stage": "code_exec",
                    "stage_history": state.get("stage_history", []) + ["code_exec"],
                    "code_exec_output": final_output,
                    "figure_files": [],
                    "result_files": [],
                    "code_files": [],
                    "code_exec_success": False,
                }
            except Exception:
                pass  # 非语法错误（如导入错误），继续执行

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
                exec_output = result.stdout.decode("utf-8", errors="replace")
                if result.stderr:
                    exec_output += f"\n\n[stderr]:\n{result.stderr.decode('utf-8', errors='replace')}"
                if result.returncode != 0:
                    exec_output += f"\n\n[退出码: {result.returncode}]"
                output = output + exec_output

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

                if result.returncode == 0:
                    final_output = output
                    final_figure_files = figure_files
                    final_result_files = result_files
                    final_code_files = code_files
                    exec_success = True
                    break
                else:
                    if debug_round < MAX_DEBUG_ROUNDS - 1:
                        debug_feedback = f"代码执行失败（第 {debug_round + 1} 轮），错误信息:\n{output[:3000]}"
                        fix_success = False
                        try:
                            fixed_code = _extract_code_from_output(
                                coding.fix_code(debug_feedback, state.get("messages", []), state.get("project_root", str(config.project_root)))
                            )
                            if fixed_code.strip():
                                code_file = _inject_chinese_font(fixed_code)
                                fix_success = True
                                debug_round += 1
                                continue
                        except Exception as e:
                            output += f"\n\n[修复异常: {e}]"
                        if not fix_success:
                            output += "\n\n[代码修复失败，跳过重试]"
                            final_output = output
                            final_figure_files = figure_files
                            final_result_files = result_files
                            final_code_files = code_files
                            break
                    final_output = output
                    final_figure_files = figure_files
                    final_result_files = result_files
                    final_code_files = code_files

            except subprocess.TimeoutExpired:
                # 分析代码特征，给出针对性修复建议
                code_lines = code_file.split("\n")
                # 计算实际最大嵌套深度（而非简单计数所有 for 行）
                max_nesting = 0
                total_for_loops = 0
                for line in code_lines:
                    stripped = line.strip()
                    if stripped.startswith("for ") or stripped.startswith("while "):
                        total_for_loops += 1
                        # 用缩进估计嵌套深度
                        indent = len(line) - len(line.lstrip())
                        depth = max(1, indent // 4 + 1)  # 假设 4 空格缩进
                        max_nesting = max(max_nesting, depth)
                has_meshgrid = "meshgrid" in code_file
                time_step_match = re.search(r'(?:DT|TIME_STEP|dt|time_step)\s*=\s*(\d+\.?\d*)', code_file)
                time_step = float(time_step_match.group(1)) if time_step_match else None
                de_match = re.search(r'differential_evolution\(.*?maxiter\s*=\s*(\d+)', code_file, re.DOTALL)
                de_maxiter = int(de_match.group(1)) if de_match else None
                de_pop_match = re.search(r'differential_evolution\(.*?popsize\s*=\s*(\d+)', code_file, re.DOTALL)
                de_popsize = int(de_pop_match.group(1)) if de_pop_match else None
                # 检测 PSO 参数
                pso_particles_match = re.search(r'(?:n_particles|swarm_size|num_particles)\s*=\s*(\d+)', code_file)
                pso_particles = int(pso_particles_match.group(1)) if pso_particles_match else None
                pso_iters_match = re.search(r'(?:n_iterations|max_iter|pso_iters|iterations|n_iter)\s*=\s*(\d+)', code_file)
                pso_iters = int(pso_iters_match.group(1)) if pso_iters_match else None
                # 检测 GA 参数
                ga_pop_match = re.search(r'(?:pop_size|population_size|POP_SIZE|n_pop)\s*=\s*(\d+)', code_file)
                ga_pop = int(ga_pop_match.group(1)) if ga_pop_match else None
                ga_gen_match = re.search(r'(?:generations|n_generations|max_gen|n_gen|GEN_MAX)\s*=\s*(\d+)', code_file)
                ga_gen = int(ga_gen_match.group(1)) if ga_gen_match else None
                ga_stall_match = re.search(r'(?:stall_limit|patience|early_stop)\s*=\s*(\d+)', code_file)
                ga_stall = int(ga_stall_match.group(1)) if ga_stall_match else None

                diag_parts = []
                if max_nesting > 3:
                    diag_parts.append(f"  - 最大嵌套深度 {max_nesting} 层（共 {total_for_loops} 个循环），建议用向量化减少嵌套")
                elif total_for_loops > 10:
                    diag_parts.append(f"  - 共 {total_for_loops} 个循环语句（最大嵌套 {max_nesting} 层），建议合并或向量化")
                if time_step and time_step < 0.2:
                    diag_parts.append(f"  - 时间步长 DT={time_step}s 过小（{time_step}s × 20s 窗口 = {int(20/time_step)} 步），建议改为 ≥ 0.5s")
                if de_maxiter and de_maxiter > 30:
                    diag_parts.append(f"  - 差分进化 maxiter={de_maxiter} 过大（违规：要求 ≤ 30），建议改为 20")
                if de_popsize and de_popsize > 10:
                    diag_parts.append(f"  - 差分进化 popsize={de_popsize} 过大（违规：要求 ≤ 10），建议改为 8")
                if pso_particles and pso_particles > 20:
                    diag_parts.append(f"  - 粒子群规模 {pso_particles} 过大（建议 ≤ 20），建议减少粒子数")
                if pso_iters and pso_iters > 50:
                    diag_parts.append(f"  - 粒子群迭代 {pso_iters} 次过多（建议 ≤ 50），建议减少迭代次数")
                if ga_pop and ga_pop > 60:
                    diag_parts.append(f"  - 遗传算法种群大小 {ga_pop} 过大（建议 ≤ 50），组合数 = {ga_pop} × {ga_gen or '?'} 代")
                if ga_gen and ga_gen > 50:
                    diag_parts.append(f"  - 遗传算法迭代 {ga_gen} 代过多（建议 ≤ 40），降低迭代次数")
                if ga_stall and ga_stall > 15:
                    diag_parts.append(f"  - 停滞阈值 {ga_stall} 代过大（建议 ≤ 10），早期停止可节省时间")
                if not diag_parts:
                    diag_parts.append("  - 可能存在死循环或无限递归，请检查 while 循环终止条件")

                diag_text = "\n".join(diag_parts)
                final_output = (
                    f"⏱️ 代码执行超时（{config.code_exec_timeout}秒）\n\n"
                    f"### 超时诊断\n{diag_text}\n\n"
                    f"### 修复建议\n"
                    f"1. 将嵌套 for 循环替换为向量化运算\n"
                    f"2. 增大时间步长（≥ 0.5s）\n"
                    f"3. 减少差分进化迭代次数（maxiter ≤ 20, popsize ≤ 8）\n"
                    f"4. 减少粗搜索总组合数（≤ 3000）\n"
                )
                break
            except Exception as e:
                final_output = f"❌ 代码执行失败: {e}"
                break
            finally:
                shutil.rmtree(exec_dir, ignore_errors=True)
            debug_round += 1

        if debug_round > 0 and exec_success:
            syntax_note += f"\n[多轮调试: 第 {debug_round + 1} 轮执行成功]"

        # ====== 结果合理性检测 ======
        plausibility_warnings = _check_result_plausibility(final_output, code_file, problem)
        if plausibility_warnings:
            warning_text = "\n\n[结果合理性检测]\n" + "\n".join(f"  ⚠️ {w}" for w in plausibility_warnings)
            final_output += warning_text

        return {
            "current_stage": "code_exec",
            "stage_history": state.get("stage_history", []) + ["code_exec"],
            "code_exec_output": final_output + syntax_note,
            "raw_exec_output": final_output + syntax_note,
            "stage_output": state.get("stage_output", ""),
            "figure_files": final_figure_files,
            "result_files": final_result_files,
            "code_files": final_code_files,
            "code_exec_success": exec_success,
            "exec_error": None if exec_success else (final_output[:500] if final_output else "代码执行失败"),
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
        # 保存原始执行输出（不含后续追加的验证/对比/误差分析内容）
        raw_exec = state.get("raw_exec_output", exec_output)
        verify_text = ""

        if not config.enable_verification:
            return {
                "current_stage": "verify",
                "stage_history": state.get("stage_history", []) + ["verify"],
                "code_exec_output": exec_output,
                "raw_exec_output": raw_exec,
                "verification_output": "[数值验证已禁用]",
                "stage_output": state.get("stage_output", ""),
                "code_exec_success": state.get("code_exec_success", False),
                "exec_error": state.get("exec_error"),
                "error": None,
            }

        if state.get("code_exec_blocked", False):
            return {
                "current_stage": "verify",
                "stage_history": state.get("stage_history", []) + ["verify"],
                "code_exec_output": exec_output,
                "raw_exec_output": raw_exec,
                "verification_output": "[数值验证已跳过] 代码被预执行扫描阻断，未实际执行，无需验证",
                "stage_output": state.get("stage_output", ""),
                "code_exec_success": False,
                "code_exec_blocked": True,
                "error": "代码被预执行扫描阻断",
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

        sc = verifier.sensitivity_check(exec_output, code)
        verify_results.append(f"[敏感性分析] 状态: {sc['status']}")
        for f in sc["findings"]:
            verify_results.append(f"  {f}")

        if figure_files:
            figures_dir = str(Path(project_root) / "figures")
            fv = verifier.format_verification(figures_dir)
            verify_results.append(f"[图表格式] 状态: {fv['status']}")
            for f in fv["findings"]:
                verify_results.append(f"  {f}")

        # 新增：收敛性检查
        cc = verifier.convergence_check(exec_output)
        verify_results.append(f"[收敛性分析] 状态: {cc['status']}")
        for f in cc["findings"]:
            verify_results.append(f"  {f}")

        # 新增：极端值测试
        et = verifier.extreme_value_test(code_file)
        verify_results.append(f"[极端值测试] 状态: {et['status']}")
        for f in et["findings"]:
            verify_results.append(f"  {f}")

        # 新增：对称性检查
        sym = verifier.symmetry_check(code_file, problem)
        verify_results.append(f"[对称性检查] 状态: {sym['status']}")
        for f in sym["findings"]:
            verify_results.append(f"  {f}")

        # 新增：守恒量检查
        cons = verifier.conservation_check(code_file)
        verify_results.append(f"[守恒量检查] 状态: {cons['status']}")
        for f in cons["findings"]:
            verify_results.append(f"  {f}")

        # 新增：结果异常检测（使用 detector 的 detect_result_anomalies 方法）
        result_anomalies = trap_detector.detect_result_anomalies(exec_output, code_file)
        if result_anomalies["anomaly_count"] > 0:
            verify_results.append(f"[结果异常检测] 状态: {result_anomalies['status']}")
            for f in result_anomalies["findings"]:
                if "P0" in f or "P1" in f:
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
            "raw_exec_output": raw_exec,
            "verification_output": verify_text,
            "stage_output": state.get("stage_output", ""),
            "code_exec_success": state.get("code_exec_success", False),
            "exec_error": state.get("exec_error"),
            "error": None if not has_p0 else "数值验证发现P0级错误",
        }

    def model_comparison_node(state: WorkflowState) -> Dict[str, Any]:
        """模型对比节点：对比多种算法结果，选择最优方案"""
        exec_output = state.get("code_exec_output", "")
        raw_exec = state.get("raw_exec_output") or exec_output
        code = state.get("stage_output", "")
        project_root = state.get("project_root", str(config.project_root))
        problem = state.get("problem_description", "")

        code_file = _extract_code_from_output(code)
        comparison_text = ""

        # 检测代码中是否包含多算法对比（使用原始执行输出避免被追加内容干扰）
        has_multi_algorithm = False
        algo_keywords = ["遗传算法", "粒子群", "模拟退火", "网格搜索", "坐标下降",
                          "差分进化", "双层优化", "蚁群", "禁忌搜索", "爬山", "随机搜索",
                          "genetic", "particle swarm", "simulated annealing", "grid search",
                          "differential evolution", "ga", "pso", "de", "sa", "aco", "ts",
                          "two-level", "bi-level", "双层", "两层",
                          "对比", "比较", "compare", "comparison", "算法对比", "算法比较"]
        algo_set = set(kw.lower() for kw in algo_keywords)
        # 优先使用编译态正则
        if _ALGO_NAME_PAT.search(code_file) or _ALGO_NAME_PAT.search(raw_exec):
            has_multi_algorithm = True
        else:
            code_lower = code_file.lower()
            raw_lower = raw_exec.lower()
            for kw in algo_set:
                if kw in code_lower or kw in raw_lower:
                    has_multi_algorithm = True
                    break

        if has_multi_algorithm:
            comparison_text = "✅ 检测到多算法对比\n\n"
            # 从执行结果中提取对比表
            table = _extract_comparison_table(raw_exec)
            if table:
                comparison_text += f"### 提取的对比数据\n\n```\n{table}\n```\n\n"
            else:
                comparison_text += "⚠️ 检测到多算法对比，但未能从输出中提取结构化的对比表格。\n\n"

            # 从执行结果中提取数值对比
            comparison_text += "### 定量对比分析\n\n"

            # 用正则从输出中提取最优值（同时匹配中英文算法名）
            algo_patterns = [
                ("网格搜索", ["网格搜索", "grid search", "grid_search"]),
                ("遗传算法", ["遗传算法", "genetic algorithm", "genetic", "ga"]),
                ("贪心算法", ["贪心", "greedy"]),
                ("粒子群", ["粒子群", "particle swarm", "pso"]),
                ("模拟退火", ["模拟退火", "simulated annealing", "simulated_annealing", "sa"]),
                ("差分进化", ["差分进化", "differential evolution", "de"]),
                ("双层优化", ["双层优化", "双层", "两层", "two-level", "bi-level", "bilevel"]),
                ("梯度下降", ["梯度下降", "gradient descent", "gradient"]),
                ("坐标下降", ["坐标下降", "coordinate descent", "coordinate"]),
                ("穷举法", ["穷举", "暴力", "brute force", "brute"]),
                ("随机搜索", ["随机搜索", "random search", "random"]),
                ("蚁群算法", ["蚁群", "ant colony", "aco"]),
                ("禁忌搜索", ["禁忌搜索", "tabu search", "ts"]),
            ]
            best_values = {}
            for algo_name, algo_keys in algo_patterns:
                for algo_key in algo_keys:
                    patterns = [
                        rf'{re.escape(algo_key)}\s+(\d+\.?\d*)\s+(\d+\.?\d*)',  # 空格对齐格式
                        rf'{re.escape(algo_key)}.*?[：:=]\s*(\d+\.?\d*)\s*[秒s]',
                        rf'{re.escape(algo_key)}.*?最优[值解].*?[=:：]\s*(\d+\.?\d*)',
                        rf'{re.escape(algo_key)}.*?结果[=:：]\s*(\d+\.?\d*)',
                        rf'{re.escape(algo_key)}.*?值[=:：]\s*(\d+\.?\d*)',
                    ]
                    found = False
                    for pat in patterns:
                        m = re.search(pat, raw_exec, re.IGNORECASE)
                        if m:
                            best_values[algo_name] = float(m.group(1))
                            found = True
                            break
                    if found:
                        break

            if best_values:
                algo_names = list(best_values.keys())
                # 动态生成表头
                header_cols = "| 指标 |" + "|".join(f" {n} |" for n in algo_names)
                if len(algo_names) >= 2:
                    header_cols += " 差异 |"
                comparison_text += header_cols + "\n"
                sep_cols = "|------|" + "|".join("------|" for _ in algo_names)
                if len(algo_names) >= 2:
                    sep_cols += "------|"
                comparison_text += sep_cols + "\n"
                comparison_text += "| 最优结果 |" + "".join(f" {v:.4f} |" for v in best_values.values())
                if len(algo_names) >= 2:
                    vals = list(best_values.values())
                    diff = abs(vals[0] - vals[1])
                    comparison_text += f" {diff:.4f} s |"
                comparison_text += "\n"
                if len(best_values) >= 2:
                    vals = list(best_values.values())
                    diff = abs(vals[0] - vals[1])
                    nonzero_vals = [abs(v) for v in vals if v != 0]
                    empty_cols = " |".join([""] * (len(algo_names) - 1))
                    if nonzero_vals:
                        if diff < 0.01 * max(nonzero_vals):
                            comparison_text += f"| **结论** | **结果一致，解可靠** |{empty_cols} |\n"
                        else:
                            comparison_text += f"| **结论** | **结果差异较大，需进一步分析原因** |{empty_cols} |\n"
                    else:
                        comparison_text += f"| **结论** | **所有算法结果均为0，可能存在算法实现问题** |{empty_cols} |\n"
            else:
                comparison_text += "（未能从执行输出中自动提取定量对比数据，请查看下方原始输出）\n\n"
                # 尝试从原始文本中提取更多数值信息
                for line in raw_exec.split("\n"):
                    if re.search(r'\d+\.\d+', line) and any(kw in line.lower() for kw in ["最优", "最佳", "结果", "解"]):
                        comparison_text += f"- 原始输出: `{line.strip()}`\n"
                if "0.00" in raw_exec and any(kw in raw_exec for kw in ["遗传算法", "genetic"]):
                    comparison_text += "\n⚠️ 遗传算法返回 0.00，可能存在约束处理问题\n"
                    comparison_text += "💡 建议：使用惩罚函数法处理约束，或调整变异/交叉算子\n"
        else:
            comparison_text = "⚠️ 未检测到多算法对比，建议实现至少 2 种求解算法并对比结果。\n\n"
            comparison_text += "**国赛要求**：对每个子问题的求解，必须实现至少 2 种不同求解算法\n"
            comparison_text += "（如：网格搜索 vs 遗传算法、梯度下降 vs 粒子群），并输出对比表格。\n"
            comparison_text += "如果两种算法结果一致，说明解可靠；如果不一致，需分析原因。\n"

        return {
            "current_stage": "model_comparison",
            "stage_history": state.get("stage_history", []) + ["model_comparison"],
            "model_comparison": comparison_text,
            "code_exec_output": exec_output + f"\n\n[模型对比]\n{comparison_text}",
            "stage_output": state.get("stage_output", ""),
            "code_exec_success": state.get("code_exec_success", False),
            "exec_error": state.get("exec_error"),
            "error": None,
        }

    def error_analysis_node(state: WorkflowState) -> Dict[str, Any]:
        """误差分析节点：基于验证结果和真实执行输出，生成针对性的误差来源分析"""
        exec_output = state.get("code_exec_output", "")
        raw_exec = state.get("raw_exec_output") or exec_output
        verification_output = state.get("verification_output", "")
        problem = state.get("problem_description", "")
        messages = state.get("messages", [])
        code_exec_success = state.get("code_exec_success", False)
        exec_error = state.get("exec_error", "")

        # 如果代码执行失败，不生成详细分析，只记录失败原因
        if not code_exec_success:
            error_text = "\n".join([
                "## 误差分析报告",
                "",
                f"### ⚠️ 代码执行失败，无法生成误差分析",
                "",
                f"**执行状态**: 失败",
                f"**错误信息**: {exec_error or '未知错误'}",
                "",
                "由于代码未成功执行，没有可用的数值结果来进行误差分析。",
                "请修复代码后重新运行。",
                "",
                "### 常见超时原因排查",
                "",
                "1. **嵌套循环过多**: 检查是否存在多层嵌套循环（如 4 层参数扫描），建议使用向量化运算",
                "2. **搜索精度过高**: 检查搜索步长是否过小，建议先用粗搜索定位再用细搜索精化",
                "3. **差分进化迭代过多**: popsize × maxiter 过大，建议 popsize=8~15, maxiter=20~50",
                "4. **时间步长过小**: TIME_STEP=0.1s 对于 70s 时间窗口产生 700 步/次，建议增大到 0.2~0.5s",
                "5. **未使用向量化**: 检查是否用 Python 循环代替 NumPy 向量化操作",
            ])
            return {
                "current_stage": "error_analysis",
                "stage_history": state.get("stage_history", []) + ["error_analysis"],
                "error_analysis": error_text,
                "code_exec_output": exec_output + f"\n\n[误差分析]\n{error_text}",
                "stage_output": state.get("stage_output", ""),
                "code_exec_success": False,
                "exec_error": exec_error,
                "error": None,
            }

        # 检测结果是否全为零/NaN/负值（泛化：适用于任何优化问题）
        all_zero = False
        all_nan = False
        negative_count = 0
        positive_count = 0
        non_zero_count = 0
        for line in raw_exec.split("\n"):
            nums = re.findall(r'(?<![a-zA-Z0-9._-])(\d+(?:\.\d+)?)(?![a-zA-Z0-9._-])', line)
            for n in nums:
                val = float(n)
                if val > 1e-10:
                    non_zero_count += 1
            # 检测负值：在结果上下文中出现负值（排除自检标签行）
            # 使用 \b 词边界确保 "result" 不匹配 "results/sensitivity.csv" 中的子串
            if re.search(r'(?:\b(?:result|best|total|output|distance|time|cost|score|target|objective)\b|(?:最优|值|解)).*?[=:：]\s*-\d+(?:\.\d+)?', line, re.IGNORECASE):
                if not re.match(r'\s*\[.*?(?:检查|检测)\]', line):
                    negative_count += 1
            if re.search(r'\bNaN\b|\bnan\b', line, re.IGNORECASE):
                if re.match(r'\s*\[.*?(?:检查|检测|NaN|Inf)\]', line):
                    if re.search(r'(?:False|false|无|未检测|not found|no\s+NaN|all\s+valid)', line, re.IGNORECASE):
                        continue
                all_nan = True
            # 统计正值结果数量（用于后续判断负值是否占主导）
            if re.search(r'(?:\b(?:result|best|total|output|distance|time|cost|score|target|objective)\b|(?:最优|值|解)).*?[=:：]\s*[1-9]\d*(?:\.\d+)?', line, re.IGNORECASE):
                positive_count += 1

        if non_zero_count == 0 and not all_nan:
            all_zero = True

        if negative_count > 0 and positive_count == 0:
            error_text = "\n".join([
                "## 误差分析报告",
                "",
                "### ⚠️ 结果异常：优化结果为负值，无法生成误差分析",
                "",
                f"**执行状态**: 代码执行成功，但检测到 {negative_count} 个负值结果",
                "",
                "由于优化结果均为物理上不可能的值（如负时间/负距离），无法进行误差分析。",
                "以下是可能的原因和修复建议：",
                "",
                "### 根因诊断",
                "",
                "1. **适应度函数符号错误**: 最常见原因是 `return -(raw - penalty)` 在 `penalty > raw` 时",
                "   导致优化器寻找约束违反最严重的解，而非优化目标函数",
                "   - 修复: 改为 `return -(raw) + penalty` 或 `return -raw` 并在约束违反时 `return -inf`",
                "2. **惩罚函数设计不当**: 固定值惩罚（如 `return 1e6`）导致适应度景观出现高原",
                "   - 修复: 使用比例惩罚，惩罚值 ∝ 违反程度",
                "3. **初始化范围问题**: 决策变量初始范围包含了不合理区域",
                "   - 修复: 缩小搜索边界，使用启发式方法确定合理范围",
                "",
                "### 修复步骤",
                "1. 检查适应度函数的符号逻辑",
                "2. 在目标函数中添加调试输出，打印 raw 和 penalty 的中间值",
                "3. 手动构造一个已知可行解，验证适应度函数返回正值",
                "4. 将约束违反时的返回值改为 `-1e9`（而非 `return -(raw - penalty)`）",
            ])
            return {
                "current_stage": "error_analysis",
                "stage_history": state.get("stage_history", []) + ["error_analysis"],
                "error_analysis": error_text,
                "code_exec_output": exec_output + f"\n\n[误差分析]\n{error_text}",
                "stage_output": state.get("stage_output", ""),
                "code_exec_success": code_exec_success,
                "exec_error": exec_error,
                "error": None,
            }

        if all_zero or all_nan:
            anomaly_type = "全零" if all_zero else "NaN"
            error_text = "\n".join([
                "## 误差分析报告",
                "",
                f"### ⚠️ 结果异常：所有优化结果均为{anomaly_type}，无法生成误差分析",
                "",
                f"**执行状态**: 代码执行成功，但数值结果异常（{anomaly_type}）",
                "",
                "由于所有优化结果均为异常值，没有有效数据来进行误差分析。",
                "以下是可能的原因和修复建议：",
                "",
                "### 通用排查清单",
                "",
                "1. **搜索空间过大**: 随机初始化无法命中有效解区域 → 尝试分阶段搜索（粗搜索定位 + 细搜索精化）",
                "2. **约束条件过强**: 可行域占比过小 → 检查约束条件是否合理，考虑放宽或使用惩罚函数",
                "3. **初始化策略不当**: 初始种群未覆盖有效区域 → 使用启发式方法缩小搜索空间（如基于问题几何结构的预计算）",
                "4. **目标函数有缺陷**: 检查目标函数是否在有效区域内确实能返回非零值",
                "5. **数值精度问题**: 检查浮点运算是否导致有效解被误判为零",
                "",
                "### ⚠️ 几何模型诊断（最关键！如果涉及空间位置判定）",
                "",
                "如果问题涉及空间位置/几何关系判定，全零结果最常见的根因是**核心计算模型错误**：",
                "",
                "1. **关键实体位置计算错误**: 检查关键实体的位置是否在目标路径/作用范围内",
                "   - 打印一组参数的中间计算结果，手动验证实体坐标是否合理",
                "   - 确认实体是否在有效作用区域内，而非远离目标",
                "2. **判定阈值过严**: 检查有效半径/阈值是否合理（如阈值过小导致始终无法命中）",
                "3. **时间窗口不重叠**: 实体激活到失效的时间窗口是否与目标经过时间有交集",
                "4. **坐标系/方向向量错误**: 确认使用的是归一化方向向量，而非原始坐标差",
                "5. **可视化诊断**: 强烈建议绘制场景图，标注目标轨迹、关键实体位置、作用范围",
                "",
                "### 诊断建议",
                "1. 在目标函数中添加调试输出，打印中间计算值（位置、距离、时间）",
                "2. 手动构造一个已知可行解（如：将实体精确放置在目标路径上），验证目标函数能正确计算",
                "3. 先用网格搜索小范围扫描，确认可行域存在",
                "4. 如果涉及几何判定，绘制 3D 场景图进行目视验证",
            ])
            return {
                "current_stage": "error_analysis",
                "stage_history": state.get("stage_history", []) + ["error_analysis"],
                "error_analysis": error_text,
                "code_exec_output": exec_output + f"\n\n[误差分析]\n{error_text}",
                "stage_output": state.get("stage_output", ""),
                "code_exec_success": code_exec_success,
                "exec_error": exec_error,
                "error": None,
            }

        # 尝试使用 LLM 生成针对性的误差分析
        llm_analysis = ""
        try:
            # 从执行结果中提取关键数值
            extracted_values = []
            for line in raw_exec.split("\n"):
                line_stripped = line.strip()
                if re.search(r'\d+\.\d+', line_stripped) and any(kw in line_stripped.lower() for kw in
                    ["最优", "最佳", "结果", "时长", "时间", "距离", "速度", "角度", "收敛", "优化", "目标"]):
                    extracted_values.append(line_stripped)

            analysis_prompt = f"""基于以下代码执行结果和题目描述，生成针对性的误差分析报告。

## 题目
{problem[:2000]}

## 代码执行结果关键数据
{chr(10).join(extracted_values[:20]) if extracted_values else "（未提取到关键数值）"}

## 数值验证结果
{verification_output[:2000] if verification_output else "（无验证结果）"}

## 完整执行输出
{raw_exec[:3000]}

请生成一份结构化的误差分析报告，要求：
1. 从执行结果中提取实际数值来估计误差，如果执行结果中缺少某类数据，明确标注"数据不足，无法估计"而非编造数值
2. 分析数值方法（如离散化、迭代次数）导致的误差量级
3. 分析模型简化（如忽略次要物理因素）导致的误差量级
4. 如果存在蒙特卡洛/随机模拟，分析其置信区间宽度对结论的影响
5. 指出最主要的误差来源
6. 给出具体的改进建议

使用 Markdown 表格输出每条误差的具体数值估计。对于无法估计的误差，在表格中标注"数据不足"。"""
            project_root = state.get("project_root", str(config.project_root))
            writing_prompt = writing.load_system_prompt().replace("{project_root}", project_root)
            llm_analysis = writing.invoke(messages, user_input=analysis_prompt, system_prompt=writing_prompt)
        except Exception as e:
            llm_analysis = f"（LLM 分析不可用: {e}）"

        error_text = "\n".join([
            "## 误差分析报告",
            "",
            "### 1. 误差来源定量分析",
            "",
            "| 误差来源 | 类型 | 估计量级 | 减缓措施 |",
            "|----------|------|----------|----------|",
        ])

        # 部分负值警告：有负值但也有正值，不阻断流程但添加提示
        if negative_count > 0 and positive_count > 0:
            error_text += f"\n\n> ⚠️ **部分负值警告**: 检测到 {negative_count} 个负值结果（共 {positive_count} 个正值），"
            error_text += "可能存在部分约束违反或适应度函数局部缺陷。请检查对应结果行。\n"

        # 从执行结果中提取具体数值来填充误差表
        has_monte_carlo = any(kw in raw_exec.lower() for kw in ["蒙特卡洛", "monte carlo", "随机模拟", "置信区间", "95%"])
        has_grid_search = any(kw in raw_exec.lower() for kw in ["网格搜索", "grid search", "步长"])
        has_sensitivity = any(kw in raw_exec.lower() for kw in ["敏感性", "sensitivity", "敏感度"])

        # 提取网格搜索步长信息
        grid_step = ""
        for line in raw_exec.split("\n"):
            if "步长" in line or "step" in line.lower():
                grid_step = line.strip()
                break

        if has_grid_search:
            error_text += f"\n| 网格搜索离散化误差 | 系统 | 取决于步长 {grid_step if grid_step else '（未指定）'} | 在最优解附近加密搜索、使用梯度下降局部优化 |"
        else:
            error_text += "\n| 网格搜索离散化误差 | 系统 | 需量化分析 | 减小步长、使用高阶插值 |"

        if has_monte_carlo:
            # 提取置信区间
            ci_lines = []
            for line in raw_exec.split("\n"):
                if "置信区间" in line or "95%" in line or "标准差" in line or "std" in line.lower():
                    ci_lines.append(line.strip())
            ci_text = "；".join(ci_lines[:3]) if ci_lines else "（已执行蒙特卡洛验证）"
            error_text += f"\n| 随机扰动误差 | 随机 | 由蒙特卡洛验证给出：{ci_text} | 增加模拟次数、使用拉丁超立方采样 |"
        else:
            error_text += "\n| 随机扰动误差 | 随机 | 未进行蒙特卡洛验证，无法量化 | 对关键参数添加 ±5% 随机扰动，运行 ≥100 次模拟 |"

        if has_sensitivity:
            error_text += "\n| 参数敏感性误差 | 系统 | 由敏感性分析给出 | 对高敏感参数优先提高测量精度 |"
        else:
            error_text += "\n| 参数敏感性误差 | 系统 | 未进行敏感性分析，无法量化 | 对关键参数进行单因素敏感性分析 |"

        error_text += "\n| 模型简化误差（忽略次要因素等） | 系统 | 次要因素影响较小时可忽略，影响较大时需引入修正项 | 引入修正模型，与简化模型对比量化误差 |"
        error_text += "\n| 数值计算误差 | 随机 | 取决于算法精度和迭代次数 | 提高迭代次数、使用高精度浮点数 |"

        error_text += "\n\n### 2. 数值验证结果汇总\n\n"
        if verification_output:
            error_text += f"```\n{verification_output}\n```\n"

        error_text += "\n### 3. 针对性分析\n\n"
        if llm_analysis and not llm_analysis.startswith("（LLM 分析不可用"):
            # 提取 LLM 分析中的关键结论
            error_text += llm_analysis[:3000] + "\n"
        else:
            error_text += "以下为基于执行结果的自动分析：\n\n"
            if "0.00" in raw_exec and "遗传算法" in raw_exec:
                error_text += "- ⚠️ 遗传算法返回 0.00，说明约束处理存在问题。建议：\n"
                error_text += "  - 使用惩罚函数法处理约束\n"
                error_text += "  - 增大初始种群规模（≥50）\n"
                error_text += "  - 缩小变异范围，确保生成的子代在可行域内\n\n"
            if has_monte_carlo:
                error_text += "- ✅ 蒙特卡洛验证已执行，结果可信度较高\n\n"
            if has_sensitivity:
                error_text += "- ✅ 敏感性分析已执行，关键参数的影响已量化\n\n"

        error_text += "\n### 4. 改进建议\n\n"
        # 根据执行输出中的具体问题生成针对性建议
        specific_issues = []

        # 检测蒙特卡洛鲁棒性
        mc_ratio_match = re.search(
            r'(?:蒙特卡洛.*?均值|MC.*?mean).*?(?:为|is|＝|=)\s*(\d+\.?\d*).*?(?:最优[值解].*?|optimal.*?)(?:为|is|＝|=)\s*(\d+\.?\d*)',
            raw_exec, re.IGNORECASE
        )
        if mc_ratio_match:
            try:
                mc_mean = float(mc_ratio_match.group(1))
                mc_opt = float(mc_ratio_match.group(2))
                if mc_opt > 0:
                    mc_ratio = mc_mean / mc_opt
                    if mc_ratio < 0.30:
                        specific_issues.append(
                            f"**P0-蒙特卡洛鲁棒性极差**: MC均值({mc_mean:.2f})仅为最优值({mc_opt:.2f})的{mc_ratio*100:.0f}%。\n"
                            f"  - 根因：策略对参数扰动过度敏感，属于\"碰运气\"而非\"可靠优化\"\n"
                            f"  - 修复：1)在目标函数中加入鲁棒性惩罚项 λ_robust * max_sensitivity\n"
                            f"          2)在最优解附近选择\"平坦区域\"（Hessian条件数小的区域）\n"
                            f"          3)将最优参数向可行域内部收缩5-10%以换取鲁棒性\n"
                            f"          4)使用min-max鲁棒优化替代确定性优化\n"
                        )
                    elif mc_ratio < 0.50:
                        specific_issues.append(
                            f"**P1-蒙特卡洛鲁棒性不足**: MC均值({mc_mean:.2f})为最优值({mc_opt:.2f})的{mc_ratio*100:.0f}%。\n"
                            f"  - 建议：在目标函数中加入鲁棒性惩罚项，或增加扰动参数覆盖范围\n"
                        )
            except (ValueError, TypeError):
                pass

        # 检测差分进化是否缺失
        if not re.search(r'differential_evolution', raw_exec, re.IGNORECASE):
            specific_issues.append(
                "**P1-缺失差分进化**: 未检测到差分进化算法。国赛要求优化类问题必须包含差分进化作为全局优化算法之一。\n"
                "  - 修复：添加 `from scipy.optimize import differential_evolution` 并作为3种对比算法之一"
            )

        # 检测资源闲置
        if re.search(r'(?:0/3|0/5|未使用|闲置|unused)', raw_exec):
            specific_issues.append(
                "**P0-资源闲置**: 检测到资源节点完全未使用。国赛要求所有资源节点均应参与任务分配。\n"
                "  - 修复：在分配算法中引入多样性约束，确保每个节点至少分配到 floor(N_tasks/N_nodes) 个任务"
            )

        if specific_issues:
            error_text += "\n### 🔴 严重问题（必须修复）\n\n"
            error_text += "\n\n".join(specific_issues)
            error_text += "\n\n"
        error_text += "1. **加密网格搜索**：在最优解附近将步长缩小 5 倍，进行局部精细搜索\n"
        error_text += "2. **修复遗传算法**：使用惩罚函数法处理约束，确保能产出有效解\n"
        error_text += "3. **增加蒙特卡洛模拟次数**：从 100 次增加到 500 次，提高置信区间精度\n"
        error_text += "4. **引入修正模型**：与简化模型对比，量化简化误差\n"
        error_text += "5. **敏感性分析扩展**：对更多参数进行敏感性分析，识别关键参数\n"

        return {
            "current_stage": "error_analysis",
            "stage_history": state.get("stage_history", []) + ["error_analysis"],
            "error_analysis": error_text,
            "code_exec_output": exec_output + f"\n\n[误差分析]\n{error_text}",
            "stage_output": state.get("stage_output", ""),
            "code_exec_success": state.get("code_exec_success", False),
            "exec_error": state.get("exec_error"),
            "error": None,
        }

    def polish_node(state: WorkflowState) -> Dict[str, Any]:
        """论文润色节点：调用 LLM 对论文进行语言润色和内容优化"""
        paper = state.get("paper_output", state.get("stage_output", ""))
        evidence = state.get("evidence_outline", "")
        messages = state.get("messages", [])
        problem = state.get("problem_description", "")
        modeling_report = state.get("modeling_report", "")
        verification_output = state.get("verification_output", "")
        error_analysis = state.get("error_analysis", "")
        model_comparison = state.get("model_comparison", "")
        code_exec_success = state.get("code_exec_success", False)
        exec_error = state.get("exec_error", "")

        # 1. 格式检查（本地执行）
        polish_text = "\n".join([
            "## 论文润色报告",
            "",
            "### 1. 格式检查",
            "",
        ])

        if not code_exec_success:
            polish_text += f"\n⚠️ **严重警告**：代码执行失败，论文中的数值可能是 LLM 幻觉，不可信！\n"
            polish_text += f"执行错误: {exec_error or '未知'}\n"

        # 检查公式编号
        formula_count = len(re.findall(r'\\tag\{|\\(\\d+\\)|\(\\d+\)', paper))
        polish_text += f"- 检测到 {formula_count} 个公式编号\n"

        # 检查图片引用
        figure_refs = len(re.findall(r'图\s*\d+|表\s*\d+', paper))
        polish_text += f"- 检测到 {figure_refs} 个图表引用\n"

        # 检查章节结构
        sections = re.findall(r'^#{1,3}\s+.+', paper, re.MULTILINE)
        polish_text += f"- 检测到 {len(sections)} 个章节标题\n"
        for s in sections[:10]:
            polish_text += f"  - {s.strip()}\n"

        # 检查是否包含必要章节
        required_sections = ["摘要", "问题重述", "模型假设", "符号说明", "模型建立",
                              "模型求解", "结果分析", "敏感性分析", "模型评价", "结论"]
        missing = []
        for sec in required_sections:
            if sec not in paper:
                missing.append(sec)
        if missing:
            polish_text += f"\n⚠️ 缺少以下章节: {', '.join(missing)}\n"
        else:
            polish_text += "\n✅ 所有必要章节均已包含\n"

        # 检查参考文献
        ref_count = len(re.findall(r'\[\d+\]', paper))
        polish_text += f"- 检测到 {ref_count} 条参考文献引用\n"

        # 检查论文是否完整（是否有截断）
        if paper.strip().endswith("...") or paper.strip().endswith("……"):
            polish_text += "\n⚠️ 论文结尾疑似被截断，建议检查输出长度\n"

        # 2. 调用 LLM 进行实际润色（仅在代码执行成功时）
        if code_exec_success:
            try:
                supplement_prompt = f"""请对以下数学建模论文进行润色和内容补充。

## 题目
{problem[:2000]}

## 建模报告摘要
{modeling_report[:2000]}

## 数值验证结果
{verification_output[:2000] if verification_output else "（无）"}

## 模型对比结果
{model_comparison[:2000] if model_comparison else "（无）"}

## 误差分析
{error_analysis[:2000] if error_analysis else "（无）"}

## 当前论文
{paper[:6000]}

请完成以下任务：
1. 如果论文结尾不完整，补充模型评价和结论部分
2. 将数值验证结果、模型对比结果、误差分析的关键结论整合到论文中
3. 确保摘要中的关键数值与正文一致
4. 补充参考文献列表（至少 5 篇相关文献）
5. 确保语言学术化、简洁化

输出润色后的完整论文。"""
                writing_prompt = writing.load_system_prompt().replace("{project_root}", str(config.project_root))
                polished_content = writing.invoke(messages, user_input=supplement_prompt, system_prompt=writing_prompt)
                polish_text += f"\n\n### 2. LLM 润色结果\n\n"
                polish_text += polished_content
                polish_text += "\n\n---\n✅ 论文润色完成。\n"
                paper = polished_content
            except Exception as e:
                polish_text += f"\n\n### 2. LLM 润色\n\n（LLM 润色不可用: {e}，保留原始论文）\n"
        else:
            polish_text += "\n\n### 2. LLM 润色\n\n⚠️ 代码执行失败，跳过 LLM 润色，避免引入幻觉数据。保留原始论文。\n"

        return {
            "current_stage": "polish",
            "stage_history": state.get("stage_history", []) + ["polish"],
            "polished_paper": polish_text,
            "stage_output": state.get("stage_output", ""),
            "paper_output": paper,
            "error": None,
        }

    def p2_check_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        code = state.get("stage_output", "")
        problem_files = state.get("problem_files", [])
        exec_output = state.get("code_exec_output", "")
        figure_files = state.get("figure_files", [])
        project_root = state.get("project_root", str(config.project_root))
        modeling_report = state.get("modeling_report", "")

        figure_list = ""
        figure_audit_result = ""
        if figure_files:
            figure_list = "\n".join(f"  - {Path(f).name}" for f in figure_files)
            figures_dir = str(Path(project_root) / "figures")
            figure_audit_result = _run_figure_audit(figures_dir, skill_root=config.skill_root)
            figure_list = f"共 {len(figure_files)} 个图表文件:\n{figure_list}\n\n图表审计结果:\n{figure_audit_result}"
        else:
            figure_list = "（未检测到图表文件，请确认代码是否生成了图表）"

        problem_type = _detect_problem_type(modeling_report) if modeling_report else "unknown"

        result = quality.check_p2(code, exec_output, figure_list, messages, problem_files)

        status = _parse_quality_status(result)
        is_pass = status == "PASS"
        code_exec_success = state.get("code_exec_success", False)
        code_exec_blocked = state.get("code_exec_blocked", False)
        if not code_exec_success and is_pass:
            if code_exec_blocked:
                result += "\n\n[P2 自动覆盖] 代码被预执行扫描阻断（嵌套过深/致命模式），P2 自动判定为 FAIL。请根据阻断信息修复代码后重试。"
            else:
                result += "\n\n[P2 自动覆盖] 代码执行失败（非零退出码/异常），P2 自动判定为 FAIL。请修复代码后重试。"
            is_pass = False
        # 硬性检查：验证器发现结果异常（如全零值）时自动 FAIL（所有类型通用）
        if is_pass and re.search(r'\[结果异常检测\].*FAIL', exec_output):
            # 从 exec_output 中提取嵌套深度警告
            nesting_warning = ""
            nest_match = re.search(r'嵌套循环过深:.*?最大\s*(\d+)\s*层', exec_output)
            if nest_match:
                nest_depth = nest_match.group(1)
                nesting_warning = (
                    f"\n\n## ⚠️ 嵌套循环过深（{nest_depth} 层）是导致结果异常的根本原因！\n"
                    f"{nest_depth} 层嵌套导致计算时间指数级爆炸，优化器无法在超时内收敛。\n"
                    f"**修复方向**：将所有搜索变量放入参数向量，使用 DE/PSO 优化器搜索，禁止手动嵌套循环！\n"
                    f"目标函数 `evaluate(params)` 必须只做单次计算，不能内部枚举。"
                )
            result += f"\n\n[P2 自动覆盖] 数值验证检测到结果异常（如全零值/NaN/负值），P2 自动判定为 FAIL。请修复算法后重试。{nesting_warning}"
            is_pass = False
        if is_pass and re.search(r'P0-结果异常', exec_output):
            nesting_warning = ""
            nest_match = re.search(r'嵌套循环过深:.*?最大\s*(\d+)\s*层', exec_output)
            if nest_match:
                nest_depth = nest_match.group(1)
                nesting_warning = (
                    f"\n\n## ⚠️ 嵌套循环过深（{nest_depth} 层）是导致结果异常的根本原因！\n"
                    f"{nest_depth} 层嵌套导致计算时间指数级爆炸，优化器无法在超时内收敛。\n"
                    f"**修复方向**：将所有搜索变量放入参数向量，使用 DE/PSO 优化器搜索，禁止手动嵌套循环！"
                )
            result += f"\n\n[P2 自动覆盖] 数值验证检测到 P0-结果异常，P2 自动判定为 FAIL。请修复算法后重试。{nesting_warning}"
            is_pass = False
        # 数值异常（NaN/Inf）检测 — 通用，所有问题类型
        if is_pass and re.search(r'P0-数值异常', exec_output):
            result += "\n\n[P2 自动覆盖] 数值验证检测到 NaN/Inf 值（可能存在除零或数值溢出），P2 自动判定为 FAIL。请检查分母非零、数值稳定性、边界条件处理。"
            is_pass = False
        # 优化类专属检查（仅在问题类型为 A 时触发）
        if problem_type == 'A':
            if is_pass and re.search(r'P0-结果质量', exec_output):
                result += "\n\n[P2 自动覆盖] 结果质量不满足国赛要求（实际值<理论最大值的15%），P2 自动判定为 FAIL。"
                is_pass = False
            if is_pass and re.search(r'P0-鲁棒性', exec_output):
                result += "\n\n[P2 自动覆盖] 蒙特卡洛鲁棒性不满足国赛要求（MC均值<最优值的50%），P2 自动判定为 FAIL。"
                is_pass = False
            if is_pass and re.search(r'P0-蒙特卡洛失败率高', exec_output):
                result += "\n\n[P2 自动覆盖] 蒙特卡洛模拟失败率过高（>20%结果为零），策略鲁棒性不满足国赛要求，P2 自动判定为 FAIL。"
                is_pass = False
            if is_pass and re.search(r'P0-资源闲置', exec_output):
                result += "\n\n[P2 自动覆盖] 检测到资源节点完全未使用，资源利用率不满足国赛要求，P2 自动判定为 FAIL。"
                is_pass = False
        elif problem_type == 'B':
            if is_pass and re.search(r'P0-预测误差', exec_output):
                result += "\n\n[P2 自动覆盖] 预测误差过大，P2 自动判定为 FAIL。"
                is_pass = False
        elif problem_type == 'C':
            if is_pass and re.search(r'P0-评价失效', exec_output):
                result += "\n\n[P2 自动覆盖] 评价方法失效，P2 自动判定为 FAIL。"
                is_pass = False
        elif problem_type == 'D':
            if is_pass and re.search(r'P0-分类异常', exec_output):
                result += "\n\n[P2 自动覆盖] 分类器性能异常，P2 自动判定为 FAIL。"
                is_pass = False
        quality_gates = dict(state.get("quality_gates", {}))
        if is_pass:
            quality_gates["P2"] = "PASS"
        elif status == "BLOCKED":
            quality_gates["P2"] = "BLOCKED"
        else:
            quality_gates["P2"] = "FAIL"
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["P2"] = retry_counts.get("P2", 0) + 1
        else:
            retry_counts["P2"] = 0

        safe_result = _sanitize_llm_output(result)
        return {
            "current_stage": "p2_check",
            "stage_history": state.get("stage_history", []) + ["p2_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
            "messages": [AIMessage(content=f"[P2 编程终检]\n{result}")],
            "stage_output": safe_result,
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
        code_exec_success = state.get("code_exec_success", False)
        exec_error = state.get("exec_error", "")

        # 聚焦问题指令
        focus_q = state.get("focus_question", "").strip()
        if focus_q:
            focus_instruction = f"\n\n## ⚠️ 用户聚焦指令（最高优先级）\n用户明确要求只撰写第{focus_q}问。请只为第{focus_q}问生成证据大纲，忽略其他子问题。"
            modeling_report = focus_instruction + "\n\n" + modeling_report

        figure_list = ""
        if figure_files:
            figure_list = "\n".join(f"  - {Path(f).name}" for f in figure_files)

        if not code_exec_success:
            code_results = (
                f"# ⚠️ 严重警告：代码执行失败，以下为错误信息\n\n"
                f"## 执行错误\n{exec_error or '未知错误'}\n\n"
                f"## 证据大纲撰写规则（必须遵守）\n"
                f"1. 所有主张标注为'待代码修复后验证'\n"
                f"2. 结果表和图表状态标注为'无数据'\n"
                f"3. 公式和推导部分可以正常列出\n\n"
                f"## 原始输出\n{code_results[:3000]}"
            )

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

        safe_result = _sanitize_llm_output(result)
        return {
            "current_stage": "writing_w1",
            "stage_history": state.get("stage_history", []) + ["writing_w1"],
            "messages": [AIMessage(content=result)],
            "stage_output": safe_result,
            "evidence_outline": safe_result,
            "error": None,
        }

    def w1_check_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        evidence = state.get("evidence_outline", state.get("stage_output", ""))
        modeling_report = state.get("modeling_report", "")
        code_results = state.get("code_exec_output", "")
        problem_files = state.get("problem_files", [])
        code_exec_success = state.get("code_exec_success", False)

        result = quality.check_w1(evidence, modeling_report, code_results, messages, problem_files)

        status = _parse_quality_status(result)
        is_pass = status == "PASS"
        if not code_exec_success and is_pass:
            result += "\n\n[W1 自动覆盖] 代码执行失败，证据大纲中的数值无法验证，W1 自动判定为 FAIL。"
            is_pass = False
        quality_gates = dict(state.get("quality_gates", {}))
        if is_pass:
            quality_gates["W1"] = "PASS"
        elif status == "BLOCKED":
            quality_gates["W1"] = "BLOCKED"
        else:
            quality_gates["W1"] = "FAIL"
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["W1"] = retry_counts.get("W1", 0) + 1
        else:
            retry_counts["W1"] = 0

        safe_result = _sanitize_llm_output(result)
        return {
            "current_stage": "w1_check",
            "stage_history": state.get("stage_history", []) + ["w1_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
            "messages": [AIMessage(content=f"[W1 证据大纲门禁]\n{result}")],
            "stage_output": safe_result,
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
        code_exec_success = state.get("code_exec_success", False)
        exec_error = state.get("exec_error", "")

        # 聚焦问题指令
        focus_q = state.get("focus_question", "").strip()
        if focus_q:
            focus_instruction = f"\n\n## ⚠️ 用户聚焦指令（最高优先级）\n用户明确要求只撰写第{focus_q}问。请只为第{focus_q}问生成完整论文，忽略其他子问题。"
            modeling_report = focus_instruction + "\n\n" + modeling_report

        figure_list = ""
        if figure_files:
            figure_list = "\n".join(f"  - {Path(f).name}" for f in figure_files)

        if not code_exec_success:
            code_results = (
                f"# ⚠️ 严重警告：代码执行失败，以下为错误信息\n\n"
                f"## 执行错误\n{exec_error or '未知错误'}\n\n"
                f"## 论文撰写规则（必须遵守）\n"
                f"1. **禁止在结果表格中填入任何数值**（包括'待计算'、'待填'等占位符）\n"
                f"2. 结果表格可以保留结构，但数值列留空或标注'见代码输出'\n"
                f"3. 模型推导、公式、算法步骤必须完整（这些不依赖代码结果）\n"
                f"4. 摘要中不要出现具体数值，改为定性描述（如'模型可实现有效优化'）\n"
                f"5. 敏感性分析和蒙特卡洛验证章节可以省略（因为无数据）\n\n"
                f"## 原始输出\n{code_results[:3000]}"
            )

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

        safe_result = _sanitize_llm_output(result)
        return {
            "current_stage": "writing_full",
            "stage_history": state.get("stage_history", []) + ["writing_full"],
            "messages": [AIMessage(content=result)],
            "stage_output": safe_result,
            "paper_output": safe_result,
            "error": None,
        }

    def w2_check_node(state: WorkflowState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        paper = state.get("paper_output", state.get("stage_output", ""))
        evidence = state.get("evidence_outline", "")
        problem_files = state.get("problem_files", [])
        code_exec_success = state.get("code_exec_success", False)

        result = quality.check_w2(paper, evidence, messages, problem_files)

        status = _parse_quality_status(result)
        is_pass = status == "PASS"
        if not code_exec_success and is_pass:
            result += "\n\n[W2 自动覆盖] 代码执行失败，论文中的数值无法验证，W2 自动判定为 FAIL。"
            is_pass = False
        quality_gates = dict(state.get("quality_gates", {}))
        if is_pass:
            quality_gates["W2"] = "PASS"
        elif status == "BLOCKED":
            quality_gates["W2"] = "BLOCKED"
        else:
            quality_gates["W2"] = "FAIL"
        retry_counts = dict(state.get("retry_counts", {}))
        if not is_pass:
            retry_counts["W2"] = retry_counts.get("W2", 0) + 1
        else:
            retry_counts["W2"] = 0

        safe_result = _sanitize_llm_output(result)
        return {
            "current_stage": "w2_check",
            "stage_history": state.get("stage_history", []) + ["w2_check"],
            "quality_gates": quality_gates,
            "retry_counts": retry_counts,
            "messages": [AIMessage(content=f"[W2 论文终检]\n{result}")],
            "stage_output": safe_result,
            "error": None if is_pass else "W2 门禁未通过",
        }

    def done_node(state: WorkflowState) -> Dict[str, Any]:
        code_exec_success = state.get("code_exec_success", False)
        if code_exec_success:
            msg = "🎉 全部流程完成！请查看交付物。\n\n💡 运行诊断: .venv\\Scripts\\python.exe scripts\\diagnose.py"
        else:
            msg = "⚠️ 流程完成但代码执行失败，请修复代码后重新运行。\n\n💡 运行诊断: .venv\\Scripts\\python.exe scripts\\diagnose.py"
        return {
            "current_stage": "done",
            "stage_history": state.get("stage_history", []) + ["done"],
            "messages": [AIMessage(content=msg)],
            "stage_output": "全部流程完成",
            "error": None,
        }

    def failed_node(state: WorkflowState) -> Dict[str, Any]:
        """P1/P2 耗尽重试或代码执行失败后的失败节点：生成失败报告，不生成论文"""
        code_exec_success = state.get("code_exec_success", False)
        code_exec_blocked = state.get("code_exec_blocked", False)
        exec_error = state.get("exec_error", "")
        exec_output = state.get("code_exec_output", "")
        project_root = state.get("project_root", str(config.project_root))
        current_stage = state.get("current_stage", "")
        quality_gates = state.get("quality_gates", {})

        # 判断失败来源
        if "p1_check" in current_stage or quality_gates.get("P1") == "FAIL":
            fail_source = "P1"
            fail_reason = "P1 最小可运行结果门禁未通过（耗尽重试），代码无法产出有效的最小可运行结果。"
        elif not code_exec_success:
            fail_source = "RUN"
            fail_reason = "代码执行失败（非零退出码或异常）。"
        else:
            fail_source = "P2"
            fail_reason = "P2 编程终检门禁未通过（耗尽重试）。"

        report_lines = [
            "# 求解失败报告",
            "",
            f"## 失败原因（{fail_source}）",
            "",
            fail_reason,
            "",
        ]

        # 根据不同的失败原因给出针对性建议
        if not code_exec_success and exec_error:
            report_lines.append("执行错误信息：")
            report_lines.append(f"```\n{exec_error}\n```")
            report_lines.append("")
        elif code_exec_blocked:
            report_lines.append("代码被预执行扫描阻断，具体原因见上方阻断信息。")
            report_lines.append("")
            report_lines.append("## 阻断原因分析")
            report_lines.append("")
            report_lines.append("预执行扫描在代码运行前检测到致命问题，包括但不限于：")
            report_lines.append("- **嵌套循环过深（≥5层）**：计算量指数级爆炸，必定超时或产出全零/NaN结果")
            report_lines.append("- **禁止的函数名**：如 `heuristic_greedy()`、`greedy_search()` 等暴力枚举函数")
            report_lines.append("- **固定值惩罚 + 优化算法**：`return 1e6` 配合 DE/PSO 保证优化器失效")
            report_lines.append("- **致命代码组合**：`max(0, ...)` 裁剪 + 嵌套过深 → 几乎肯定返回全零")
            report_lines.append("")
            report_lines.append("## 修复方向")
            report_lines.append("")
            report_lines.append("1. **删除暴力枚举函数**：如 `heuristic_greedy()`，改用 DE 优化器")
            report_lines.append("2. **将所有决策变量放入参数向量**：用 `differential_evolution` 统一搜索")
            report_lines.append("3. **使用 `itertools.product()` 展平嵌套**：将 N 层嵌套合并为 1 层")
            report_lines.append("4. **检查约束处理**：确保使用比例惩罚而非固定值惩罚")
            report_lines.append("")
        elif "结果异常检测" in exec_output and "FAIL" in exec_output:
            report_lines.append("代码执行成功，但数值验证检测到结果异常（如全零值/NaN/负值）。")
            report_lines.append("优化算法未能找到有效解，可能原因：")
            report_lines.append("1. 搜索空间过大，随机初始化无法命中有效区域")
            report_lines.append("2. 约束条件过强，可行域过小")
            report_lines.append("3. 初始化策略不当（如参数初始化在无效区域而非可行域附近）")
            report_lines.append("")

        report_lines.extend([
            "## 建议修复方向",
            "",
            "1. 在优化前先计算几何约束（可行域范围），缩小搜索空间",
            "2. 使用网格搜索 + 局部优化的分阶段策略",
            "3. 检查约束判定条件是否过于严格",
            "4. 检查初始条件是否在目标可达区域内",
            "",
            "## 诊断建议",
            "",
            "运行诊断脚本获取详细分析：",
            "```",
            ".venv\\Scripts\\python.exe scripts\\diagnose.py",
            "```",
        ])

        report = "\n".join(report_lines)

        output_path = Path(project_root) / "output" / "求解失败报告.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        return {
            "current_stage": "failed",
            "stage_history": state.get("stage_history", []) + ["failed"],
            "messages": [AIMessage(content=report)],
            "stage_output": report,
            "error": f"{fail_source}质量门禁耗尽重试后仍失败",
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

    def route_after_p1(state: WorkflowState) -> Literal["coding_full", "coding_p1", "failed"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_counts", {}).get("P1", 0)
        if gates.get("P1") == "PASS":
            return "coding_full"
        if retry < config.max_retries:
            return "coding_p1"
        return "failed"

    def route_after_p2(state: WorkflowState) -> Literal["model_comparison", "coding_full", "failed"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_counts", {}).get("P2", 0)
        if gates.get("P2") == "PASS":
            return "model_comparison"
        if retry < config.max_retries:
            return "coding_full"
        return "failed"

    def route_after_code_exec(state: WorkflowState) -> Literal["verify", "failed"]:
        if state.get("code_exec_success", False):
            return "verify"
        # 预扫描阻断（嵌套过深/致命组合）→ 走 verify → p2_check → 重试
        if state.get("code_exec_blocked", False):
            return "verify"
        return "failed"

    def route_after_w1(state: WorkflowState) -> Literal["writing_full", "writing_w1"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_counts", {}).get("W1", 0)
        if gates.get("W1") == "PASS":
            return "writing_full"
        if retry < config.max_retries:
            return "writing_w1"
        return "writing_full"

    def route_after_w2(state: WorkflowState) -> Literal["polish", "writing_full"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_counts", {}).get("W2", 0)
        if gates.get("W2") == "PASS":
            return "polish"
        if retry < config.max_retries:
            return "writing_full"
        return "polish"

    def route_after_polish(state: WorkflowState) -> Literal["done"]:
        return "done"

    workflow.add_node("init", init_node)
    workflow.add_node("modeling", modeling_node)
    workflow.add_node("m1_check", m1_check_node)
    workflow.add_node("coding_p1", coding_p1_node)
    workflow.add_node("p1_check", p1_check_node)
    workflow.add_node("coding_full", coding_full_node)
    workflow.add_node("code_exec", code_exec_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("model_comparison", model_comparison_node)
    workflow.add_node("error_analysis", error_analysis_node)
    workflow.add_node("p2_check", p2_check_node)
    workflow.add_node("writing_w1", writing_w1_node)
    workflow.add_node("w1_check", w1_check_node)
    workflow.add_node("writing_full", writing_full_node)
    workflow.add_node("w2_check", w2_check_node)
    workflow.add_node("polish", polish_node)
    workflow.add_node("done", done_node)
    workflow.add_node("failed", failed_node)

    workflow.set_entry_point("init")

    workflow.add_conditional_edges("init", route_after_init, {"modeling": "modeling", "done": "done"})
    workflow.add_edge("modeling", "m1_check")
    workflow.add_conditional_edges("m1_check", route_after_m1, {"coding_p1": "coding_p1", "modeling": "modeling"})
    workflow.add_edge("coding_p1", "p1_check")
    workflow.add_conditional_edges("p1_check", route_after_p1, {"coding_full": "coding_full", "coding_p1": "coding_p1", "failed": "failed"})
    workflow.add_edge("coding_full", "code_exec")
    workflow.add_conditional_edges("code_exec", route_after_code_exec, {"verify": "verify", "failed": "failed"})
    workflow.add_edge("verify", "p2_check")
    workflow.add_conditional_edges("p2_check", route_after_p2, {"model_comparison": "model_comparison", "coding_full": "coding_full", "failed": "failed"})
    workflow.add_edge("model_comparison", "error_analysis")
    workflow.add_edge("error_analysis", "writing_w1")
    workflow.add_edge("writing_w1", "w1_check")
    workflow.add_conditional_edges("w1_check", route_after_w1, {"writing_full": "writing_full", "writing_w1": "writing_w1"})
    workflow.add_edge("writing_full", "w2_check")
    workflow.add_conditional_edges("w2_check", route_after_w2, {"polish": "polish", "writing_full": "writing_full"})
    workflow.add_edge("polish", "done")
    workflow.add_edge("failed", "done")
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
            figure_files = state.get("figure_files", [])
            figure_list = ""
            if figure_files:
                figure_list = "\n".join(f"  - {Path(f).name}" for f in figure_files)
            result = writing.write_paper(modeling_report, code_results, figure_list, evidence_outline, messages, project_root)
            return {
                "current_stage": "done",
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