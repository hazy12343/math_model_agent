import os
import re
import shutil
import sys
from typing import Literal, Dict, Any, List
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


def _check_result_plausibility(output: str, code: str, problem: str) -> List[str]:
    """检测代码执行结果是否合理，返回 P0/P1 级警告"""
    warnings = []
    output_lower = output.lower()

    # 1. 检测算法是否全部返回 0
    zero_patterns = [
        r'最优[值解].*?[=:]\s*0\.?0?',
        r'[=:]\s*0\.0+',
        r'全部为\s*0',
        r'最优.*?0\.0+',
    ]
    for pat in zero_patterns:
        if re.search(pat, output):
            warnings.append("P0-算法失败: 检测到算法返回 0 值，可能存在约束过强或算法实现错误")
            break

    # 2. 检测迭代类算法是否收敛
    if not re.search(r'已收敛|converged|收敛', output, re.IGNORECASE):
        warnings.append("P1-收敛性: 未检测到收敛性判断，迭代类算法必须输出收敛状态")

    # 3. 检测蒙特卡洛验证
    mc_keywords = ["蒙特卡洛", "monte carlo", "随机模拟", "置信区间", "95%", "扰动"]
    if not any(kw in output_lower for kw in mc_keywords):
        warnings.append("P1-蒙特卡洛: 未检测到蒙特卡洛验证，国赛要求对关键参数进行随机扰动验证")

    # 4. 检测多算法对比
    algo_keywords = ["遗传算法", "粒子群", "模拟退火", "网格搜索", "坐标下降",
                      "genetic", "particle swarm", "simulated annealing", "grid search"]
    if not any(kw in output_lower for kw in algo_keywords):
        warnings.append("P1-多算法: 未检测到多种算法对比，国赛要求至少 2 种算法求解并对比")

    # 5. 检测敏感性分析
    sens_keywords = ["敏感性分析", "sensitivity", "敏感度", "参数扫描"]
    if not any(kw in output_lower for kw in sens_keywords):
        warnings.append("P1-敏感性: 未检测到敏感性分析，国赛要求对关键参数进行敏感性分析")

    # 6. 检测结果中是否有 NaN/Inf（在数值输出上下文中）
    lines = output.split("\n")
    nan_in_output = False
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["nan", "inf", "-inf"]):
            if any(c in line for c in ("=", ":", "结果", "value", "output")):
                nan_in_output = True
                break
    if nan_in_output:
        warnings.append("P0-数值异常: 结果输出中包含 NaN/Inf 值，代码可能存在除零或数值溢出")

    # 7. 检测资源利用率
    # 检查常见资源浪费模式（通用化：检查是否明确说明了资源使用情况）
    if "仅使用" in output or "未使用" in output or "剩余" in output:
        pass  # 已明确说明，不算警告
    else:
        # 通用资源利用率检测：查找 "可用 X，实际 Y" 模式
        resource_patterns = [
            (r'可用\s*(\d+\.?\d*).*?实际\s*(\d+\.?\d*)', "资源利用不足"),
            (r'(\d+\.?\d*)\s*个.*?使用\s*(\d+\.?\d*)\s*个', "资源利用不足"),
        ]
        for pat, desc in resource_patterns:
            match = re.search(pat, output)
            if match:
                try:
                    available = float(match.group(1))
                    used = float(match.group(2))
                    if available > 0 and used < available:
                        warnings.append(f"P1-{desc}: 可用 {available}，实际使用 {used}，资源利用率仅 {used/available*100:.0f}%")
                except (ValueError, ZeroDivisionError):
                    pass

    # 8. 检测搜索精度是否足够（检查搜索结果是否远低于理论最大值）
    # 通用模式：匹配 "理论最大/上限" vs "实际/最优结果" 的数值对比
    theoretical_max_patterns = [
        (r'理论[最大上限].*?(\d+\.?\d*)', r'(?:总|实际|最优).*?(\d+\.?\d*)'),
        (r'最优.*?上限.*?(\d+\.?\d*)', r'实际.*?(\d+\.?\d*)'),
        (r'(?:upper.bound|理论|最大).*?(\d+\.?\d*)', r'(?:result|结果|实际|最优).*?(\d+\.?\d*)'),
    ]
    for theory_pat, actual_pat in theoretical_max_patterns:
        theory_match = re.search(theory_pat, output, re.IGNORECASE)
        actual_match = re.search(actual_pat, output, re.IGNORECASE)
        if theory_match and actual_match:
            theory_max = float(theory_match.group(1))
            actual = float(actual_match.group(1))
            if theory_max > 0 and actual / theory_max < 0.1:
                warnings.append(f"P0-结果质量过低: 实际结果({actual})仅为理论最大值({theory_max})的 {actual/theory_max*100:.1f}%，搜索精度可能严重不足")

    # 9. 检测"未收敛"状态（只有全部或大部分未收敛才警告）
    unconverged_count = len(re.findall(r'未收敛|not converged', output, re.IGNORECASE))
    # 也检测"未找到有效解"和全零结果
    all_zero = bool(re.search(r'全部为\s*0', output))
    no_solution = bool(re.search(r'未找到有效解', output, re.IGNORECASE))
    if all_zero or no_solution:
        warnings.append("P0-算法未收敛: 迭代算法未收敛或未找到有效解，需要调整算法参数或约束处理方式")
    elif unconverged_count >= 3:
        warnings.append(f"P0-算法未收敛: {unconverged_count} 个任务未收敛，可能需要调整算法参数")
    elif unconverged_count > 0:
        warnings.append(f"P1-部分未收敛: {unconverged_count} 个任务未收敛，但多数任务已收敛，整体结果可能仍可用")

    # 10. 检测是否有结果但过于粗糙
    # 如果搜索点数 < 200 且搜索空间 > 1000，给出警告
    search_points_match = re.search(r'(\d+)\s*个点', output)
    if search_points_match:
        points = int(search_points_match.group(1))
        if points < 200:
            warnings.append(f"P1-搜索精度: 仅使用 {points} 个搜索点，建议增加搜索点数以提高精度")

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
            # 格式1: "遗传算法  0.00  1.05" (空格分隔)
            if len(parts) >= 2 and any(kw in stripped for kw in [
                "算法", "遗传", "粒子", "贪心", "网格", "模拟退火", "梯度", "随机", "穷举",
                "genetic", "greedy", "pso", "grid", "模型", "方法", "model", "method"
            ]):
                algo = parts[0]
                vals = parts[1:]
                table_lines.append(f"| {algo} | {' | '.join(vals[:2])} |")
                continue
            # 格式2: "模型A (描述): 0.00 s" (冒号分隔)
            if ":" in stripped or "：" in stripped:
                kv = re.split(r'[：:]', stripped, maxsplit=1)
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

        # 聚焦问题指令
        focus_q = state.get("focus_question", "").strip()
        if focus_q:
            focus_instruction = f"\n\n## ⚠️ 用户聚焦指令（最高优先级）\n用户明确要求只实现第{focus_q}问的代码。请只生成第{focus_q}问的求解代码，忽略其他子问题。"
            modeling_report = focus_instruction + "\n\n" + modeling_report

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
            exec_file = exec_dir / "solution.py"
            exec_file.write_text(code_file, encoding="utf-8")

            # 预执行安全扫描：检测可能超时的代码模式
            pre_scan_warnings = []
            code_lines = code_file.split("\n")
            max_nest = 0
            total_loops = 0
            for line in code_lines:
                stripped = line.strip()
                if stripped.startswith("for ") or stripped.startswith("while "):
                    total_loops += 1
                    indent = len(line) - len(line.lstrip())
                    depth = max(1, indent // 4 + 1)
                    max_nest = max(max_nest, depth)
            de_scan = re.search(r'differential_evolution\(.*?popsize\s*=\s*(\d+).*?maxiter\s*=\s*(\d+)', code_file, re.DOTALL)
            if de_scan:
                de_pop = int(de_scan.group(1))
                de_iter = int(de_scan.group(2))
                if de_pop > 10 or de_iter > 30:
                    pre_scan_warnings.append(f"⚠️ 差分进化参数违规: popsize={de_pop} (要求≤10), maxiter={de_iter} (要求≤30)")
            pso_scan_pop = re.search(r'(?:n_particles|swarm_size)\s*=\s*(\d+)', code_file)
            if pso_scan_pop:
                pso_pop = int(pso_scan_pop.group(1))
                if pso_pop > 20:
                    pre_scan_warnings.append(f"⚠️ PSO 粒子数违规: {pso_pop} (要求≤20)")
            pso_scan_iter = None
            for m in re.finditer(r'for\s+\w+\s+in\s+range\((\d+)\)', code_file):
                n = int(m.group(1))
                if n > 50:
                    pso_scan_iter = n
                    break
            if pso_scan_iter:
                pre_scan_warnings.append(f"⚠️ PSO 迭代次数违规: {pso_scan_iter} (要求≤50)")
            dt_scan = re.search(r'(?:DT|dt|TIME_STEP|time_step)\s*=\s*(\d+\.?\d*)', code_file)
            if dt_scan:
                dt_val = float(dt_scan.group(1))
                if dt_val < 0.2:
                    pre_scan_warnings.append(f"⚠️ 时间步长违规: DT={dt_val} (要求≥0.2)")
            if max_nest > 3:
                pre_scan_warnings.append(f"⚠️ 嵌套循环过深: 最大 {max_nest} 层 (共 {total_loops} 个循环, 要求≤3层)")
            # 检测固定值惩罚（return 1e6 / return 1e9 等）
            fixed_penalty = re.search(r'return\s+(1e\d+|1E\d+|\d{6,})', code_file)
            if fixed_penalty:
                pre_scan_warnings.append(f"⚠️ 固定值惩罚: '{fixed_penalty.group(0)}' — 这会导致优化器失效！请改用比例惩罚（penalty += 1000 * violation）")

            if pre_scan_warnings:
                output = "### ⚠️ 预执行扫描警告（代码可能超时，但将继续执行）\n"
                output += "\n".join(f"  - {w}" for w in pre_scan_warnings)
                output += "\n\n---\n\n"
            else:
                output = ""

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
                pso_iters_match = re.search(r'(?:range\((\d+)\)|n_iterations\s*=\s*(\d+))', code_file)
                # 只捕获大迭代数（>50 的才可能是 PSO 主循环）
                pso_iters = None
                for m in re.finditer(r'for\s+\w+\s+in\s+range\((\d+)\)', code_file):
                    n = int(m.group(1))
                    if n > 50:
                        pso_iters = n
                        break

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
        raw_exec = state.get("raw_exec_output", exec_output)
        code = state.get("stage_output", "")
        project_root = state.get("project_root", str(config.project_root))
        problem = state.get("problem_description", "")

        code_file = _extract_code_from_output(code)
        comparison_text = ""

        # 检测代码中是否包含多算法对比（使用原始执行输出避免被追加内容干扰）
        has_multi_algorithm = False
        algo_keywords = ["遗传算法", "粒子群", "模拟退火", "网格搜索", "坐标下降",
                          "genetic", "particle swarm", "simulated annealing", "grid search",
                          "对比", "比较", "compare", "comparison", "算法对比", "算法比较"]
        for kw in algo_keywords:
            if kw in code_file.lower() or kw in raw_exec.lower():
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
                ("遗传算法", ["遗传算法", "genetic algorithm", "genetic"]),
                ("贪心算法", ["贪心", "greedy"]),
                ("粒子群", ["粒子群", "particle swarm", "pso"]),
                ("模拟退火", ["模拟退火", "simulated annealing", "simulated_annealing"]),
                ("梯度下降", ["梯度下降", "gradient descent", "gradient"]),
                ("坐标下降", ["坐标下降", "coordinate descent", "coordinate"]),
                ("穷举法", ["穷举", "暴力", "brute force", "brute"]),
                ("随机搜索", ["随机搜索", "random search", "random"]),
            ]
            best_values = {}
            for algo_name, algo_keys in algo_patterns:
                for algo_key in algo_keys:
                    patterns = [
                        rf'{re.escape(algo_key)}\s+(\d+\.?\d*)\s+(\d+\.?\d*)',  # 空格对齐格式
                        rf'{re.escape(algo_key)}.*?[：:]\s*(\d+\.?\d*)\s*[秒s]',
                        rf'{re.escape(algo_key)}.*?最优[值解].*?[=:]\s*(\d+\.?\d*)',
                        rf'{re.escape(algo_key)}.*?结果[：:]\s*(\d+\.?\d*)',
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
                comparison_text += "| 最优结果 |" + "".join(f" {v:.4f} s |" for v in best_values.values())
                if len(algo_names) >= 2:
                    vals = list(best_values.values())
                    diff = abs(vals[0] - vals[1])
                    comparison_text += f" {diff:.4f} s |"
                comparison_text += "\n"
                if len(best_values) >= 2:
                    vals = list(best_values.values())
                    diff = abs(vals[0] - vals[1])
                    nonzero_vals = [abs(v) for v in vals if v != 0]
                    if nonzero_vals:
                        if diff < 0.01 * max(nonzero_vals):
                            comparison_text += "| **结论** | **结果一致，解可靠** |\n"
                        else:
                            comparison_text += "| **结论** | **结果差异较大，需进一步分析原因** |\n"
                    else:
                        comparison_text += "| **结论** | **所有算法结果均为0，可能存在算法实现问题** |\n"
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
        raw_exec = state.get("raw_exec_output", exec_output)
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

        # 检测结果是否全为零/NaN（泛化：适用于任何优化问题）
        all_zero = False
        all_nan = False
        non_zero_count = 0
        for line in raw_exec.split("\n"):
            nums = re.findall(r'(?<![a-zA-Z0-9._-])(\d+\.\d+)(?![a-zA-Z0-9._-])', line)
            for n in nums:
                val = float(n)
                if val > 1e-10:
                    non_zero_count += 1
                elif val != 0.0:
                    pass  # negative values are fine
            if re.search(r'\bNaN\b|\bnan\b', line, re.IGNORECASE):
                all_nan = True

        if non_zero_count == 0 and not all_nan:
            all_zero = True

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
                "### 诊断建议",
                "1. 在目标函数中添加调试输出，打印中间计算值",
                "2. 手动构造一个已知可行解，验证目标函数能正确计算",
                "3. 先用网格搜索小范围扫描，确认可行域存在",
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

        error_text += "\n| 模型简化误差（忽略空气阻力等） | 系统 | 低速运动时可忽略，高速运动时需引入阻力修正项 | 引入空气阻力模型，与无阻力模型对比 |"
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
        error_text += "1. **加密网格搜索**：在最优解附近将步长缩小 5 倍，进行局部精细搜索\n"
        error_text += "2. **修复遗传算法**：使用惩罚函数法处理约束，确保能产出有效解\n"
        error_text += "3. **增加蒙特卡洛模拟次数**：从 100 次增加到 500 次，提高置信区间精度\n"
        error_text += "4. **引入空气阻力模型**：与无阻力模型对比，量化简化误差\n"
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
                polish_text += polished_content[:5000] + "\n\n..."
                polish_text += "\n\n---\n✅ 论文润色完成。润色后的论文已更新到 `paper_output` 字段。\n"
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
        code_exec_success = state.get("code_exec_success", False)
        if not code_exec_success and is_pass:
            result += "\n\n[P2 自动覆盖] 代码执行失败（非零退出码/异常），P2 自动判定为 FAIL。请修复代码后重试。"
            is_pass = False
        # 硬性检查：验证器发现结果异常（如全零值）时自动 FAIL
        if is_pass and re.search(r'\[结果异常检测\].*FAIL', exec_output):
            result += "\n\n[P2 自动覆盖] 数值验证检测到结果异常（如全零值/NaN/负值），P2 自动判定为 FAIL。请修复算法后重试。"
            is_pass = False
        if is_pass and re.search(r'P0-结果异常', exec_output):
            result += "\n\n[P2 自动覆盖] 数值验证检测到 P0-结果异常，P2 自动判定为 FAIL。请修复算法后重试。"
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
        """P2 耗尽重试后的失败节点：生成失败报告，不生成论文"""
        code_exec_success = state.get("code_exec_success", False)
        exec_error = state.get("exec_error", "")
        exec_output = state.get("code_exec_output", "")
        project_root = state.get("project_root", str(config.project_root))

        report_lines = [
            "# 求解失败报告",
            "",
            "## 失败原因",
            "",
        ]

        if not code_exec_success:
            report_lines.append("代码执行失败（非零退出码或异常）：")
            report_lines.append(f"```\n{exec_error}\n```")
        elif "结果异常检测" in exec_output and "FAIL" in exec_output:
            report_lines.append("代码执行成功，但数值验证检测到结果异常（如全零值/NaN/负值）。")
            report_lines.append("优化算法未能找到有效解，可能原因：")
            report_lines.append("1. 搜索空间过大，随机初始化无法命中有效区域")
            report_lines.append("2. 约束条件过强，可行域过小")
            report_lines.append("3. 初始化策略不当（如参数初始化在无效区域而非可行域附近）")
        else:
            report_lines.append("代码执行成功，但 P2 质量门禁未通过。")
            report_lines.append("请检查代码逻辑和数值结果。")

        report_lines.extend([
            "",
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
            "error": "P2质量门禁耗尽重试后仍失败",
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

    def route_after_p2(state: WorkflowState) -> Literal["model_comparison", "coding_full", "failed"]:
        gates = state.get("quality_gates", {})
        retry = state.get("retry_counts", {}).get("P2", 0)
        if gates.get("P2") == "PASS":
            return "model_comparison"
        if retry < config.max_retries:
            return "coding_full"
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
    workflow.add_conditional_edges("p1_check", route_after_p1, {"coding_full": "coding_full", "coding_p1": "coding_p1"})
    workflow.add_edge("coding_full", "code_exec")
    workflow.add_edge("code_exec", "verify")
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