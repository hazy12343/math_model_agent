"""
========== 项目产物诊断工具 ==========
用法: python scripts/diagnose.py [projects目录路径]

分析 agent 生成的 projects 产物，输出结构化诊断报告。
用于迭代优化：跑完 agent → 运行本脚本 → 根据报告定位问题 → 修改项目代码。

输出格式: 简洁的 PASS/FAIL/WARN 清单，适合粘贴到对话中。
"""

import sys
import re
from pathlib import Path


def diagnose(project_dir: Path) -> str:
    output_dir = project_dir / "output"
    figures_dir = project_dir / "figures"
    results_dir = project_dir / "results"

    if not output_dir.exists():
        return f"ERROR: 输出目录不存在: {output_dir}"

    lines = []
    lines.append("=" * 60)
    lines.append(f"诊断报告: {project_dir.name}")
    lines.append("=" * 60)

    # ===== 1. 代码执行检查 =====
    lines.append("\n## 1. 代码执行")
    exec_file = output_dir / "代码执行结果.txt"
    code_file = output_dir / "solution_full.py"

    exec_success = True
    exec_error = ""
    if exec_file.exists():
        content = exec_file.read_text(encoding="utf-8", errors="replace")
        has_traceback = "Traceback" in content or "Error" in content
        has_exit_err = "退出码:" in content and "退出码: 0" not in content
        has_timeout = "超时" in content
        has_syntax_err = "语法错误" in content

        if has_traceback or has_exit_err:
            exec_success = False
            # 提取关键错误信息
            for kw in ["ValueError", "TypeError", "NameError", "IndexError",
                        "KeyError", "AttributeError", "ZeroDivisionError", "SyntaxError"]:
                if kw in content:
                    idx = content.index(kw)
                    snippet = content[idx:idx + 200].split("\n")[0]
                    exec_error = snippet
                    break
            if not exec_error:
                exec_error = "代码执行异常（非零退出码）"
            lines.append(f"  FAIL - 代码执行失败")
            lines.append(f"    错误: {exec_error}")
        elif has_timeout:
            exec_success = False
            exec_error = "超时"
            lines.append(f"  FAIL - 代码执行超时")
        elif has_syntax_err:
            exec_success = False
            exec_error = "语法错误"
            lines.append(f"  FAIL - 代码语法错误（修复失败）")
        else:
            lines.append(f"  PASS - 代码执行成功")
    else:
        lines.append(f"  WARN - 未找到代码执行结果文件")
        exec_success = False
        exec_error = "缺少执行结果文件"

    # ===== 2. 图表检查 =====
    lines.append("\n## 2. 图表")
    if figures_dir.exists():
        img_exts = {".png", ".svg", ".jpg", ".jpeg", ".pdf", ".eps"}
        figures = [f for f in figures_dir.iterdir() if f.suffix.lower() in img_exts]
        figure_count = len(figures)
        if figure_count >= 8:
            lines.append(f"  PASS - {figure_count} 张图（达标: ≥8）")
        elif figure_count >= 3:
            lines.append(f"  WARN - {figure_count} 张图（不足，建议 ≥8）")
        else:
            lines.append(f"  FAIL - {figure_count} 张图（严重不足）")
        if figures:
            lines.append(f"    文件: {', '.join(f.name for f in figures[:10])}")
    else:
        lines.append(f"  FAIL - 图表目录不存在，0 张图")

    # ===== 3. 论文检查 =====
    lines.append("\n## 3. 论文")
    paper_file = output_dir / "完整论文.md"
    if paper_file.exists():
        paper = paper_file.read_text(encoding="utf-8", errors="replace")

        # 检查章节完整性
        required = ["摘要", "问题重述", "模型假设", "符号说明", "模型建立",
                     "模型求解", "结果分析", "模型评价", "结论"]
        missing = [s for s in required if s not in paper]
        if missing:
            lines.append(f"  WARN - 缺少章节: {', '.join(missing)}")
        else:
            lines.append(f"  PASS - 全部 {len(required)} 个必要章节齐全")

        # 检查占位符
        placeholders = re.findall(r'XX\.XX|XX\.X|待计算|待填|TODO|TBD|XXX', paper)
        if placeholders:
            lines.append(f"  FAIL - 发现 {len(placeholders)} 个占位符: {placeholders[:5]}...")
        else:
            lines.append(f"  PASS - 无占位符")

        # 检查参考文献
        refs = re.findall(r'\[\d+\]', paper)
        if len(refs) >= 5:
            lines.append(f"  PASS - {len(refs)} 条参考文献引用")
        else:
            lines.append(f"  WARN - 仅 {len(refs)} 条参考文献引用（建议 ≥5）")

        # 检查公式
        formulas = re.findall(r'\$.*?\$|\\begin\{equation\}', paper)
        lines.append(f"  INFO - 约 {len(formulas)} 个公式")

        # 检查图表引用
        fig_refs = re.findall(r'图\s*\d+|表\s*\d+', paper)
        lines.append(f"  INFO - {len(fig_refs)} 个图表引用")

        # 检查代码失败但论文有数值 = 幻觉
        if not exec_success:
            has_numbers = bool(re.findall(r'\d+\.\d+\s*[秒sm]', paper))
            if has_numbers and "代码执行失败" not in paper:
                lines.append(f"  FAIL - 代码执行失败但论文中有具体数值，疑似LLM幻觉！")
    else:
        lines.append(f"  FAIL - 未找到论文文件")

    # ===== 4. 数值验证 =====
    lines.append("\n## 4. 数值验证")
    verify_file = output_dir / "数值验证结果.md"
    if verify_file.exists():
        verify = verify_file.read_text(encoding="utf-8", errors="replace")
        p0_count = verify.count("P0")
        p1_count = verify.count("P1")
        if p0_count > 0:
            lines.append(f"  FAIL - {p0_count} 个 P0 级错误, {p1_count} 个 P1 级警告")
        elif p1_count > 0:
            lines.append(f"  WARN - {p1_count} 个 P1 级警告")
        else:
            lines.append(f"  PASS - 无严重问题")
    else:
        lines.append(f"  INFO - 未找到数值验证文件")

    # ===== 5. 结果文件 =====
    lines.append("\n## 5. 结果文件")
    if results_dir.exists():
        result_files = list(results_dir.iterdir())
        csv_files = [f for f in result_files if f.suffix.lower() == ".csv"]
        if csv_files:
            lines.append(f"  PASS - {len(csv_files)} 个 CSV 文件: {', '.join(f.name for f in csv_files)}")
        else:
            lines.append(f"  WARN - 无 CSV 结果文件（缺少敏感性分析等数据）")
        other_files = [f for f in result_files if f.suffix.lower() != ".csv"]
        if other_files:
            lines.append(f"  INFO - 其他文件: {', '.join(f.name for f in other_files)}")
    else:
        lines.append(f"  WARN - 结果目录不存在")

    # ===== 6. 代码质量 =====
    lines.append("\n## 6. 代码质量")
    if code_file.exists():
        code = code_file.read_text(encoding="utf-8", errors="replace")
        code_lines_count = len(code.split("\n"))
        lines.append(f"  INFO - {code_lines_count} 行")

        # 检查是否有防御性编程措施
        has_try = "try:" in code
        has_assert = "assert " in code
        has_outer = "np.outer" in code
        has_nan_check = "nan" in code.lower() or "isnan" in code.lower()
        if has_try and has_assert:
            lines.append(f"  PASS - 有 try/except + assert 防御措施")
        elif has_try:
            lines.append(f"  WARN - 有 try/except，缺少 assert 形状检查")
        else:
            lines.append(f"  WARN - 缺少 try/except 异常处理")

        if has_outer:
            lines.append(f"  PASS - 使用了 np.outer（广播安全）")
        elif "np.array" in code and "*" in code:
            lines.append(f"  WARN - 使用了 NumPy 数组乘法，需检查广播安全性")

        # 检查多算法
        algo_count = 0
        for kw in ["网格搜索", "遗传算法", "粒子群", "模拟退火", "梯度下降",
                     "differential_evolution", "genetic", "pso", "particle_swarm",
                     "simulated_annealing", "ant_colony", "aco", "tabu_search",
                     "差分进化", "蚁群", "禁忌搜索", "爬山", "hill_climbing",
                     "双层优化", "double_layer", "two_stage"]:
            if kw in code:
                algo_count += 1
        if algo_count >= 2:
            lines.append(f"  PASS - {algo_count} 种算法对比")
        elif algo_count == 1:
            lines.append(f"  WARN - 仅 1 种算法（国赛要求 ≥2）")
        else:
            lines.append(f"  FAIL - 未检测到标准算法名称")

    else:
        lines.append(f"  FAIL - 未找到代码文件")

    # ===== 7. 综合评分 =====
    lines.append("\n" + "=" * 60)
    lines.append("## 综合评分")

    score = 0
    max_score = 0
    checks = []

    if exec_success:
        score += 30; max_score += 30
        checks.append("代码执行: +30")
    else:
        max_score += 30
        checks.append("代码执行: +0/30 (FAIL)")

    if figures_dir.exists():
        fig_count = len([f for f in figures_dir.iterdir() if f.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg", ".pdf", ".eps"}])
        if fig_count >= 8:
            score += 20; max_score += 20
            checks.append(f"图表: +20/20 ({fig_count}张)")
        elif fig_count >= 3:
            score += 10; max_score += 20
            checks.append(f"图表: +10/20 ({fig_count}张)")
        else:
            max_score += 20
            checks.append(f"图表: +0/20 ({fig_count}张)")
    else:
        max_score += 20
        checks.append("图表: +0/20 (无)")

    if paper_file.exists():
        paper = paper_file.read_text(encoding="utf-8", errors="replace")
        missing = [s for s in required if s not in paper]
        placeholders = re.findall(r'XX\.XX|XX\.X|待计算|待填|TODO|TBD', paper)

        if not missing:
            score += 15; max_score += 15
            checks.append("论文章节: +15/15")
        else:
            score += 5; max_score += 15
            checks.append(f"论文章节: +5/15 (缺{len(missing)}个)")

        if not placeholders:
            score += 15; max_score += 15
            checks.append("无占位符: +15/15")
        else:
            max_score += 15
            checks.append(f"无占位符: +0/15 ({len(placeholders)}个)")

        if not exec_success and not any(kw in paper for kw in ["代码执行失败", "严重警告"]):
            score -= 10
            checks.append("幻觉检测: -10 (代码失败但论文有数值)")
    else:
        max_score += 30
        checks.append("论文: +0/30 (无)")

    if code_file.exists():
        if algo_count >= 2:
            score += 20; max_score += 20
            checks.append(f"多算法: +20/20 ({algo_count}种)")
        elif algo_count == 1:
            score += 10; max_score += 20
            checks.append(f"多算法: +10/20 (1种)")
        else:
            max_score += 20
            checks.append("多算法: +0/20 (无)")
    else:
        max_score += 20
        checks.append("多算法: +0/20 (无代码)")

    for c in checks:
        lines.append(f"  {c}")

    pct = score / max_score * 100 if max_score > 0 else 0
    lines.append(f"\n  总分: {score}/{max_score} ({pct:.0f}%)")

    if pct >= 80:
        lines.append("  等级: 国赛省一 ~ 国奖水平")
    elif pct >= 60:
        lines.append("  等级: 省二 ~ 省一水平")
    elif pct >= 40:
        lines.append("  等级: 省三水平")
    else:
        lines.append("  等级: 成功参赛奖 / 需大幅改进")

    # ===== 8. 改进建议 =====
    lines.append("\n## 8. 改进建议")
    suggestions = []

    if not exec_success:
        suggestions.append(f"P0: 修复代码执行错误 → {exec_error}")

    if figures_dir.exists():
        fig_count = len([f for f in figures_dir.iterdir() if f.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg", ".pdf", ".eps"}])
        if fig_count < 8:
            suggestions.append(f"P1: 增加图表数量（当前 {fig_count}，目标 ≥8）")

    if paper_file.exists():
        if placeholders:
            suggestions.append(f"P0: 移除论文中的占位符（{len(placeholders)} 个）")
        if missing:
            suggestions.append(f"P1: 补充论文缺失章节: {', '.join(missing)}")

    if code_file.exists():
        if algo_count < 2:
            suggestions.append("P1: 实现至少 2 种求解算法对比")
        if not has_try:
            suggestions.append("P2: 添加 try/except 异常处理提高代码鲁棒性")

    if not suggestions:
        suggestions.append("无关键改进建议，产物质量良好。")

    for s in suggestions:
        lines.append(f"  - {s}")

    lines.append("\n" + "=" * 60)
    lines.append("诊断完成。将以上报告粘贴到对话中，即可继续优化项目。")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        project_dir = Path(sys.argv[1])
    else:
        project_dir = Path(__file__).parent.parent / "projects"

    if not project_dir.exists():
        print(f"ERROR: 目录不存在: {project_dir}")
        print(f"用法: python scripts/diagnose.py [projects目录路径]")
        sys.exit(1)

    report = diagnose(project_dir)
    print(report)

    # 同时保存报告到 projects 目录
    report_path = project_dir / "诊断报告.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()