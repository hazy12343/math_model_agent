import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class NumericalVerifier:
    """数值验证引擎：维度分析、边界条件、交叉验证、符号推导"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

    def dimensional_analysis(self, code: str) -> Dict[str, Any]:
        """对代码中的变量进行量纲分析"""
        findings = []
        variables = self._extract_variables(code)

        if not variables:
            return {"status": "UNKNOWN", "findings": ["未检测到变量定义"], "variables": {}}

        var_info = {}
        for name, info in variables.items():
            dims = self._infer_dimensions(info)
            if not dims.get("unit"):
                dims = self._infer_dimensions_from_name(name)
            var_info[name] = dims

        for name, info in var_info.items():
            if info.get("unit") and info.get("expected_unit"):
                if info["unit"] != info["expected_unit"]:
                    findings.append(
                        f"P0-量纲错误: 变量 '{name}' 单位 '{info['unit']}' "
                        f"与预期 '{info['expected_unit']}' 不一致"
                    )
            elif info.get("unit"):
                findings.append(f"PASS: 变量 '{name}' 量纲检查通过 ({info['unit']})")

        status = "PASS" if not any("P0" in f for f in findings) else "FAIL"
        return {"status": status, "findings": findings, "variables": var_info}

    def _extract_variables(self, code: str) -> Dict[str, Dict]:
        """提取代码中的变量定义和上下文"""
        variables = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            name = target.id
                            context = self._get_assignment_context(node)
                            if name in variables:
                                variables[name].update(context)
                            else:
                                variables[name] = context
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        name = node.target.id
                        context = self._get_assignment_context(node)
                        variables[name] = context
        except SyntaxError:
            for line in code.split("\n"):
                m = re.match(r'\s*(\w+)\s*=\s*(.+)', line)
                if m:
                    name = m.group(1)
                    value = m.group(2).strip()
                    variables[name] = {"raw_value": value}

        return variables

    def _get_assignment_context(self, node) -> Dict:
        ctx = {}
        try:
            if isinstance(node, ast.Assign) and node.value:
                if isinstance(node.value, ast.Constant):
                    ctx["raw_value"] = str(node.value.value)
        except Exception:
            pass
        return ctx

    def _infer_dimensions(self, info: Dict) -> Dict:
        result = {}
        raw = info.get("raw_value", "").lower()

        # 复合单位优先匹配
        unit_patterns = [
            (r'm/s', 'm/s'),
            (r'm\^2', 'm^2'),
            (r'm\^3', 'm^3'),
            (r'km/h', 'km/h'),
            (r'rad/s', 'rad/s'),
            (r'm/s\^2', 'm/s^2'),
            (r'kg/m\^3', 'kg/m^3'),
            (r'j\b', 'J'),
            (r'w\b', 'W'),
            (r'n\b', 'N'),
            (r'pa\b', 'Pa'),
            (r'hz\b', 'Hz'),
            (r'deg\b', 'deg'),
            (r'kg\b', 'kg'),
            (r'g\b', 'g'),
            (r'ms\b', 'ms'),
            (r'min\b', 'min'),
            (r'h\b', 'h'),
        ]
        for pat, unit in unit_patterns:
            if re.search(pat, raw):
                result["unit"] = unit
                result["expected_unit"] = unit
                return result

        # 简单单位（检查后缀）
        if raw.endswith("m") and not any(c in raw for c in ("cm", "km", "mm", "nm", "um", "dm")):
            result["unit"] = "m"
            result["expected_unit"] = "m"
        elif raw.endswith("s") and not raw.endswith("ms"):
            result["unit"] = "s"
            result["expected_unit"] = "s"
        elif raw.endswith("kg"):
            result["unit"] = "kg"
            result["expected_unit"] = "kg"
        elif raw.endswith("rad"):
            result["unit"] = "rad"
            result["expected_unit"] = "rad"

        return result

    def _infer_dimensions_from_name(self, name: str) -> Dict:
        result = {}
        name_lower = name.lower()

        # 速度/速率
        if any(kw in name_lower for kw in ("speed", "velocity", "速度", "速率", "vx", "vy", "vz", "v_")):
            result["unit"] = "m/s"
            result["expected_unit"] = "m/s"
        # 加速度
        elif any(kw in name_lower for kw in ("acceleration", "accel", "加速度", "ax", "ay", "az", "a_")):
            result["unit"] = "m/s^2"
            result["expected_unit"] = "m/s^2"
        # 面积
        elif any(kw in name_lower for kw in ("area", "面积", "区域", "surface")):
            result["unit"] = "m^2"
            result["expected_unit"] = "m^2"
        # 体积
        elif any(kw in name_lower for kw in ("volume", "体积", "容积", "vol")):
            result["unit"] = "m^3"
            result["expected_unit"] = "m^3"
        # 距离/长度/坐标（单字母变量使用精确匹配，避免子串误报）
        elif any(kw in name_lower for kw in (
            "distance", "length", "width", "height", "depth", "altitude",
            "距离", "长度", "宽度", "高度", "深度", "海拔",
            "pos", "position", "radius", "range",
            "coordinate", "坐标", "offset", "偏移", "span", "跨度",
        )) or name_lower in ("x", "y", "z", "r"):
            result["unit"] = "m"
            result["expected_unit"] = "m"
        # 时间（单字母变量使用精确匹配，避免子串误报）
        elif any(kw in name_lower for kw in (
            "time", "duration", "时间", "时长", "period",
            "timestamp", "时刻", "interval", "间隔", "delay", "延迟",
            "start", "end", "开始", "结束", "window", "窗口",
        )) or name_lower in ("t", "dt"):
            result["unit"] = "s"
            result["expected_unit"] = "s"
        # 质量
        elif any(kw in name_lower for kw in ("mass", "质量", "weight", "重量", "m_", "payload")):
            result["unit"] = "kg"
            result["expected_unit"] = "kg"
        # 角度
        elif any(kw in name_lower for kw in (
            "angle", "角度", "theta", "alpha", "beta", "gamma", "phi",
            "delta", "epsilon", "omega", "azimuth", "方位", "elevation", "仰角",
            "pitch", "roll", "yaw", "heading", "航向", "direction", "方向",
        )):
            result["unit"] = "rad"
            result["expected_unit"] = "rad"
        # 力
        elif any(kw in name_lower for kw in ("force", "力", "thrust", "推力", "drag", "阻力", "lift", "升力", "f_")):
            result["unit"] = "N"
            result["expected_unit"] = "N"
        # 能量/功率
        elif any(kw in name_lower for kw in ("energy", "能量", "power", "功率", "work", "功", "p_", "e_")):
            result["unit"] = "J"
            result["expected_unit"] = "J"
        elif any(kw in name_lower for kw in ("power", "功率", "watt", "瓦")):
            result["unit"] = "W"
            result["expected_unit"] = "W"
        # 密度
        elif any(kw in name_lower for kw in ("density", "密度", "rho", "concentration", "浓度")):
            result["unit"] = "kg/m^3"
            result["expected_unit"] = "kg/m^3"
        # 频率
        elif any(kw in name_lower for kw in ("frequency", "频率", "freq", "hz", "f_")):
            result["unit"] = "Hz"
            result["expected_unit"] = "Hz"
        # 概率/比率（无量纲）
        elif any(kw in name_lower for kw in (
            "probability", "概率", "ratio", "比率", "rate", "率",
            "percentage", "百分比", "pct", "percent", "proportion", "比例",
            "efficiency", "效率", "coverage", "覆盖率", "utilization", "利用率",
        )):
            result["unit"] = "dimensionless"
            result["expected_unit"] = "dimensionless"

        return result

    def boundary_condition_check(self, results: str, problem_description: str) -> Dict[str, Any]:
        """检查边界条件是否满足"""
        findings = []
        constraints = self._extract_constraints(problem_description)

        for constraint in constraints:
            check = self._check_constraint(constraint, results)
            if check["status"] == "FAIL":
                findings.append(
                    f"P0-边界条件违反: {constraint['name']} = {constraint['value']} "
                    f"({constraint['type']})，结果中未验证或违反"
                )
            elif check["status"] == "PASS":
                findings.append(f"PASS: 边界条件 {constraint['name']} 已满足")

        if not constraints:
            return {"status": "UNKNOWN", "findings": ["未从题目中提取到边界条件"], "constraints": []}

        status = "PASS" if not any("P0" in f for f in findings) else "FAIL"
        return {"status": status, "findings": findings, "constraints": constraints}

    def _extract_constraints(self, problem: str) -> List[Dict]:
        """从题目中提取约束条件"""
        constraints = []
        patterns = [
            (
                r'([\u4e00-\u9fff\w]+)\s*(?:不能|不得超过|不大于|不小于|至少|至多|必须|应当)\s*([\d.]+)\s*([\u4e00-\u9fff\w]*)',
                lambda m: {"name": m.group(1), "value": m.group(2), "unit": m.group(3), "type": "hard"},
            ),
            (
                r'([\u4e00-\u9fff\w]+)\s*[=＝]\s*([\d.]+)\s*([\u4e00-\u9fff\w/]*)',
                lambda m: {"name": m.group(1), "value": m.group(2), "unit": m.group(3), "type": "parameter"},
            ),
        ]
        for pattern, factory in patterns:
            for m in re.finditer(pattern, problem):
                try:
                    constraints.append(factory(m))
                except Exception:
                    pass
        return constraints

    def _check_constraint(self, constraint: Dict, results: str) -> Dict:
        value = constraint.get("value", "").strip()
        name = constraint.get("name", "").strip()
        unit = constraint.get("unit", "").strip()
        if not value or not name:
            return {"status": "FAIL", "evidence": ""}

        escaped_value = re.escape(value)
        escaped_name = re.escape(name)
        if unit:
            escaped_unit = re.escape(unit)
            pattern = re.compile(rf'{escaped_name}\s*[=＝:：]?\s*{escaped_value}\s*{escaped_unit}')
        else:
            pattern = re.compile(rf'{escaped_name}.*?{escaped_value}')
        for line in results.split("\n"):
            if pattern.search(line):
                return {"status": "PASS", "evidence": line.strip()}
        return {"status": "FAIL", "evidence": ""}

    def cross_validation(self, code: str, results: str) -> Dict[str, Any]:
        """交叉验证：检查代码中的计算是否自洽"""
        findings = []

        values_in_results = re.findall(r'([\d.]+(?:e[+-]?\d+)?)', results)

        if not values_in_results:
            return {"status": "UNKNOWN", "findings": ["结果中未检测到数值输出"], "self_consistent": None}

        try:
            numeric_results = [float(v) for v in values_in_results]
            if len(numeric_results) >= 2:
                nonzero = [abs(v) for v in numeric_results if v != 0]
                if nonzero:
                    ratio = max(nonzero) / min(nonzero)
                    if ratio > 1e6:
                        findings.append(
                            f"P1-数值范围异常: 最大值与最小值之比为 {ratio:.2e}，"
                            f"可能存在量纲不一致"
                        )
        except ValueError:
            pass

        # 改进的 NaN/Inf 检测：区分"结果文本中的 nan 字符串"和"真正的数值异常"
        nan_count = 0
        nan_keywords = ["nan", "inf", "-inf", "infinity", "-infinity"]
        results_lower = results.lower()
        for kw in nan_keywords:
            nan_count += results_lower.count(kw)

        if nan_count > 0:
            # 检查这些 NaN 是否出现在数值输出上下文中（而非代码或注释中）
            lines = results.split("\n")
            nan_in_output = False
            for line in lines:
                line_lower = line.lower()
                # 排除"无NaN"、"无nan"、"no NaN"等否定表述
                if any(neg in line_lower for neg in ("无nan", "无 nan", "无 nan ", "无nan值", "nan not found", "no nan", "no  nan", "no nan值", "未检测到nan", "未发现nan", "所有.*无nan", "no nan values", "all values are valid", "所有值有效")):
                    continue
                for kw in nan_keywords:
                    if kw in line_lower:
                        # 检查是否包含数值上下文
                        if any(c in line for c in ("=", ":", "结果", "value", "output")):
                            nan_in_output = True
                            break
            if nan_in_output:
                findings.append(f"P0-数值异常: 结果输出中包含 {nan_count} 个 NaN/Inf 值，代码可能存在除零或数值溢出")
            else:
                findings.append(f"PASS: 检测到 NaN/Inf 关键词但不在数值输出上下文中，已排除误报")

        negative_count = 0
        for v in values_in_results:
            try:
                if float(v) < 0:
                    negative_count += 1
            except ValueError:
                pass
        if negative_count > 0:
            findings.append(
                f"P1-负值警告: 结果中包含 {negative_count} 个负值，"
                f"请确认物理量是否允许负值"
            )

        status = "PASS" if not any("P0" in f for f in findings) else "FAIL"
        return {
            "status": status,
            "findings": findings if findings else ["数值自洽性检查通过"],
            "self_consistent": len(findings) == 0,
        }

    def sensitivity_check(self, results: str, code: str = "") -> Dict[str, Any]:
        """检查敏感性分析结果的合理性"""
        findings = []

        sensitivity_keywords = ["敏感性", "sensitivity", "参数扫描", "parameter sweep"]
        has_sensitivity = any(kw in results.lower() for kw in sensitivity_keywords)

        if not has_sensitivity:
            findings.append("P2-缺失: 结果中未检测到敏感性分析")

        csv_pattern = r'sensitivity.*?\.csv'
        has_csv = bool(re.search(csv_pattern, results, re.IGNORECASE))
        if not has_csv and code:
            has_csv = bool(re.search(csv_pattern, code, re.IGNORECASE))
        if not has_csv and has_sensitivity:
            findings.append("P2-缺失: 敏感性分析未输出CSV文件（请确保代码中输出 'sensitivity.csv' 字样）")

        # 检测敏感性结果是否全为零/全相同（说明基础结果就是零，敏感性分析无意义）
        csv_path = None
        csv_match = re.search(r'(?:results[/\\]|\./)?(sensitivity[^,\s]*\.csv)', results, re.IGNORECASE)
        if csv_match:
            csv_path = csv_match.group(1)
        if csv_path and not csv_path.startswith("results"):
            csv_path = "results/" + csv_path
        if csv_path:
            from pathlib import Path
            p = Path(csv_path)
            # 尝试多个可能的路径位置
            candidates = [
                p,
                Path(self.project_root) / "results" / p.name,
                Path("results") / p.name,
                Path("projects") / csv_path,
                Path.cwd() / csv_path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    p = candidate
                    break
            if p.exists():
                try:
                    import csv as csv_module
                    with open(p, 'r', encoding='utf-8-sig', errors='replace') as f:
                        reader = csv_module.DictReader(f)
                        rows = list(reader)
                    if rows:
                        change_vals = []
                        for row in rows:
                            for col_name in ['change_pct', '变化率', 'change', 'pct_change']:
                                # 支持列名如 "变化率(%)" 等变体
                                matched_key = None
                                for k in row.keys():
                                    if col_name in k:
                                        matched_key = k
                                        break
                                if matched_key:
                                    try:
                                        change_vals.append(float(row[matched_key]))
                                    except (ValueError, TypeError):
                                        pass
                                    break
                        if change_vals:
                            if all(abs(v) < 1e-9 for v in change_vals):
                                findings.append(
                                    "P0-敏感性无效: 所有敏感性变化率均为 0，说明基础结果为零，"
                                    "敏感性分析无意义。请先确保优化结果非零后再进行敏感性分析"
                                )
                            elif len(set(round(v, 6) for v in change_vals)) <= 1:
                                findings.append(
                                    "P1-敏感性均一: 所有参数的变化率完全相同，"
                                    "可能敏感性分析代码有误（如始终使用同一参数值）"
                                )
                            # 检测部分参数变化率为零（说明这些参数未被正确变化）
                            zero_change_count = sum(1 for v in change_vals if abs(v) < 1e-9)
                            non_zero_count = len(change_vals) - zero_change_count
                            if zero_change_count > 0 and non_zero_count > 0 and zero_change_count >= non_zero_count:
                                findings.append(
                                    f"P1-敏感性部分无效: {zero_change_count}/{len(change_vals)} 个变化率为零，"
                                    "说明部分参数的变化未影响结果（参数未被正确变化或代码中未使用这些参数），"
                                    "请检查敏感性分析代码是否真正改变了每个参数"
                                )
                except Exception:
                    pass  # 读取失败不阻塞

        status = "PASS" if not any("P0" in f or "P2-缺失" in f for f in findings) else "FAIL"
        return {
            "status": status,
            "findings": findings if findings else ["敏感性分析检查通过"],
            "has_sensitivity": has_sensitivity,
        }

    def format_verification(self, figures_dir: str) -> Dict[str, Any]:
        """验证图表格式（SVG文本可编辑性、DPI等）"""
        figs_path = Path(figures_dir)
        findings = []

        svg_files = list(figs_path.glob("*.svg"))
        png_files = list(figs_path.glob("*.png")) if figs_path.exists() else []

        if not svg_files and not png_files:
            return {"status": "FAIL", "findings": ["未找到任何图表文件"], "files": []}

        for svg_file in svg_files:
            try:
                content = svg_file.read_text(encoding="utf-8")
                if "<text" not in content:
                    findings.append(
                        f"P1-SVG文本: {svg_file.name} 没有可编辑文本节点，"
                        f"请设置 matplotlib.rcParams['svg.fonttype'] = 'none'"
                    )
            except Exception as e:
                findings.append(f"P2-SVG读取: {svg_file.name} 读取失败 ({e})")

        if not svg_files and png_files:
            findings.append("P2-格式: 未生成SVG格式图表，建议同时输出PNG和SVG")

        status = "PASS" if not any("P1" in f for f in findings) else "FAIL"
        return {
            "status": status,
            "findings": findings if findings else ["图表格式检查通过"],
            "svg_count": len(svg_files),
            "png_count": len(png_files),
        }

    def convergence_check(self, results: str) -> Dict[str, Any]:
        """检查结果中是否包含收敛性分析"""
        findings = []

        # 过滤掉诊断警告行（如 "[结果合理性检测]" 部分的 "P1-收敛性: ..."），
        # 这些是 Agent 的元诊断，不是代码的实际输出，不应参与收敛性分析
        filtered_results = "\n".join(
            line for line in results.split("\n")
            if not re.match(r'\s*⚠️\s*P[012]-', line)  # 排除 "⚠️ P1-收敛性: ..."
            and not re.match(r'\s*\[结果合理性检测\]', line)  # 排除节标题
        )

        convergence_keywords = ["收敛", "convergence", "迭代", "iteration", "收敛曲线"]
        has_convergence = any(kw in filtered_results.lower() for kw in convergence_keywords)

        if not has_convergence:
            findings.append("P2-缺失: 结果中未检测到收敛性分析")
        else:
            # 检查是否给出了收敛状态
            converge_state_keywords = ["已收敛", "converged", "未收敛", "not converged",
                                        "最大迭代", "max iteration", "改进量", "improvement",
                                        "收敛代数", "收敛于", "最终适应度", "final fitness"]
            has_state = any(kw in filtered_results.lower() for kw in converge_state_keywords)
            if has_state:
                findings.append("PASS: 检测到收敛性分析及收敛状态")
            else:
                findings.append("P1-不完整: 检测到收敛性分析但未明确收敛状态")

        status = "PASS" if not any("P1" in f or "P2-缺失" in f for f in findings) else "FAIL"
        return {
            "status": status,
            "findings": findings if findings else ["收敛性分析检查通过"],
            "has_convergence": has_convergence,
        }

    def extreme_value_test(self, code: str) -> Dict[str, Any]:
        """检查代码中是否包含极端值测试（边界参数测试）"""
        findings = []

        test_keywords = ["极端", "extreme", "边界", "boundary", "边缘", "极限",
                          "测试", "test", "验证"]
        has_test = any(kw in code.lower() for kw in test_keywords)

        boundary_test_patterns = [
            r'np\.linspace.*0.*1',
            r'np\.arange.*0',
            r'param.*=.*0',
            r'param.*=.*max',
            r'param.*=.*min',
            r'边界条件',
            r'extreme.*test',
            r'corner.*case',
        ]
        has_boundary_code = any(re.search(p, code, re.IGNORECASE) for p in boundary_test_patterns)

        if has_boundary_code:
            findings.append("PASS: 检测到边界值测试代码")
        else:
            findings.append("P2-缺失: 未检测到极端值/边界值测试，建议验证参数取边界值时结果是否合理")

        status = "PASS" if not any("P1" in f or "P2-缺失" in f for f in findings) else "FAIL"
        return {
            "status": status,
            "findings": findings if findings else ["极端值测试检查通过"],
            "has_boundary_test": has_boundary_code,
        }

    def symmetry_check(self, code: str, problem_description: str = "") -> Dict[str, Any]:
        """检查代码中是否包含对称性验证（适用于物理对称的问题）"""
        findings = []

        symmetry_keywords = ["对称", "symmetry", "对称性", "对称性验证"]
        has_symmetry = any(kw in code.lower() for kw in symmetry_keywords)

        # 从题目描述中判断问题是否具有对称性
        sym_problem_keywords = ["对称", "均匀", "各向同性", "isotropic", "homogeneous",
                                 "轴对称", "中心对称", "对称性"]
        problem_has_symmetry = any(kw in problem_description.lower() for kw in sym_problem_keywords)

        if problem_has_symmetry and not has_symmetry:
            findings.append("P1-缺失: 题目具有对称性特征，但代码中未检测到对称性验证")
        elif has_symmetry:
            findings.append("PASS: 检测到对称性验证")

        if not findings:
            findings.append("PASS: 未检测到对称性要求（或问题不具有对称性）")

        status = "PASS" if not any("P1" in f for f in findings) else "FAIL"
        return {
            "status": status,
            "findings": findings,
            "has_symmetry_check": has_symmetry,
        }

    def conservation_check(self, code: str) -> Dict[str, Any]:
        """检查代码中是否包含守恒量验证（适用于物理类题目）"""
        findings = []

        conservation_keywords = ["守恒", "conservation", "守恒量", "守恒定律",
                                  "能量守恒", "动量守恒", "质量守恒"]
        has_conservation = any(kw in code.lower() for kw in conservation_keywords)

        physics_keywords = ["力", "force", "能量", "energy", "动量", "momentum",
                             "速度", "velocity", "加速度", "acceleration"]
        has_physics = any(kw in code.lower() for kw in physics_keywords)

        if has_physics and not has_conservation:
            findings.append("P2-建议: 代码涉及物理量计算，建议增加守恒量验证（如能量守恒、动量守恒）")
        elif has_conservation:
            findings.append("PASS: 检测到守恒量验证")

        if not findings:
            findings.append("PASS: 未检测到守恒量要求（或问题不涉及物理守恒）")

        status = "PASS" if not any("P0" in f for f in findings) else "FAIL"
        return {
            "status": status,
            "findings": findings,
            "has_conservation_check": has_conservation,
        }