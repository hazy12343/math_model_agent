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
                elif isinstance(node.value, (ast.Num,)):
                    ctx["raw_value"] = str(node.value.n)
        except Exception:
            pass
        return ctx

    def _infer_dimensions(self, info: Dict) -> Dict:
        result = {}
        raw = info.get("raw_value", "")

        if "m/s" in raw.lower():
            result["unit"] = "m/s"
            result["expected_unit"] = "m/s"
        elif "m^2" in raw.lower():
            result["unit"] = "m^2"
            result["expected_unit"] = "m^2"
        elif raw.endswith("m") and not any(c in raw for c in ("cm", "km", "mm")):
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

        if any(kw in name_lower for kw in ("speed", "velocity", "速度", "速率", "vx", "vy", "vz")):
            result["unit"] = "m/s"
            result["expected_unit"] = "m/s"
        elif any(kw in name_lower for kw in ("area", "面积", "区域")):
            result["unit"] = "m^2"
            result["expected_unit"] = "m^2"
        elif any(kw in name_lower for kw in ("distance", "length", "width", "height", "depth",
                                                "距离", "长度", "宽度", "高度", "深度",
                                                "x", "y", "z", "pos", "position", "radius", "r")):
            result["unit"] = "m"
            result["expected_unit"] = "m"
        elif any(kw in name_lower for kw in ("time", "duration", "时间", "时长", "period", "t", "dt")):
            result["unit"] = "s"
            result["expected_unit"] = "s"
        elif any(kw in name_lower for kw in ("mass", "质量", "weight", "重量", "m_")):
            result["unit"] = "kg"
            result["expected_unit"] = "kg"
        elif any(kw in name_lower for kw in ("angle", "角度", "theta", "alpha", "beta", "gamma", "phi")):
            result["unit"] = "rad"
            result["expected_unit"] = "rad"

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
        value = constraint.get("value", "")
        name = constraint.get("name", "")
        unit = constraint.get("unit", "")
        if value and name:

            escaped_value = re.escape(value)
            pattern = re.compile(
                rf'{re.escape(name)}\s*[=＝:：]?\s*{escaped_value}\s*{re.escape(unit)}'
                if unit else
                rf'{re.escape(name)}.*?{escaped_value}'
            )
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

        nan_count = 0
        nan_keywords = ["nan", "inf", "-inf", "infinity", "-infinity", "none", "null"]
        results_lower = results.lower()
        for kw in nan_keywords:
            nan_count += results_lower.count(kw)
        if nan_count > 0:
            findings.append(f"P0-数值异常: 结果中包含 {nan_count} 个 NaN/Inf/None 值")

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

    def sensitivity_check(self, results: str) -> Dict[str, Any]:
        """检查敏感性分析结果的合理性"""
        findings = []

        sensitivity_keywords = ["敏感性", "sensitivity", "参数扫描", "parameter sweep"]
        has_sensitivity = any(kw in results.lower() for kw in sensitivity_keywords)

        if not has_sensitivity:
            findings.append("P2-缺失: 结果中未检测到敏感性分析")

        csv_pattern = r'sensitivity.*?\.csv'
        has_csv = bool(re.search(csv_pattern, results, re.IGNORECASE))
        if not has_csv and has_sensitivity:
            findings.append("P2-缺失: 敏感性分析未输出CSV文件")

        status = "PASS" if not findings else "FAIL"
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