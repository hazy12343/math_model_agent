import re
import csv
import io
from typing import List, Dict, Any, Optional
from pathlib import Path


class TrapDetector:
    """陷阱检测系统：数据异常、约束遗漏、数值陷阱"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

    def detect_data_anomalies(self, problem_description: str, attachment_content: str) -> Dict[str, Any]:
        """检测数据中的异常"""
        findings = []

        if attachment_content:
            if self._looks_like_csv(attachment_content):
                self._check_csv_anomalies(attachment_content, findings)
                self._check_missing_values(attachment_content, findings)

        self._check_implicit_constraints(problem_description, findings)
        self._check_dimensional_traps(problem_description, findings)

        if not findings:
            findings.append("未检测到明显的数据陷阱")

        status = "PASS" if not any("P0" in f for f in findings) else "FAIL"
        return {
            "status": status,
            "findings": findings,
            "anomaly_count": len([f for f in findings if "P0" in f or "P1" in f]),
        }

    def _looks_like_csv(self, content: str) -> bool:
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return False
        first_line = lines[0]
        comma_count = first_line.count(",")
        tab_count = first_line.count("\t")
        if comma_count >= 1 or tab_count >= 1:
            for line in lines[1:5]:
                if line.strip():
                    if comma_count >= 1:
                        if line.count(",") != comma_count:
                            return False
                    if tab_count >= 1:
                        if line.count("\t") != tab_count:
                            return False
            return True
        return False

    def _check_csv_anomalies(self, content: str, findings: List[str]):
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return

        try:
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
            if len(rows) < 2:
                return
            header = rows[0]
            data_rows = rows[1:]
            num_cols = len(header)

            for i, row in enumerate(data_rows):
                if len(row) != num_cols:
                    findings.append(
                        f"P1-数据异常: 第{i+2}行列数({len(row)})与表头({num_cols})不一致"
                    )

            for j in range(num_cols):
                col_values = []
                for row in data_rows:
                    if j < len(row):
                        val = row[j].strip().strip('"')
                        try:
                            col_values.append(float(val))
                        except ValueError:
                            pass

                if col_values:
                    if len(set(col_values)) == 1 and len(col_values) > 2:
                        col_name = header[j].strip() if j < len(header) else f"列{j+1}"
                        findings.append(
                            f"P1-数据异常: 列'{col_name}'所有值相同({col_values[0]})，可能为常量列"
                        )
        except Exception:
            pass

    def _check_missing_values(self, content: str, findings: List[str]):
        missing_indicators = ["NA", "N/A", "null", "NULL", "None", "NaN", "nan", "", "-", "--"]
        try:
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
            if len(rows) < 2:
                return

            missing_count = 0
            for row in rows[1:]:
                for val in row:
                    clean = val.strip().strip('"').strip()
                    if clean in missing_indicators:
                        missing_count += 1

            if missing_count > 0:
                findings.append(
                    f"P1-缺失值: 数据中包含 {missing_count} 个缺失值，"
                    f"需要明确处理策略（插值/删除/标记）"
                )
        except Exception:
            pass

    def _check_implicit_constraints(self, problem: str, findings: List[str]):
        """检查隐式约束"""
        implicit_patterns = [
            (r'(?:实际|现实|真实|物理).*?(?:不能|不会|不可)', "P0-物理约束: 题目隐含物理限制条件，需在模型中显式处理"),
            (r'(?:整数|正整数|自然数|偶数|奇数)', "P1-整数约束: 题目要求整数解，需使用整数规划或取整策略"),
            (r'(?:单调|递增|递减|非负|非正)', "P1-单调性约束: 题目要求单调性，需在模型中加入约束"),
            (r'(?:对称|镜像|周期|循环)', "P1-对称性: 题目暗示对称性，可利用简化模型"),
        ]
        for pattern, msg in implicit_patterns:
            if re.search(pattern, problem):
                findings.append(msg)

    def _check_dimensional_traps(self, problem: str, findings: List[str]):
        """检查量纲陷阱"""
        unit_patterns = re.findall(r'([\d.]+)\s*(km|千米|公里|cm|厘米|mm|毫米|dm|分米)', problem)
        if unit_patterns:
            findings.append(
                "P1-单位陷阱: 题目使用了非标准单位（km/cm/mm），"
                "需统一转换为SI单位（m）后再计算"
            )

        time_patterns = re.findall(r'([\d.]+)\s*(min|分钟|h|小时|day|天|年|year)', problem)
        if time_patterns:
            findings.append(
                "P1-时间单位: 题目使用了非秒时间单位，"
                "需统一转换为秒（s）后再计算"
            )

    def detect_model_risks(self, modeling_report: str) -> Dict[str, Any]:
        """检测模型选择中的潜在风险"""
        findings = []

        risk_patterns = [
            (r'线性回归|最小二乘|linear regression', "P1-线性假设: 使用线性模型，需验证数据是否满足线性假设"),
            (r'正态分布|高斯分布|normal distribution', "P1-正态假设: 假设正态分布，需进行正态性检验"),
            (r'K-means|K均值|kmeans', "P1-K-means: 需说明K值选择依据，对初始中心敏感"),
            (r'神经网络|深度学习|neural network|deep learning', "P1-神经网络: 可解释性差，需说明为何选择此方法而非更简洁的模型"),
            (r'蒙特卡洛|Monte Carlo|monte carlo', "P1-蒙特卡洛: 需说明收敛性判断标准和样本量选择依据"),
        ]
        for pattern, msg in risk_patterns:
            if re.search(pattern, modeling_report, re.IGNORECASE):
                findings.append(msg)

        if not findings:
            findings.append("未检测到明显的模型选择风险")

        return {
            "status": "PASS" if not any("P0" in f for f in findings) else "FAIL",
            "findings": findings,
            "risk_count": len([f for f in findings if "P0" in f or "P1" in f]),
        }

    def detect_numerical_traps(self, code: str) -> Dict[str, Any]:
        """检测代码中的数值陷阱"""
        findings = []

        if re.search(r'==\s*[\d.]+', code):
            findings.append("P1-浮点比较: 代码中使用 == 比较浮点数，应使用 np.isclose() 或容差比较")

        if re.search(r'(?:tol|eps|atol|rtol|epsilon|threshold|tolerance)\s*=\s*1e-?\d+', code):
            findings.append("P2-硬编码容差: 代码中硬编码了数值容差（如 tol=1e-6），建议定义为常量并标注选择依据")

        if "for " in code and "range(" in code:
            loops = re.findall(r'for.*?in\s+range\((\d+)\)', code)
            if loops:
                large_loops = [int(n) for n in loops if int(n) > 1000]
                if large_loops:
                    findings.append(
                        f"P2-性能: 存在大循环(range({max(large_loops)}))，"
                        f"考虑是否可用向量化运算替代"
                    )

        if re.search(r'np\.append|\.append\(', code) and "for " in code:
            findings.append("P2-性能: 循环中使用append，建议预分配数组或使用列表推导式")

        if not findings:
            findings.append("未检测到明显的数值陷阱")

        return {
            "status": "PASS" if not any("P0" in f for f in findings) else "FAIL",
            "findings": findings,
            "trap_count": len([f for f in findings if "P0" in f or "P1" in f]),
        }

    def detect_result_anomalies(self, output: str, code: str) -> Dict[str, Any]:
        """检测执行结果中的异常模式"""
        findings = []
        output_lower = output.lower()

        # 1. 检测所有结果全为 0
        zero_patterns = [
            r'最优[值解].*?[=:]\s*0\.?0?',
            r'[=:]\s*0\.0+',
            r'全部为\s*0',
        ]
        zero_count = 0
        for pat in zero_patterns:
            zero_count += len(re.findall(pat, output))
        if zero_count >= 2:
            findings.append("P0-结果异常: 检测到多个结果为零，可能算法实现有误或约束过强")

        # 2. 检测 NaN/Inf 出现在结果输出中（非代码行）
        nan_in_output = False
        for line in output.split("\n"):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["nan", "inf", "-inf"]):
                if any(c in line for c in ("=", ":", "结果", "value", "output", "最优", "最佳")):
                    nan_in_output = True
                    break
        if nan_in_output:
            findings.append("P0-数值异常: 结果输出中包含 NaN/Inf 值，代码可能存在除零或数值溢出")

        # 3. 检测结果是否在合理范围内
        # 提取所有浮点数及其上下文
        for line in output.split("\n"):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["负值", "负数", "negative", "error", "错误", "失败"]):
                if any(c in line for c in ("=", ":", "结果", "输出", "value")):
                    findings.append(f"P1-结果异常: 检测到负值/错误提示: {line.strip()[:80]}")
                    break

        # 4. 检测迭代类算法的收敛性问题
        has_iteration = bool(re.search(r'迭代|iteration|代数|generation|epoch', output_lower))
        has_convergence = bool(re.search(r'收敛|converge|converged', output_lower))
        if has_iteration and not has_convergence:
            findings.append("P1-收敛性: 代码包含迭代算法但未输出收敛状态，无法判断是否收敛")

        # 5. 检测蒙特卡洛验证
        has_mc = bool(re.search(r'蒙特卡洛|monte carlo|随机模拟', output_lower))
        if not has_mc:
            findings.append("P1-蒙特卡洛: 未检测到蒙特卡洛验证，国赛建议对关键参数进行随机扰动验证")

        # 6. 检测结果数量级是否合理（对常见物理量）
        has_distance = bool(re.search(r'距离|distance|位移|disp', output_lower))
        if has_distance:
            # 提取距离值
            dist_values = re.findall(r'(?:距离|distance|位移)[^:]*?[=:]\s*(\d+\.?\d*)', output)
            if dist_values:
                for val in dist_values:
                    try:
                        fval = float(val)
                        if fval > 1e6:
                            findings.append(f"P1-量级异常: 距离值 {fval} 过大，可能单位转换错误")
                        elif fval < 1e-6:
                            findings.append(f"P1-量级异常: 距离值 {fval} 过小，可能单位转换错误")
                    except ValueError:
                        pass

        if not findings:
            findings.append("未检测到明显的结果异常")

        return {
            "status": "PASS" if not any("P0" in f for f in findings) else "FAIL",
            "findings": findings,
            "anomaly_count": len([f for f in findings if "P0" in f or "P1" in f]),
        }