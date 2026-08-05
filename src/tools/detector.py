import re
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

    def _check_csv_anomalies(self, content: str, findings: List[str]):
        """检查CSV数据异常"""
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return

        header = lines[0]
        data_lines = lines[1:]
        num_cols = len(header.split(","))

        for i, line in enumerate(data_lines):
            cols = line.split(",")
            if len(cols) != num_cols:
                findings.append(
                    f"P1-数据异常: 第{i+2}行列数({len(cols)})与表头({num_cols})不一致"
                )

        for j in range(num_cols):
            col_values = []
            for line in data_lines:
                cols = line.split(",")
                if j < len(cols):
                    val = cols[j].strip().strip('"')
                    try:
                        col_values.append(float(val))
                    except ValueError:
                        pass

            if col_values:
                if len(set(col_values)) == 1 and len(col_values) > 2:
                    col_name = header.split(",")[j].strip() if j < len(header.split(",")) else f"列{j+1}"
                    findings.append(
                        f"P1-数据异常: 列'{col_name}'所有值相同({col_values[0]})，可能为常量列"
                    )

    def _check_missing_values(self, content: str, findings: List[str]):
        """检查缺失值"""
        missing_indicators = ["NA", "N/A", "null", "NULL", "None", "NaN", "nan", "", "-", "--"]
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return

        missing_count = 0
        for line in lines[1:]:
            cols = line.split(",")
            for col in cols:
                val = col.strip().strip('"').strip()
                if val in missing_indicators:
                    missing_count += 1

        if missing_count > 0:
            findings.append(
                f"P1-缺失值: 数据中包含 {missing_count} 个缺失值，"
                f"需要明确处理策略（插值/删除/标记）"
            )

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

        if re.search(r'1e-?\d+\s*[+\-*/]', code):
            findings.append("P2-硬编码: 代码中包含硬编码的数值容差，建议定义为常量")

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