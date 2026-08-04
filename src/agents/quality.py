from typing import List, Optional
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from src.config import AppConfig
from src.agents.base import BaseAgent


class QualityCheckAgent(BaseAgent):
    def __init__(self, config: AppConfig):
        super().__init__(config, "质检Subagent")

    def load_system_prompt(self) -> str:
        skill_root = self.config.skill_root
        subagent_doc = self._read_file(skill_root / "references/Subagent调度.md")
        return f"""# 你的角色：独立质检 Subagent

你是只读质检员，未参与任何产物编写。你只返回证据，不修改任何文件。

{subagent_doc}

# 核心规则
1. 只读核对，不修改产物
2. 返回证据回执：范围、输入快照、状态(PASS/FAIL/BLOCKED)、证据、发现、返工建议
3. 存在未解决的P0问题（正确性/官方硬约束）时返回FAIL；仅有P1、P2问题时返回PASS并注明
4. 缺少依赖、权限或可验证证据时返回BLOCKED
5. 不要被告知"主Agent认为已经正确"，独立判断
6. 如果产物满足核心要求，即使有小的改进空间，也应返回PASS"""

    def _read_problem_files(self, problem_files: List[str]) -> str:
        if not problem_files:
            return ""
        parts = []
        for f in problem_files:
            content = self._read_file(Path(f))
            # 只取前 6000 字符，避免 token 爆炸
            parts.append(f"--- 附件: {Path(f).name} ---\n{content[:6000]}")
        return "\n\n".join(parts)

    def check_m1(
        self,
        modeling_report: str,
        terminology_table: str,
        problem_description: str,
        messages: List[BaseMessage],
        problem_files: List[str] = None,
    ) -> str:
        problem_files_str = problem_files or []
        attachment_content = self._read_problem_files(problem_files_str)

        user_msg = f"""执行 M1 建模终检：

## 题目原文
{problem_description[:8000]}

## 附件内容
{attachment_content[:8000]}

## 题目分析报告
{modeling_report[:8000]}

## 术语表格
{terminology_table[:5000]}

请核对：
1. 子问题是否全部覆盖
2. 假设是否有依据
3. 公式与符号是否一致
4. 单位与约束是否明确
5. 模型数量是否合规（每子问题≤2个独立模型）
6. 模型是否可实现
7. 验证方案是否完备
8. 文献是否可追溯

请返回标准回执。"""
        return self.invoke(messages, user_input=user_msg)

    def check_p1(
        self,
        code: str,
        results: str,
        modeling_report: str,
        messages: List[BaseMessage],
        problem_files: List[str] = None,
    ) -> str:
        problem_files_str = problem_files or []
        attachment_content = self._read_problem_files(problem_files_str)

        user_msg = f"""执行 P1 最小可运行结果门禁：

## 附件内容
{attachment_content[:5000]}

## 建模报告
{modeling_report[:5000]}

## 代码
{code[:8000]}

## 运行结果
{results[:5000]}

请核对：
1. 代码是否可执行
2. 退出码是否正常
3. 输入到结果的追溯是否完整
4. 单位、数值范围是否正确
5. 关键约束是否满足
6. 模型合同是否匹配

请返回标准回执。"""
        return self.invoke(messages, user_input=user_msg)

    def check_p2(
        self,
        code: str,
        results: str,
        figure_list: str,
        messages: List[BaseMessage],
        problem_files: List[str] = None,
    ) -> str:
        problem_files_str = problem_files or []
        attachment_content = self._read_problem_files(problem_files_str)

        user_msg = f"""执行 P2 编程终检：

## 附件内容
{attachment_content[:5000]}

## 代码
{code[:8000]}

## 结果
{results[:8000]}

## 图表清单
{figure_list[:5000]}

请核对：
1. 代码完整性
2. 结果正确性
3. 图表数量与质量
4. 复现清单完整性
5. 文件完整性

请返回标准回执。"""
        return self.invoke(messages, user_input=user_msg)

    def check_w1(
        self,
        evidence_outline: str,
        modeling_report: str,
        code_results: str,
        messages: List[BaseMessage],
        problem_files: List[str] = None,
    ) -> str:
        problem_files_str = problem_files or []
        attachment_content = self._read_problem_files(problem_files_str)

        user_msg = f"""执行 W1 证据大纲门禁：

## 附件内容
{attachment_content[:5000]}

## 建模报告
{modeling_report[:5000]}

## 代码结果
{code_results[:5000]}

## 证据大纲
{evidence_outline[:8000]}

请核对：
1. 每个子问题结论是否有精确证据路径
2. 摘要关键数值与结果表是否一致
3. 图表、公式和引用是否有章节落点
4. Word/LaTeX是否共用同一证据源

请返回标准回执。"""
        return self.invoke(messages, user_input=user_msg)

    def check_w2(
        self,
        paper_content: str,
        evidence_outline: str,
        messages: List[BaseMessage],
        problem_files: List[str] = None,
    ) -> str:
        problem_files_str = problem_files or []
        attachment_content = self._read_problem_files(problem_files_str)

        user_msg = f"""执行 W2 论文终检：

## 附件内容
{attachment_content[:5000]}

## 证据大纲
{evidence_outline[:5000]}

## 论文内容
{paper_content[:10000]}

请核对：
1. 当届规则合规性
2. 主张-证据一致性
3. 数值与单位正确性
4. 图表引用完整性
5. 文献可追溯性
6. Word/LaTeX一致性（如同时生成）

请返回标准回执。"""
        return self.invoke(messages, user_input=user_msg)