from pathlib import Path
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from src.config import AppConfig
from src.agents.base import BaseAgent
from src.tools.file_reader import FileReaderTool
from src.tools.paper_search import PaperSearchTool


class WritingAgent(BaseAgent):
    def __init__(self, config: AppConfig):
        super().__init__(config, "论文手")
        self.file_reader = FileReaderTool(config.skill_root)
        self.paper_search = PaperSearchTool(config.skill_root)

    def load_system_prompt(self) -> str:
        parts = []

        skill_content = self._load_role_skill("references/roles/论文手/SKILL.md")
        parts.append(f"# 你的角色：论文手\n\n{skill_content}")

        workflow = self._load_reference("references/roles/论文手/references/工作流程.md")
        if workflow:
            parts.append(f"\n\n# 工作流程\n\n{workflow}")

        chapter_template = self._load_reference("references/roles/论文手/references/章节模板.md")
        if chapter_template:
            parts.append(f"\n\n# 章节模板\n\n{chapter_template}")

        writing_spec = self._load_reference("references/roles/论文手/references/写作规范.md")
        if writing_spec:
            parts.append(f"\n\n# 写作规范\n\n{writing_spec}")

        parts.append(f"""
\n\n# 当前任务配置
- 竞赛类型：{self.config.competition}
- 语言：{self.config.language}
- 项目根目录：{{project_root}}

# 核心任务
1. 读取题目、建模分析、代码结果和图表
2. 建立 Claim-Evidence 映射
3. 撰写完整论文
4. 默认同时生成 Word 和 LaTeX/PDF 两种格式

# 重要规则
- 所有结论必须有真实数据支撑
- 图表必须来自实际代码输出
- 引用必须可追溯
- 两种格式论文内容必须一致
- 默认至少8幅正式图

# 论文创新要求（关键）
- 在"模型建立"部分明确标注每个创新点，并说明创新来源（模型简化/算法改进/问题转化/多目标权衡）
- 创新点必须有数学公式支撑，不能只是文字描述
- 在"模型评价"部分对比标准方法和创新方法的差异
- 如果建模报告中有"模型风险提示"，论文中必须回应对这些风险的解决方案
""")
        return "\n".join(parts)

    def get_tools(self) -> List[BaseTool]:
        return [
            self.file_reader.read_pdf_tool,
            self.file_reader.read_excel_tool,
            self.file_reader.read_text_tool,
            self.paper_search.search_tool,
        ]

    def build_evidence_outline(
        self,
        modeling_report: str,
        code_results: str,
        figure_list: str,
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)

        user_msg = f"""请建立证据大纲（W1阶段）：

## 题目分析报告
{modeling_report[:5000]}

## 代码结果
{code_results[:5000]}

## 图表清单
{figure_list[:3000]}

请为每个子问题建立 Claim-Evidence 映射：
1. 核心主张
2. 支撑公式
3. 结果表位置
4. 拟用图表
5. 代码输出或文献支撑

输出证据大纲，格式使用 Markdown。"""
        return self.invoke(messages, user_input=user_msg)

    def write_paper(
        self,
        modeling_report: str,
        code_results: str,
        figure_list: str,
        evidence_outline: str,
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)

        user_msg = f"""W1已通过，请撰写完整论文（W2阶段）：

## 题目分析报告
{modeling_report[:5000]}

## 代码结果
{code_results[:5000] if code_results else '（代码尚未成功运行，请在论文中用"待计算"标注数值结果，但仍需完成论文的全部结构和推导）'}

## 图表清单
{figure_list[:3000]}

## 证据大纲
{evidence_outline[:3000]}

请完成：
1. 按官方结构撰写完整正文
2. 生成 Word 论文内容
3. 生成 LaTeX 源码
4. 确保两种格式内容一致

**重要**：如果代码结果不可用，请在数值位置标注"待计算"并注明"结果将在代码运行后填入"，但论文的模型推导、公式、算法步骤必须完整。"""
        return self.invoke(messages, user_input=user_msg)

    def fix_paper(
        self,
        feedback: str,
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)
        user_msg = f"""质检反馈了以下问题，请修正论文：

## 反馈
{feedback}

请修正论文并重新输出。"""
        return self.invoke(messages, user_input=user_msg)