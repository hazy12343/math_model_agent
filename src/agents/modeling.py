from pathlib import Path
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from src.config import AppConfig
from src.agents.base import BaseAgent
from src.tools.file_reader import FileReaderTool


class ModelingAgent(BaseAgent):
    def __init__(self, config: AppConfig):
        super().__init__(config, "建模手")
        self.file_reader = FileReaderTool(config.skill_root)

    def load_system_prompt(self) -> str:
        parts = []

        skill_content = self._load_role_skill("references/roles/建模手/SKILL.md")
        parts.append(f"# 你的角色：建模手\n\n{skill_content}")

        workflow = self._load_reference("references/roles/建模手/references/工作流程.md")
        if workflow:
            parts.append(f"\n\n# 工作流程\n\n{workflow}")

        design_theory = self._load_reference("references/roles/建模手/references/建模设计理论.md")
        if design_theory:
            parts.append(f"\n\n# 建模设计理论\n\n{design_theory}")

        common_patterns = self._load_reference("references/roles/建模手/references/常见模式.md")
        if common_patterns:
            parts.append(f"\n\n# 常见模式\n\n{common_patterns}")

        algo_index = self._load_reference("references/算法索引.md")
        if algo_index:
            parts.append(f"\n\n# 可用算法索引\n\n{algo_index}")

        parts.append(f"""
\n\n# 当前任务配置
- 竞赛类型：{self.config.competition}
- 语言：{self.config.language}
- 项目根目录：{{project_root}}

# 核心任务
1. 仔细阅读题目和所有附件
2. 建立问题、目标、约束、数据字段和输出要求清单
3. 数据驱动地检查缺失、异常、量纲、时间和空间范围
4. 为每个子问题设计模型方案
5. 输出两个固定产物：题目分析报告.md 和 术语表格.md

# 重要规则
- 每道子问题**必须提供至少 2 种不同思路的模型**（如：解析法 vs 数值法、确定性 vs 随机性、连续 vs 离散）
- 同一物理机制的不同近似精度按一个模型族计数
- 模型必须覆盖题目约束、数据特征和评价目标
- 避免直接套用过于常见的简单模型来冒充创新
- 选模型时先读取算法索引，再按需加载具体算法说明

# 多模型对比要求（国赛关键）
- 对每个子问题，必须提出至少 2 种建模思路，并在报告中建立对比表格：
  | 对比项 | 模型A（主模型） | 模型B（对比模型） |
  |--------|----------------|------------------|
  | 建模思路 | ... | ... |
  | 适用条件 | ... | ... |
  | 计算复杂度 | ... | ... |
  | 预期精度 | ... | ... |
  | 优势 | ... | ... |
  | 劣势 | ... | ... |
- 主模型用于最终的求解和优化，对比模型用于验证和讨论
- 如果两种模型结论一致，说明结果可靠；如果不一致，分析原因

# 模型创新要求（关键）
- 对每个子问题，在标准模型基础上提出至少1个创新改进点
- 创新可以来自：模型简化（巧妙的近似）、算法改进（收敛加速）、问题转化（将复杂约束转化为等价形式）、多目标权衡（非平凡权重设计）
- 创新点必须有明确的数学表达，不能只是概念描述
- 在报告中用"创新点"标签标注每个创新
- 如果题目有陷阱（如非标准单位、隐含约束），必须在报告中显式处理

# 理论推导深度要求（国赛关键）
- 每个重要公式必须给出推导步骤或推导依据
- 模型简化时必须给出"量级分析"：说明为什么可以忽略某些因素（如：Re >> 1，惯性力远大于粘性力，故忽略粘性）
- 量纲一致性必须显式说明：每个公式左右两侧的量纲是否一致
- 增加"模型适用性分析"章节：说明什么条件下模型失效、边界在哪里
- 展示 Buckingham π 定理应用（如适用）

# 误差分析要求（国赛关键）
- 在报告中增加"误差分析"章节，包含误差来源表：
  | 误差来源 | 类型 | 影响程度 | 减缓措施 |
  |----------|------|----------|----------|
  | 测量误差 | 随机/系统 | ... | ... |
  | 模型简化误差 | 系统 | ... | ... |
  | 数值计算误差 | 随机 | ... | ... |
- 给出置信区间估计方法，如有可能给出具体区间
""")
        return "\n".join(parts)

    def get_tools(self) -> List[BaseTool]:
        return [
            self.file_reader.read_pdf_tool,
            self.file_reader.read_excel_tool,
            self.file_reader.read_text_tool,
            self._load_algo_tool,
        ]

    @property
    def _load_algo_tool(self):
        @tool
        def load_algorithm_info(algorithm_category: str) -> str:
            """加载指定类别的算法说明文档。可选值：优化、预测、评价、图论、统计、综合、机器学习、遗传算法、粒子群优化、模拟退火、假设检验、回归分析"""
            category_map = {
                "优化": "01-优化算法说明.md",
                "预测": "02-预测类算法说明.md",
                "评价": "03-评价类算法说明.md",
                "图论": "04-图论与网络分析算法说明.md",
                "统计": "05-统计分析与数据处理算法说明.md",
                "综合": "06-综合类算法说明.md",
                "机器学习": "07-机器学习算法说明.md",
                "遗传算法": "08-遗传算法说明.md",
                "粒子群优化": "09-粒子群优化算法说明.md",
                "模拟退火": "10-模拟退火算法说明.md",
                "假设检验": "11-假设检验说明.md",
                "回归分析": "12-回归分析说明.md",
            }
            filename = category_map.get(algorithm_category)
            if not filename:
                return f"未找到类别 '{algorithm_category}'，可选值：{list(category_map.keys())}"
            return self._load_algorithm(filename)
        return load_algorithm_info

    def analyze_problem(
        self,
        problem_description: str,
        problem_files: List[str],
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)

        file_contents = []
        for f in problem_files:
            content = self._read_file(Path(f))
            file_contents.append(f"--- 文件：{f} ---\n{content[:3000]}")

        file_contents_text = "\n".join(file_contents)

        user_msg = f"""请分析以下数学建模题目：

## 题目描述
{problem_description}

## 附件内容
{file_contents_text}

请按照建模手的工作流程，完成以下任务：
1. 理解题目和附件
2. 拆分子问题
3. 为每个子问题选择合适的模型和算法
4. 生成题目分析报告.md 和 术语表格.md 的内容

请直接输出分析报告内容，格式使用 Markdown。"""
        return self.invoke(messages, user_input=user_msg)

    def fix_model(
        self,
        feedback: str,
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)
        user_msg = f"""编程手反馈了以下问题，请修正模型设计：

## 反馈
{feedback}

请修正题目分析报告.md 和 术语表格.md，只输出修正后的完整内容。"""
        return self.invoke(messages, user_input=user_msg)