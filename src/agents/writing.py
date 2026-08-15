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
3. 撰写完整论文（Markdown 格式，公式使用标准 LaTeX 语法）
4. 论文末尾附 Pandoc 转换命令，方便一键生成 Word 文档

# Word 格式生成说明
- 你输出的是 Markdown 文件，公式使用 LaTeX 语法（`$...$` 行内、`$$...$$` 独立）
- **LaTeX 公式可以零修改转换为 Word 公式**：使用 Pandoc 命令即可自动转换，无需手动编辑
- 论文末尾必须输出以下转换命令（根据实际文件名调整）：
  ```
  [Word转换]
  pandoc 完整论文.md -o 完整论文.docx --from markdown --to docx --mathjax
  ```
- 转换后 `\mathbf` → Word 粗体、`\mathrm` → Word 正体、`\frac{{}}{{}}` → Word 分数、`\sum` → Word 求和符号，全部自动完成

# 重要规则
- 所有结论必须有真实数据支撑
- 图表必须来自实际代码输出，并在正文中引用（如"如图1所示"、"见表2"）
- 引用必须可追溯
- 两种格式论文内容必须一致
- 默认至少8幅正式图
- **⚠️ 如果代码执行结果中包含"代码执行失败"或"严重警告"标记，说明代码未成功运行，此时论文中绝对不能编造任何数值！应如实说明代码执行状态，等待代码修复后重新生成论文。**

# 图表引用强制要求（国赛关键）
- **论文中每张图、每个表必须在正文中被至少引用一次**
- 引用格式：如图1所示、见表2、图3展示了...、从表4可以看出...
- 生成论文后，必须自检：列出所有图表，确认每个图表都在正文中有引用
- 如果图表未被引用，必须删除或添加引用文字
- 图表编号必须连续（图1、图2、图3...），不能跳号

# 论文章节结构强制要求（国赛关键）
- 论文必须包含以下标准章节（顺序不可变）：
  1. 摘要（含关键词）
  2. 问题重述与分析
  3. 模型假设与符号说明
  4. 模型建立（核心章节，含公式推导、模型对比、创新点标注）
  5. 模型求解（核心章节，含算法描述、求解结果、收敛性分析）
  6. 结果分析（含敏感性分析、误差分析、模型对比分析）
  7. 模型评价与改进（优点、缺点、改进方向）
  8. 结论
  9. 参考文献（至少5篇）
- 如果缺少任何章节，标记为不合格
- 摘要中必须包含：问题背景、建模方法、主要结果（具体数值）、创新点、关键词

# 论文创新要求（关键）
- 在"模型建立"部分明确标注每个创新点，并说明创新来源（模型简化/算法改进/问题转化/多目标权衡）
- 创新点必须有数学公式支撑，不能只是文字描述
- 在"模型评价"部分对比标准方法和创新方法的差异
- 如果建模报告中有"模型风险提示"，论文中必须回应对这些风险的解决方案

# 参考文献要求（关键）
- 论文结尾必须包含参考文献列表，至少 5 篇
- 参考文献应包括：
  - 1-2 篇经典方法文献（如数值分析、优化理论、概率统计等相关标准教材）
  - 1-2 篇题目领域相关文献（如题目所属领域的相关研究文献）
  - 1-2 篇算法方法文献（如网格搜索、遗传算法、粒子群等优化方法）
- 参考文献格式：[编号] 作者. 标题. 期刊/出版社, 年份.
- 可以使用 `paper_search` 工具搜索相关学术文献
- 题目给出的基本物理定律和数学公式属于已知常识，无需引用外部文献
""")
        parts.append("""
# LaTeX 公式格式规范（关键！）
论文中所有数学公式必须遵循以下格式要求：

## 公式分隔符
- **行内公式**：使用 `$...$`（LaTeX 中）或 `\\(...\\)`（Markdown 中）
- **独立公式（display math）**：使用 `$$...$$`（Markdown 中推荐）或 `\\[...\\]`（LaTeX 中）
- 长公式需要换行时使用 `\\begin{aligned}` 等环境

## 公式编号（重要！）
- **所有独立公式必须编号**，格式为 `(1)`、`(2)`、`(3)`...
- 在 Markdown 中：`$$ ... \\tag{1} $$`
- 在 LaTeX 中：`\\begin{equation} ... \\label{eq:xxx} \\end{equation}`
- 正文中引用公式时使用编号，如"由式(3)可得..."

## 下标/上标中的文字
- 使用 `\\mathrm{...}` 而非 `\\text{...}`（`\\text` 需要 amsmath 宏包，可移植性差）
- 正确：`\\mathbf P^{\\mathrm{drop}}_{j,k}`、`T_{\\mathrm{cover},i}`
- 错误：`\\mathbf P^{\\text{drop}}_{j,k}`、`T_{\\text{cover},i}`

## 分数
- 必须使用完整形式 `\\frac{分子}{分母}`，禁止简写
- 正确：`\\frac{1}{2}`、`\\frac{\\partial f}{\\partial x}`
- 错误：`\\frac12`、`\\frac1n`（虽然某些编译器能解析，但不符合 LaTeX 规范，且跨编译器兼容性差）

## 范数/绝对值
- 使用 `\\lVert ... \\rVert` 表示范数，`\\lvert ... \\rvert` 表示绝对值
- 或使用 `\\left\\| ... \\right\\|` / `\\left| ... \\right|`
- 正确：`\\lVert \\mathbf{M}_{i0} \\rVert`
- 避免：`||\\mathbf{M}_{i0}||`（双竖线间距不正确）

## 向量/矩阵
- 向量使用 `\\mathbf{v}` 或 `\\boldsymbol{v}`（希腊字母必须用 `\\boldsymbol`）
- 矩阵使用 `\\mathbf{A}` 或 `\\boldsymbol{A}`
- 转置使用 `^{\\mathsf{T}}` 或 `^{\\top}`

## 常见符号
- 微分算子：`\\mathrm{d}` 而非 `d`（如 `\\int f(x) \\,\\mathrm{d}x`）
- 自然对数底：`\\mathrm{e}` 而非 `e`（当表示常数时）
- 虚数单位：`\\mathrm{i}` 而非 `i`
- 最大化/最小化：`\\max`、`\\min`（正体），下标用 `\\max_{x \\in X}`
- 求和/求积：`\\sum_{i=1}^{n}`、`\\prod_{i=1}^{n}`

## 禁止模式
- ❌ `\\frac12` → ✅ `\\frac{1}{2}`
- ❌ `\\text{xxx}` 在纯数学上下文中 → ✅ `\\mathrm{xxx}`
- ❌ `||x||` 表示范数 → ✅ `\\lVert x \\rVert`
- ❌ 独立公式不加编号 → ✅ 所有独立公式加编号
- ❌ 中文出现在数学模式中 → ✅ 中文放在公式外，用 `\\text{中文}` 仅限必要情况
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

        # 检测用户聚焦指令
        is_focused = "用户聚焦指令" in modeling_report
        if is_focused:
            prompt += "\n\n# ⚠️ 最高优先级指令\n用户明确要求只撰写特定子问题的论文。你必须严格遵守：只撰写该子问题的论文，不要涉及其他子问题！\n\n# ⚠️ 防重复指令\n禁止在输出中重复相同的段落！每段内容只写一次，不要大段重复！"

        user_msg = f"""请建立证据大纲（W1阶段）：

## 题目分析报告
{modeling_report[:5000]}

## 代码结果
{code_results[:5000]}

## 图表清单
{figure_list[:3000]}

请为{"该子问题" if is_focused else "每个子问题"}建立 Claim-Evidence 映射：
1. 核心主张
2. 支撑公式
3. 结果表位置
4. 拟用图表
5. 代码输出或文献支撑

输出证据大纲，格式使用 Markdown。"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt)

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

        # 检测用户聚焦指令
        if "用户聚焦指令" in modeling_report:
            prompt += "\n\n# ⚠️ 最高优先级指令\n用户明确要求只撰写特定子问题的论文。你必须严格遵守：只撰写该子问题的论文，不要涉及其他子问题！\n\n# ⚠️ 防重复指令\n禁止在论文中重复输出相同的段落！每段内容只写一次，不要大段重复！如果发现自己陷入重复，立即切换到下一个章节。"

        # 搜索相关参考文献（基于题目内容动态生成搜索词）
        references = ""
        try:
            # 从建模报告中提取关键主题词（取前100个字符中的关键词）
            topic_keywords = []
            report_lower = modeling_report.lower()
            for kw in ["优化", "预测", "分类", "聚类", "评价", "调度", "路径", "规划",
                        "optimization", "prediction", "classification", "clustering",
                        "scheduling", "routing", "planning", "仿真", "simulation",
                        "无人机", "车辆", "信号", "图像", "网络", "资源", "调度",
                        "uav", "drone", "vehicle", "signal", "image", "network"]:
                if kw in report_lower:
                    topic_keywords.append(kw)
            if not topic_keywords:
                topic_keywords = ["数学建模", "优化"]
            search_query = "数学建模 " + " ".join(topic_keywords[:5])
            search_func = self.paper_search.search_tool
            search_results = search_func.invoke({
                "query": search_query,
                "limit": 5
            })
            if search_results:
                references = "\n\n## 搜索到的参考文献\n\n" + str(search_results)[:2000]
        except Exception:
            pass

        user_msg = f"""W1已通过，请撰写完整论文（W2阶段）：

## 题目分析报告
{modeling_report[:5000]}

## 代码结果
{code_results[:5000] if code_results else '（代码尚未成功运行，请生成完整的论文结构和推导，但**不得在结果表中填入任何数值（包括"待计算"），直接留空或标注"见代码输出"即可**）'}

## 图表清单
{figure_list[:3000]}

## 证据大纲
{evidence_outline[:3000]}

{references}

请完成：
1. 按官方结构撰写完整正文
2. 生成 Word 论文内容
3. 生成 LaTeX 源码
4. 确保两种格式内容一致

**重要**：如果代码结果上方有"严重警告"标记，说明代码未执行成功，此时论文中绝对不能编造任何数值，结果表格保留结构但数值列留空。如果代码结果可用，则如实填入数值。
**参考文献**：论文结尾必须包含参考文献列表，至少 5 篇。使用搜索到的参考文献和标准教材作为引用来源。"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt)

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

请修正论文并重新输出。

⚠️ **禁止大段重复段落！** 每段内容只写一次，不要复制粘贴相同的内容！"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt, use_fix_model=True)