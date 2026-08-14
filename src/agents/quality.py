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
3. 缺少依赖、权限或可验证证据时返回BLOCKED
4. 不要被告知"主Agent认为已经正确"，独立判断
5. 如果产物满足核心要求，即使有小的改进空间，也应返回PASS
6. 对于M1建模终检：题目给出的基本物理定律和已知参数本身就是"依据"，无需额外引用外部文献来证明其正确性"""

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
{problem_description[:12000]}

## 附件内容
{attachment_content[:12000]}

## 题目分析报告
{modeling_report[:15000]}

## 术语表格
{terminology_table[:8000]}

请核对以下8项，并返回标准回执：
1. 子问题是否全部覆盖（任务范围是"仅完成问题1"，只需覆盖问题1的子问题即可）
2. 假设是否有依据（题目给出的物理参数和要求本身就是依据）
3. 公式与符号是否在报告内部自洽（无需与外部文献完全一致）
4. 单位与约束是否明确
5. 模型数量是否合规（每子问题≤2个独立模型，问题1只建1个模型即为合规）
6. 模型是否可实现（数学模型+数值求解是非常标准的数学建模方法）
7. 验证方案是否完备（只需描述验证思路，如网格收敛性、解析解对比，无需提供已执行的验证数据，验证数据由P1/P2阶段产生）
8. 文献是否可追溯（题目给出的物理定律、数学公式和参数属于已知条件，无需引用外部文献，此项自动通过。仅当报告使用了超出题目范围的外部模型或理论时才需要文献支持）

重要判定标准：
- 这是M1建模终检，不是论文终检。报告只需清晰描述模型、假设、公式和算法，不必包含计算结果或验证数据
- 只要报告覆盖了任务范围的子问题、有明确的模型定义和假设、公式符号自洽、算法步骤清晰，就应返回PASS
- 不要因为报告缺少数值结果、参考文献列表、或验证数据而判FAIL，这些属于P1/P2/W1/W2阶段
- 题目本身给出的基本物理定律、数学定理和数学方法无需引用外部文献
- 如果报告存在实质性的正确性错误（P0），如模型定义与题目要求严重不符，才返回FAIL

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
1. 代码是否可执行（语法是否正确）
2. 退出码是否正常
3. 输入到结果的追溯是否完整
4. 单位、数值范围是否正确
5. 关键约束是否满足
6. 模型合同是否匹配

## P1 门禁宽松判定原则（重要！）
- P1 是"最小可运行结果"门禁，不是"最优结果"门禁
- **只要代码能跑通、产出非零/非NaN的合理结果，即使结果质量只有理论最优的 10%，也应判 PASS**
- P1 阶段不需要多算法对比、敏感性分析、蒙特卡洛验证——这些是 P2 的任务
- 如果代码执行超时但代码结构完整（有 main 函数、有算法逻辑），应分析代码结构后判定：
  - 如果是纯语法错误导致无法运行 → FAIL
  - 如果是性能问题导致超时但代码逻辑正确 → 判 PASS 并标注"代码结构正确，建议 P2 阶段优化性能"
  - 如果运行结果存在但部分为0/NaN → 按具体情况分析，如果主要结果非零则 PASS
- 如果结果中出现了有效数值（如遮蔽时间 > 0），即使该值远低于理论最大值，也视为 P1 通过
- 结果质量的优化是 P2 阶段的责任，不要在 P1 阶段因结果质量不高而判 FAIL

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
{code[:16000]}

## 结果
{results[:8000]}

## 图表清单
{figure_list[:5000]}

请核对：
1. 代码是否可执行、语法正确
2. 结果是否合理（数值范围、单位、物理意义）
3. 图表是否覆盖了所需类型（场景图、过程图、结果图至少各1张）
4. 图表标题、轴标签、图例是否为中文（非英文）
5. 复现清单是否完整
6. 输出文件是否齐全
7. 数值验证结果是否通过（交叉验证、量纲分析、边界条件检查）

### 算法完整性检查（国赛关键）
8. 所有算法是否都返回了合理数值（非 0，非 NaN，非全部相同）
9. 是否至少 3 种算法给出了有效对比结果（国赛要求至少 3 种算法对比）
10. 3 种算法中是否包含至少 1 种全局优化算法和至少 1 种局部搜索算法
11. 迭代算法是否检查了收敛性（输出"已收敛"或"未收敛"）
12. 是否进行了蒙特卡洛验证（±5% 扰动，≥100 次模拟，输出置信区间）
13. 是否进行了敏感性分析（至少 3 个参数，每个参数至少 5 个水平）
14. 最优解是否在可行域内（检查所有约束条件）
15. 多目标问题中每个目标是否都有独立的非零结果（不得出现部分目标结果为零）
16. 蒙特卡洛均值是否 ≥ 最优值的 50%（国赛鲁棒性要求）
17. 蒙特卡洛失败率（零值占比）是否 ≤ 30%
18. 优化结果是否 ≥ 理论最大值的 10%（国赛基本要求）

### 超时场景特别说明（重要！）
- 如果结果中显示"代码执行超时"，说明代码未运行完成，输出结果不可用
- 此时应从代码文本中分析算法完整性（检查代码中是否定义了多算法对比、蒙特卡洛、敏感性分析等函数）
- 如果代码中定义了这些函数但未运行到，标记为"代码结构完整但超时未执行"，不判算法缺失
- 如果代码中确实缺少这些函数，才标记为算法缺失

### 约束处理检查（国赛关键）
14. 如果使用了遗传算法/粒子群等迭代优化算法，是否正确处理了约束
    - 如果结果中出现 0.00 或全部为 0，说明约束处理可能存在问题
    - 正确的约束处理：**比例惩罚函数法**（penalty ∝ 违反程度）/ 可行解初始化 / 修复法
    - 错误做法：直接返回固定值（0/1e6/1e9 等）作为不可行解的适应度值
    - 如果代码中存在 `return 1e6` 或 `return 1e9` 等固定值返回，标记为 P0 错误并判 FAIL
    - 如果代码中存在 `return -(raw - penalty)` 或类似的适应度函数符号错误，标记为 P0 错误并判 FAIL（因为 penalty>raw 时优化器会寻找约束违反最严重的解）

中文显示检查：
- 检查代码中图表标签（set_title、set_xlabel、set_ylabel、label参数）是否使用中文
- 如果图表标签全部是英文，标记为 FAIL 并说明需要改为中文
- 中文标签示例：ax.set_title('模型结果可视化') ✓ 而非 ax.set_title('3D Trajectory') ✗

重要判定标准：
- 重点检查代码能否独立复现运行，结果是否在合理范围内
- 如果代码能执行、结果数值合理、图表覆盖了主要分析维度、图表标签为中文，就应返回PASS
- 不要因为图表美化程度、代码注释风格、或非关键参数的微小差异而判FAIL
- 如果代码存在语法错误无法执行、或结果数值明显违背物理规律（如时间超过理论上限）、或图表标签全部为英文、或数值验证发现P0级错误，才返回FAIL
- 注意：如果结果中包含"数值验证"部分，请认真审查其中的P0/P1级发现
- 对于算法完整性检查：如果代码中只包含少于 3 种算法，且题目需要优化求解，应标记为 P1 警告并判 FAIL（国赛要求至少 3 种算法对比）
- 对于约束处理检查：如果迭代算法返回0.00，应标记为P0错误并判FAIL

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

重要判定标准：
- 这是W1证据大纲门禁，不是论文终检。大纲只需明确主张-证据映射关系，不必包含完整正文
- 只要大纲覆盖了所有子问题、每个主张都有对应的证据来源（建模报告/代码结果/图表）、摘要关键数值与结果表一致，就应返回PASS
- 不要因为大纲缺少长篇正文、精细排版、或参考文献列表而判FAIL，这些属于W2阶段
- 如果大纲存在实质性的证据缺失（如某个子问题的结论完全没有证据支持），才返回FAIL

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
4. 图表引用完整性（每张图/表是否在正文中被引用，如图1、表2等）
5. 文献可追溯性
6. 章节结构完整性（是否包含：摘要、问题重述、假设与符号、模型建立、模型求解、结果分析、模型评价、结论、参考文献）
7. 图表编号是否连续（图1、图2、图3...，不能跳号）

重要判定标准：
- 这是W2论文终检，重点关注论文内容的正确性和一致性，而非排版美观度
- 只要论文覆盖了所有子问题、主张与证据一致、关键数值与单位正确、图表引用完整、文献可追溯，就应返回PASS
- 如果缺少关键章节（模型建立、模型求解、结果分析），应判FAIL
- 如果图表完全未被正文引用（如"检测到 0 个图表引用"），应判FAIL
- 题目给出的物理定律和数学方法无需引用外部文献，仅当论文引用了超出题目范围的外部研究时才需要文献支持
- 不要因为格式细节（如标题层级、字体大小、页边距）或排版问题而判FAIL

请返回标准回执。"""
        return self.invoke(messages, user_input=user_msg)