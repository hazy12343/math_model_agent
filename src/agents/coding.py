from pathlib import Path
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool, tool
from src.config import AppConfig
from src.agents.base import BaseAgent
from src.tools.file_reader import FileReaderTool


class CodingAgent(BaseAgent):
    def __init__(self, config: AppConfig):
        super().__init__(config, "编程手")
        self.file_reader = FileReaderTool(config.skill_root)

    def load_system_prompt(self) -> str:
        parts = []

        skill_content = self._load_role_skill("references/roles/编程手/SKILL.md")
        parts.append(f"# 你的角色：编程手\n\n{skill_content}")

        workflow = self._load_reference("references/roles/编程手/references/工作流程.md")
        if workflow:
            parts.append(f"\n\n# 工作流程\n\n{workflow}")

        vis_spec = self._load_reference("references/roles/编程手/references/可视化规范.md")
        if vis_spec:
            parts.append(f"\n\n# 可视化规范\n\n{vis_spec}")

        common_patterns = self._load_reference("references/roles/编程手/references/常见模式.md")
        if common_patterns:
            parts.append(f"\n\n# 常见模式\n\n{common_patterns}")

        parts.append(f"""
\n\n# 当前任务配置
- 竞赛类型：{self.config.competition}
- 语言：{self.config.language}
- 项目根目录：{{project_root}}

# 核心任务
1. 读取题目分析报告和术语表格
2. 选择 Python 或 MATLAB 实现
3. 数据读取、预处理和核心求解
4. 生成三类图（原始数据图、过程图、结果图），每类至少3张，合计至少9张
5. 生成结果表格和复现清单

# 重要规则
- 所有结论必须来自真实代码输出，禁止编造
- 先用小实例跑通，再全量计算
- 图表必须使用出版级样式
- 代码必须可复现

# 数值验证要求（关键）
- 代码中必须包含自检逻辑：验证关键约束是否满足、检查数值范围是否合理
- 输出中必须包含量纲检查结果（如：print(f"距离: {{distance:.2f}} m")）
- 敏感性分析必须包含参数范围选择依据
- 敏感性分析结果必须输出 CSV 文件（如 sensitivity.csv）
- 如果结果中出现 NaN/Inf/负值（对不允许负值的物理量），必须显式处理或报错

# 多算法对比要求（国赛关键）
- 对每个子问题的求解，必须实现至少 2 种不同求解算法（如：网格搜索 vs 遗传算法、梯度下降 vs 粒子群）
- 在代码中输出对比表格：
  | 算法 | 结果精度 | 计算耗时 | 稳定性 |
  |------|----------|----------|--------|
  | 算法A | ... | ... | ... |
  | 算法B | ... | ... | ... |
- 如果两种算法结果一致，说明解可靠；如果不一致，分析原因

# 收敛性分析要求（国赛关键）
- 对于迭代类算法（如坐标下降、梯度下降、遗传算法），必须输出收敛曲线
- 代码中自动判断是否收敛：连续 N 代改进量 < ε
- 输出收敛状态：已收敛 / 未收敛（达到最大迭代次数）

# 蒙特卡洛验证要求（国赛关键）
- 对最优解进行蒙特卡洛模拟验证：
  - 对关键参数（如题目中的测量值、估计值）添加 ±5% 的随机扰动
  - 运行至少 100 次蒙特卡洛模拟
  - 统计结果分布（均值、标准差、95% 置信区间）
  - 输出蒙特卡洛验证结果表

# 中文显示要求（关键）
- **所有图表的标题、轴标签、图例、注释必须使用中文**
- **代码开头必须配置中文字体**：
```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'
```
- 中文标题示例：`ax.set_title('M1 与 FY1 三维轨迹')` 而非 `ax.set_title('3D Trajectory')`
- 中文轴标签示例：`ax.set_xlabel('X 坐标 (m)')`、`ax.set_ylabel('Y 坐标 (m)')`
""")
        return "\n".join(parts)

    def get_tools(self) -> List[BaseTool]:
        return [
            self.file_reader.read_pdf_tool,
            self.file_reader.read_excel_tool,
            self.file_reader.read_text_tool,
            self._check_env_tool,
        ]

    @property
    def _check_env_tool(self):
        @tool
        def check_environment(features: str) -> str:
            """检查 Python 环境依赖。features 用逗号分隔，如 'data,visualization,optimization'"""
            import subprocess
            import sys
            script = Path(self.config.skill_root) / "references/roles/编程手/scripts/check_env.py"
            if not script.exists():
                return "[环境检查脚本不存在]"
            try:
                result = subprocess.run(
                    [sys.executable, str(script), "--features"] + features.split(","),
                    capture_output=True, text=True, timeout=30
                )
                return result.stdout + "\n" + result.stderr
            except Exception as e:
                return f"[环境检查失败: {e}]"
        return check_environment

    def implement_minimal(
        self,
        modeling_report: str,
        terminology_table: str,
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)

        user_msg = f"""请根据以下建模分析，实现最小可运行代码（P1阶段）：

## 题目分析报告
{modeling_report[:5000]}

## 术语表格
{terminology_table[:3000]}

请完成：
1. 数据读取和预处理代码（如有附件数据）
2. 核心求解链的最小实现
3. 用真实输入或结构等价小实例跑通
4. 输出关键中间结果

## 重要规则
- 代码必须自包含，所有代码放在一个文件里
- 不要跨文件 import 本项目其他模块
- 参数使用硬编码常量或在代码中定义，不要依赖外部 JSON/CSV 文件
- 如果题目没有附件数据，使用建模报告中的公式和参数构造示例数据
- **使用英文或拼音命名文件（如 'result.csv'、'sensitivity.csv'），避免中文文件名**
- **所有字符串字面量必须写在同一行内，禁止跨行字符串**
- **确保代码是语法正确的 Python，可直接运行**
- **代码块中只输出纯 Python 代码，禁止在代码块内混入 shell 命令（如 python xxx.py、pip install 等），shell 命令放在代码块外**
- **⚠️ 输出长度限制：代码总行数控制在 400 行以内，确保代码完整输出不被截断。如果代码超过 400 行，请精简注释和冗余代码**
- 使用 ```python 代码块包裹完整代码

只输出代码和运行说明，不要生成完整图表。"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt)

    def implement_full(
        self,
        modeling_report: str,
        terminology_table: str,
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)

        user_msg = f"""P1已通过，现在进行全量实现（P2阶段）：

## 题目分析报告
{modeling_report[:5000]}

## 术语表格
{terminology_table[:3000]}

请完成：
1. 全量计算和参数扫描（敏感性分析）
2. 生成三类图（原始数据图、过程图、结果图），每类至少1张，合计至少3张
3. 每个子问题在三类图中各至少1张
4. 生成结果表格
5. 生成复现清单

## 重要规则

### 中文显示（关键）
- **所有图表的标题、轴标签、图例、注释必须使用中文**
- **代码开头必须配置中文字体**：
```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'
```

### 代码必须自包含
- 所有代码放在一个文件里，不要跨文件 import 本项目其他模块
- 参数使用硬编码常量或在代码中定义，不要依赖外部 JSON/CSV 文件
- 如果题目没有附件数据，使用建模报告中的公式和参数构造示例数据

### 代码质量要求
- **使用英文或拼音命名文件和目录（如 'figures/'、'results/'、'result.csv'），禁止中文文件名**
- **所有字符串字面量必须写在同一行内，禁止跨行字符串**
- **确保代码是语法正确的 Python，可直接运行**
- **代码块中只输出纯 Python 代码，禁止在代码块内混入 shell 命令（如 python xxx.py、pip install 等），shell 命令放在代码块外**

### ⚠️ 输出长度限制（关键）
- **代码总行数控制在 600 行以内**，确保代码完整输出不被截断
- 如果代码超过 600 行，请精简注释、合并重复逻辑、减少图表数量
- **优先保证代码完整性和可执行性，而非图表数量**
- 图表至少生成 3 张（轨迹图、过程图、结果图各1张），不要为了凑数而增加代码量

### 精细搜索策略（国赛关键）
- 对网格搜索/参数扫描类任务，必须采用"先粗后精"两阶段搜索：
  1. **粗搜索**：大范围、大步长，确定最优解的大致区域
  2. **精细搜索**：在粗搜索最优解附近，将步长缩小 5~10 倍，进行局部加密
  3. 输出精细搜索前后的对比，展示优化幅度
  4. 例如：方向角步长从 45° → 5°，速度步长从 20m/s → 2m/s

### 约束处理方法（国赛关键）
- 对于迭代类优化算法（遗传算法、粒子群等），必须正确处理约束：
  1. **可行解初始化 + 可行解变异**（最优方案）：初始化时只生成满足约束的个体，变异操作确保子代仍然满足约束
  2. **惩罚函数法**（次优方案）：对违反约束的个体施加惩罚项，惩罚系数随进化代数动态增加
  3. **修复法**（备用方案）：对违反约束的个体进行修复，映射到最近可行解
- **禁止**：直接返回 0 作为不可行解的适应度值！这会导致算法完全失效。

### 输出格式
- 使用 ```python 代码块包裹完整代码
- 代码块后附上运行说明
- 图表保存到 figures/ 目录
- 结果保存到 results/ 目录

请输出完整代码和所有结果。"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt)

    def fix_code(
        self,
        feedback: str,
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)
        user_msg = f"""质检反馈了以下问题，请修正代码：

## 反馈
{feedback}

请修正代码并重新输出。

⚠️ 代码总行数控制在 600 行以内，确保完整输出不被截断。优先保证代码完整性。
⚠️ 所有图表标题、轴标签、图例、注释必须使用中文，代码开头必须配置中文字体。"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt)