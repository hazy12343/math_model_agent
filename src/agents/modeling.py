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

        # 加载前置合同（帮助结构化思考）
        pre_contract = self._load_reference("references/roles/建模手/references/前置合同.md")
        if pre_contract:
            parts.append(f"\n\n# 模型合同（必须在报告中体现）\n\n{pre_contract}")

        # 加载质检清单（输出前自检）
        quality_checklist = self._load_reference("references/roles/建模手/references/质检清单.md")
        if quality_checklist:
            parts.append(f"\n\n# ⚠️ 输出前自检清单（必须逐项确认！）\n\n{quality_checklist}")

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

# 问题类型识别（关键！不同类型的题目需要不同的建模策略）
在开始建模之前，必须先识别题目属于以下哪种（或哪几种）问题类型：

## A. 优化类问题
**特征**：给定资源和约束，求最大化/最小化某个目标函数
**建模策略**：
- 明确决策变量、目标函数、约束条件
- 分析可行域的形状和大小
- 如果问题是 NP-hard，考虑启发式/近似算法
- 必须给出理论上界/下界
**常见子类型**：资源分配、路径规划、调度排产、参数标定

## B. 预测/预报类问题
**特征**：根据历史数据预测未来值
**建模策略**：
- 分析数据的时间特征（趋势、周期、突变点）
- 选择合适的预测模型（时序模型/回归模型/ML模型）
- 必须划分训练集和测试集，报告预测误差（MAE/RMSE/MAPE）
- 必须给出预测的不确定性（置信区间/预测区间）
**常见子类型**：时间序列预测、回归预测、分类预测

## C. 评价/决策类问题
**特征**：对多个方案/对象进行综合评价和排序
**建模策略**：
- 构建评价指标体系（层次化、可量化、不冗余）
- 确定权重（主观法/客观法/组合赋权）
- 选择评价方法（TOPSIS/层次分析/模糊综合评价/数据包络分析）
- 必须进行敏感性分析（权重变化对排序的影响）
**常见子类型**：综合评价、方案优选、风险评估

## D. 分类/识别类问题
**特征**：将对象划分到预定义类别
**建模策略**：
- 特征工程（提取/选择/构造判别特征）
- 处理类别不平衡（过采样/欠采样/代价敏感）
- 报告分类性能（准确率/精确率/召回率/F1/混淆矩阵）
- 必须进行交叉验证（K-fold）
**常见子类型**：模式识别、异常检测、图像分类

## E. 仿真/模拟类问题
**特征**：通过数值模拟研究系统行为
**建模策略**：
- 确定仿真精度要求（时间步长/空间分辨率）
- 验证仿真模型（与解析解/已知特例对比）
- 进行参数扫描和敏感性分析
- 必须进行收敛性验证（网格/时间步长收敛）
**常见子类型**：物理仿真、交通流仿真、传染病模型、蒙特卡洛模拟

## F. 机理/方程类问题
**特征**：基于物理/化学/生物定律建立微分方程/代数方程
**建模策略**：
- 从第一性原理出发推导方程
- 分析方程性质（线性/非线性、稳态/瞬态、刚性）
- 选择合适的数值方法（解析解、数值积分、有限差分/有限元）
- 必须验证守恒律和量纲一致性
**常见子类型**：热传导、流体力学、化学反应动力学、种群动力学

## G. 统计/数据分析类问题
**特征**：从数据中提取统计规律和关系
**建模策略**：
- 数据预处理（缺失值、异常值、标准化）
- 选择合适的统计方法（回归/方差分析/假设检验/相关分析）
- 报告效应量和置信区间，而非仅 p 值
- 必须进行残差分析和模型诊断
**常见子类型**：回归分析、方差分析、生存分析、因子分析

**在报告开头必须明确标注识别到的问题类型及其理由。**

# 多模型对比要求（国赛关键）
- 对每个子问题，必须提出至少 2 种建模思路
- **对于优化类问题（A 类），必须提出至少 3 种不同的求解算法**（如：网格搜索+差分进化+粒子群优化 / 遗传算法+模拟退火+局部搜索）
- 在报告中建立对比表格：
  | 对比项 | 模型A（主模型） | 模型B（对比模型） | 模型C（备选模型） |
  |--------|----------------|------------------|------------------|
  | 建模思路 | ... | ... | ... |
  | 适用条件 | ... | ... | ... |
  | 计算复杂度 | ... | ... | ... |
  | 预期精度 | ... | ... | ... |
  | 优势 | ... | ... | ... |
  | 劣势 | ... | ... | ... |
- 主模型用于最终的求解和优化，对比模型用于验证和讨论
- 如果多种模型结论一致，说明结果可靠；如果不一致，分析原因

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

# 搜索空间与精度分析要求（国赛关键 — 为编程手提供搜索依据）
- 在建模报告中必须包含"搜索空间分析"章节，明确给出：
  1. 各搜索参数的范围和物理含义（如：θ ∈ [0, 2π)、v ∈ [v_min, v_max] 等）
  2. 每个参数的有效精度要求（如：角度精度 = 目标特征尺度/搜索距离，或由题目隐含精度要求反推）
  3. 推荐的最小搜索点数（搜索范围/有效精度 × 安全系数3~5）
  4. 推荐的搜索策略（如：先粗搜索定位，再细搜索精化）
  5. 理论最大值的估计（如：资源数量 × 单次作用时间 = 理论上限）
  6. 理论上界必须给出具体数值，格式为"理论上界=XXX.XX（单位）"，并说明计算方法
- 这部分分析直接决定编程手能否生成精度足够的搜索代码
- 如果建模手不给出精度要求，编程手会使用默认的粗步长导致结果完全不可用

# ⚠️ 搜索规模硬性约束（关键！建模手必须遵守）
- 推荐的算法参数必须在可执行范围内，否则编程手会超时：
  - **差分进化(DE)**: popsize ≤ 10, maxiter ≤ 30
  - **粒子群优化(PSO)**: 粒子数 ≤ 20, 迭代次数 ≤ 50
  - **遗传算法(GA)**: 种群规模 ≤ 30, 迭代代数 ≤ 50
  - **模拟退火(SA)**: 最大迭代 ≤ 500（单次迭代成本低）
  - **网格搜索**: 总组合数 ≤ 5000
  - **时间步长**: Δt ≥ 0.2s（仿真类问题），优先推荐 0.5s
- 如果推荐的算法参数超过上述限制，必须说明理由并提供"精简版"参数作为备选
- 例如：推荐的"PSO 粒子数200×迭代300次"会被编程手拒绝，应改为"精简版：粒子数15×迭代30次"

# 理论上界计算要求（国赛关键 — 用于评估优化结果质量的P0级要求！）
- 对每个优化问题，必须在报告中明确给出理论上界值，**这是一个具体数值，不是范围或描述**
- 理论上界计算的系统方法：
  1. **识别所有并行资源**：列出所有可并行工作的资源（如多个无人机、多条产线、多个时间窗口）
  2. **计算每个资源的最大贡献**：对每个资源，计算其在理想条件下（无冲突、无等待）的最大贡献值
     - 资源最大贡献 = 资源效率参数 × 有效作用时间窗口
     - 示例：单架无人机单枚弹最大遮蔽时间 = CLOUD_DUR = 20s
     - 示例：5架无人机 × 3枚弹 × 20s/枚 = 300s（忽略地理约束的理论上界）
  3. **考虑物理/地理约束的上界**：在理想条件下进一步考虑不可逾越的物理限制
     - 示例：如果FY1初始位置离目标太远，即使以最大速度飞行也无法在T_MAX内到达，则FY1的上界=0
     - 示例：导弹飞行时间有限，每枚导弹最多只能被遮蔽 T_missile_max 秒
  4. **理论上界 = min(资源上界, 物理上界, 时间上界)**
  5. 如果存在资源冲突，还应给出"考虑冲突的理论上界"（通常更紧）
- 理论上界必须在报告中有清晰的计算步骤，格式如下：
  ```
  理论上界计算：
  - 步骤1：识别并行资源
    资源列表：FY1~FY5，每架最多3枚弹，每枚弹有效时间20s
    总资源容量：5 × 3 × 20 = 300s（忽略地理约束）
  - 步骤2：考虑物理约束
    FY1初始位置(17800,0,1800)，目标距离约17801m，V_MAX=140m/s
    FY1到达目标最短时间 = 17801/140 ≈ 127s > T_MAX=70s → FY1不可达目标
    同理检查FY2~FY5...
  - 步骤3：考虑导弹飞行时间约束
    M1到达假目标约67s，M2约64s，M3约60s
    每枚导弹可被遮蔽时间 ≤ 到达时间 ≈ 60~67s
    3枚导弹总可遮蔽时间 ≤ 67+64+60 = 191s
  - 步骤4：计算理论上界
    理论上界（考虑地理约束）= min(可达资源容量, 导弹总时间) = XXX.XX s
    考虑冲突的理论上界 = XXX.XX s（更紧的上界，计算方法：...）
  ```
- 这个值将用于后续评估优化结果的质量（实际值/理论上界 应 ≥ 15%）
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
        prompt = self.get_cached_prompt().replace("{project_root}", project_root)

        # 检测用户聚焦指令，覆盖系统提示词中的"为每个子问题"指令
        if "用户聚焦指令" in problem_description:
            prompt += "\n\n# ⚠️ 最高优先级指令\n用户在题目描述中明确要求只处理特定子问题。你必须严格遵守：只分析和建模用户指定的子问题，不要处理其他子问题！\n\n# ⚠️ 防重复指令\n禁止在报告中重复输出相同的段落！每段内容只写一次，不要大段重复！"

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
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt)

    def fix_model(
        self,
        feedback: str,
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.get_cached_prompt().replace("{project_root}", project_root)
        user_msg = f"""编程手反馈了以下问题，请修正模型设计：

## 反馈
{feedback}

请修正题目分析报告.md 和 术语表格.md，只输出修正后的完整内容。

⚠️ **禁止大段重复段落！** 每段内容只写一次，不要复制粘贴相同的内容！"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt, use_fix_model=True)