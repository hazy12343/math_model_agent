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

        figure_category_count = self.config.min_figure_count // 3
        parts.append(f"""
\n\n# 当前任务配置
- 竞赛类型：{self.config.competition}
- 语言：{self.config.language}
- 项目根目录：{{project_root}}

# 核心任务
1. 读取题目分析报告和术语表格
2. 选择 Python 或 MATLAB 实现
3. 数据读取、预处理和核心求解
4. 生成三类图（原始数据图、过程图、结果图），每类至少{figure_category_count}张，合计至少{self.config.min_figure_count}张
5. 生成结果表格和复现清单

# 重要规则
- 所有结论必须来自真实代码输出，禁止编造
- 先用小实例跑通，再全量计算
- 图表必须使用出版级样式
- 代码必须可复现
- **禁止大段重复注释！** 每段注释只写一次，不要复制粘贴相同的注释块！

# 可用的第三方库
运行时环境中已安装以下第三方库，你可以直接 import 使用：
- **科学计算**: numpy, scipy, sympy, pandas
- **优化**: scipy.optimize（含 differential_evolution、minimize 等）、PuLP（线性规划）
- **机器学习**: scikit-learn, xgboost, lightgbm
- **图论与网络**: networkx
- **统计分析**: statsmodels
- **凸优化**: cvxpy
- **深度学习**: torch (PyTorch)、tensorflow
- **可视化**: matplotlib, seaborn, plotly, folium, pyecharts, wordcloud
- **主题模型/NLP**: gensim
- **文件读写**: openpyxl, PyPDF2, python-docx, requests
- 以上库均已安装，无需 pip install。如果你需要其他库，请在代码注释中说明。

# 防御性编程规范（关键！防止常见运行时错误）
- **NumPy 广播安全**：涉及不同形状数组运算时，必须显式使用 np.outer() 或 reshape() 对齐维度
  - 错误示例：`td**2 * np.array([0, 0, 1])`  # 形状 (N,) * (3,) → 可能广播失败
  - 正确示例：`np.outer(td**2, np.array([0, 0, 1]))`  # 形状 (N, 3)，明确意图
- **除零保护**：所有除法运算前检查分母是否为零，使用 np.where() 或 if 判断
- **类型一致性**：确保所有数值运算使用相同 dtype（如 float64），避免 int/float 混合运算溢出
- **边界检查**：数组索引前检查是否越界，切片操作前验证数组长度
- **空值处理**：所有从外部读取的数据（CSV、Excel）必须检查 NaN/None，使用 np.nan_to_num() 或 fillna()
- **异常捕获**：关键计算段（优化循环、文件I/O）必须用 try/except 包裹，输出友好错误信息
- **形状断言**：关键数组运算后添加 assert 验证形状，如 `assert result.shape == (N, 3)`

# 计算效率规范（⚠️ 硬性约束！违反将导致超时 → P2 门禁 FAIL）

## ⚠️ 代码生成前自检清单（必须逐项确认后才能输出代码！）
在输出代码之前，你必须自查以下每一项。如果任何一项不满足，不要输出代码，先优化设计：

- [ ] 时间步长 Δt ≥ 0.2s？____（填写实际值）
- [ ] 最大嵌套循环层数 ≤ 3？____（填写实际层数，跨函数调用链的嵌套也算！）
- [ ] 差分进化 popsize ≤ 10 且 maxiter ≤ 30？____
- [ ] 粒子群优化 粒子数 ≤ 20 且 迭代 ≤ 50？____
- [ ] 单次 evaluate 调用的总迭代次数 < 5000？____（填写估算值）
- [ ] 总评估成本 < 100,000？____（= 优化算法调用次数 × 单次 evaluate 迭代次数）
- [ ] 至少 3 种不同算法对比？____（填写算法名称，如：网格搜索+差分进化+粒子群优化）
- [ ] 至少 1 种全局优化算法？____（如差分进化/粒子群优化/遗传算法）
- [ ] 至少 1 种局部搜索算法？____（如贪心/爬山/坐标下降）
- [ ] 多目标全覆盖？____（每个目标都有独立优化结果，非零）
- [ ] 敏感性分析输出 CSV 文件？____（确认代码中写入了 sensitivity.csv）
- [ ] 资源分配覆盖所有可用节点？____（填写每个节点的任务数，不得有闲置节点）
- [ ] 敏感性分析使用真实基线值（非硬编码数值）？____（检查 change_pct 公式中无硬编码数字）
- [ ] 收敛曲线来自真实迭代记录（非合成直线）？____（检查 history 是否在循环中 append）
- [ ] 蒙特卡洛验证覆盖所有参数（非仅部分参数）？____（检查 perturbation 是否作用于全部参数）
- [ ] 零值保护：所有图表/CSV/蒙特卡洛/收敛曲线在结果为零时仍生成？____（检查是否有 `if base > 0:` 跳过块）
- [ ] 适应度函数符号正确？____（最大化问题用 `-raw + penalty` 而非 `-(raw - penalty)`）
- [ ] 图表同时保存 PNG 和 SVG 格式？____（检查 plt.savefig 是否同时调用两次）
- [ ] 蒙特卡洛扰动方式正确？____（优先使用加法扰动，避免乘法扰动在参数≈0时失效导致 std=0）
- [ ] 算法对比格式正确？____（包含"算法对比:"标题，三列分别为算法名/结果/耗时或收敛代数）

**如果以上任何一项的答案是"否"，你必须重新设计算法，不能输出当前代码！**

### 禁止的代码模式（以下模式一旦出现，代码必定超时）
- **禁止模式 1 — 多层嵌套网格搜索**：
  ```python
  # 禁止！以下模式必定超时（6层嵌套 = 50×30×30×30×20×20 = 5.4亿次迭代）
  for t1 in np.linspace(...):
      for dt_b1 in np.linspace(...):
          for t2 in np.linspace(...):
              for dt_b2 in np.linspace(...):
                  for t3 in np.linspace(...):
                      for dt_b3 in np.linspace(...):
  ```
  **正确做法**：将搜索参数交给优化器（DE/PSO），不要在目标函数内部做网格搜索！
  - 优化器负责搜索参数空间（theta, v, t1, dt1, t2, dt2, t3, dt3...）
  - 目标函数只负责计算给定参数的目标值
  - 优化器会自动高效搜索，无需手动嵌套循环

- **禁止模式 2 — 在目标函数内做全量枚举**：
  ```python
  def evaluate(params):
      for i in range(所有资源节点):
          for k in range(所有目标任务):
              for p1 in range(参数1所有可能值):
                  for p2 in range(参数2所有可能值):
                      ...
  ```
  **正确做法**：参数向量应包含所有待优化的变量，目标函数直接使用 params 计算，不做内部枚举

- **时间步长硬性下限**：仿真类任务中，时间步长 Δt 必须 ≥ 0.2s，优先使用 0.5s
  - 违反示例：DT = 0.05  # 在 20s 时间窗口内产生 400 步 → 每次 evaluate 调用 400 次迭代 → 必定超时
  - 正确示例：DT = 0.5   # 在 20s 时间窗口内仅 40 步 → 安全
  - 计算：时间步数 = 时间窗口长度 / Δt。如果单次 evaluate 的总迭代次数 > 5000，必须增大 Δt
- **禁止超过 3 层有效嵌套循环**：参数扫描类任务必须使用优化器（DE/PSO/梯度下降）代替手动网格搜索
  - 注意：即使每层循环在不同的函数中，如果它们形成调用链（如 `for node in ...: evaluate()` 内部 `for task in ...: simulate()` 内部 `while t < ...`），也算作有效嵌套
  - 错误示例：`for t1 in ...: for dt1 in ...: for t2 in ...: for dt2 in ...:`  # 4+ 层嵌套 → 必定超时
  - 正确示例：将 (t1, dt1, t2, dt2, ...) 全部放入参数向量，交给 `differential_evolution` 或 PSO 搜索
- **差分进化参数硬性上限**：`popsize` ≤ 10，`maxiter` ≤ 30，`tol` ≥ 0.01（用于提前终止）
  - 违反示例：`differential_evolution(..., popsize=15, maxiter=50, tol=0.01)`  # 2 项违规
  - 正确示例：`differential_evolution(..., popsize=8, maxiter=20, tol=0.01)`
  - 如果提供了自定义初始种群（init 参数），popsize 在该参数中设置，但 maxiter 和 tol 仍受上述限制
- **⚠️ 差分进化静默失败风险**：DE 在约束过强或搜索空间不适配时可能返回 0 或边界值，表现为"找到了结果但实际是无效解"
  - 必须验证 DE 结果：如果 DE 返回 0 或接近 0 而另一算法返回正常值，说明 DE 未找到有效解，必须在输出中明确标注"差分进化未产生有效解"
  - 必须对 DE 结果做物理可行性检查：如 coverage > 0、penalty == 0 才认为有效
  - 差分进化不应作为唯一算法，必须与至少一种其他算法（如随机搜索、网格搜索）配合使用
- **粒子群优化（PSO）参数硬性上限**：粒子数 ≤ 20，迭代次数 ≤ 50
  - 违反示例：`n_particles = 50; for it in range(100):`  # 5000 次 evaluate 调用
  - 正确示例：`n_particles = 15; for it in range(30):`  # 450 次 evaluate 调用
- **粗搜索点数限制**：粗搜索总组合数不超过 5000，超过时必须增大步长
- **单次 evaluate 调用成本估算**：在编写 evaluate/objective 函数时，必须估算单次调用的迭代次数
  - 估算公式：总迭代次数 = Π(每层循环的迭代次数)

- **禁止的验证模式**（以下模式给出"假"验证结果，P2 门禁会直接 FAIL）：
  - **禁止模式 V1 — 硬编码基线的敏感性分析**：
    ```python
    # 错误：硬编码基线值 10.0
    change_pct = (shield - 10.0) / 10.0 * 100
    ```
    **正确做法**：基线值必须来自实际计算（如 base_shield = evaluate(baseline_params)），
    `change_pct = (shield - base_shield) / base_shield * 100`
  
  - **禁止模式 V2 — 合成的收敛曲线**：
    ```python
    # 错误：合成直线，不是真实迭代记录
    history = [best_shield * (i/10) for i in range(1, 11)]
    ```
    **正确做法**：在优化循环中每次迭代记录真实值
    `history.append(best_shield)` 后 `plt.plot(history)`
  
  - **禁止模式 V3 — 只测试部分参数的蒙特卡洛**：
    ```python
    # 错误：只用 best_params[:4] 测试，忽略了其余参数
    monte_carlo_validation(best_params[:4])
    ```
    **正确做法**：对所有参数进行扰动
    `monte_carlo_validation(best_params)`  # 所有参数

- **总评估成本** = 优化算法调用次数 × 单次 evaluate 迭代次数
  - 如果总评估成本 > 100,000，必须优化 evaluate 函数或增大时间步长（或交给优化器而非手动枚举）
- **提前终止**：所有迭代算法必须实现提前终止条件（如连续 N 代无改进则退出）
- **进度输出**：在关键循环处添加 `print(f"进度: {{i}}/{{total}}")` 以便诊断性能瓶颈

# 数值验证要求（关键）
- 代码中必须包含自检逻辑：验证关键约束是否满足、检查数值范围是否合理
- 输出中必须包含量纲检查结果（如：print(f"距离: {{distance:.2f}} m")）
- 敏感性分析必须包含参数范围选择依据
- **敏感性分析结果必须输出 CSV 文件**（如 `results/sensitivity.csv`），列名格式：`param,value,cover_time,change_pct`
  - 必须在代码中写入：`with open('results/sensitivity.csv', 'w') as f: csv_writer = ...`
  - 仅打印到控制台不算满足要求！必须写入文件！
  - 如果代码执行目录不是项目根目录，先 `os.makedirs('results', exist_ok=True)`
- **⚠️ 控制台输出要求（必须！）**：敏感性分析完成后，必须 `print` 到控制台：
  ```
  print("敏感性分析: 完成（详见 results/sensitivity.csv）")
  # 同时输出关键发现
  print(f"敏感性分析: 最敏感参数={{most_sensitive}}, 最大变化率={{max_change:.1f}}%")
  ```
- **⚠️ 零值保护与输出完整性**：即使最优结果为 0（或未找到有效解），也必须：
  - **生成所有图表**（即使显示全零值，图表本身是代码工作量的证明）
  - **写入敏感性 CSV**（即使所有 change_pct 为 0，也必须有文件头和行数据）
  - **执行蒙特卡洛验证**（即使扰动后结果仍为 0，也必须有统计输出）
  - **生成收敛曲线**（即使所有迭代值相同，也必须记录并绘制）
  - 禁止用 `if base_shield > 0:` 跳过敏感性/蒙特卡洛/收敛曲线——这些是代码能力的证明，不能因"结果为零"而跳过
- 如果结果中出现 NaN/Inf/负值（对不允许负值的物理量），必须显式处理或报错

# 多算法对比要求（国赛关键 — 必须输出至少 3 种独立算法！）
- 对每个子问题的求解，必须实现**至少 3 种互不依赖的独立求解算法**，并在代码中输出对比结果
- 3种算法中必须包含至少 1 种全局优化算法（差分进化/粒子群优化/遗传算法/模拟退火）和至少 1 种局部搜索算法（贪心/爬山/坐标下降）
- **注意**：粗搜索 + DE 精化 ≠ 两种算法！它们是一个流水线，不是独立算法对比。
  - 错误：grid_search() → DE() → 输出 DE 结果。这**不是**算法对比
  - 正确：运行 grid_search() 得到结果A；运行 PSO() 得到结果B；运行 DE() 得到结果C；然后对比 A、B 和 C
- **符合要求的算法组合示例**：
  - 网格搜索 vs 差分进化 vs 粒子群优化（三个独立优化器）
  - 贪心法 vs 遗传算法 vs 模拟退火（三个独立策略）
  - 梯度下降 vs 差分进化 vs 随机搜索（三个独立优化器）
- 在代码中输出对比表格，**必须使用以下精确格式**（方便自动提取），**必须同时 print 到控制台和写入文件**：
  ```
  print("算法对比:")
  print(f"  算法                     结果                   耗时(s)     ")
  print(f"  网格搜索                 xxx.xx                 x.xx      ")
  print(f"  差分进化                 xxx.xx                 x.xx      ")
  print(f"  粒子群优化               xxx.xx                 x.xx      ")
  ```
  - 第三列可以是"耗时(s)"（推荐，记录实际 wall-clock 时间）或"收敛代数"（记录迭代收敛的代数）
  - 如果使用 `time.time()` 记录耗时，格式如 `耗时=12.35 s`
  - 如果使用收敛代数，格式如 `收敛代数=10`
- 如果多种算法结果一致（差异 < 5%），说明解可靠；如果不一致，分析原因

# 国赛级结果质量要求（关键！不满足将导致 P2 自动 FAIL）
- **结果质量阈值**：优化结果必须 ≥ 理论最大值的 10%（国赛基本要求），争取 ≥ 20%（冲击国一）
  - 如果首次运行结果 < 10%，必须自动触发以下优化：
    1. 缩小搜索空间（基于物理约束去掉不可行区域）
    2. 增加搜索精度（步长缩小 5-10 倍）
    3. 尝试不同的算法组合
  - 如果 3 种算法结果都 < 10%，说明模型设计或搜索策略有根本性问题，需要重新分析
- **多目标全覆盖**：如果题目涉及多个目标（如多枚导弹、多个节点、多个任务），必须确保每个目标都有非零结果
  - 不得出现"仅部分目标有结果，其余为零"的情况
  - 每个目标的策略参数应独立优化，而非所有目标共享同一组参数
  - 输出中必须包含每个目标的独立结果统计
- **蒙特卡洛鲁棒性**：蒙特卡洛验证的均值应 ≥ 最优值的 50%，失败率（零值占比）应 ≤ 30%
  - 如果鲁棒性不满足要求，说明策略对参数扰动过于敏感，需要：
    1. 使用更保守的策略参数（如增加安全余量）
    2. 在目标函数中引入鲁棒性惩罚项
    3. 使用随机规划或鲁棒优化方法

# 多目标全覆盖策略（国赛关键 — 防止部分目标结果为零）
- 如果题目涉及多个独立目标（如 M1, M2, M3），必须为每个目标独立执行优化：
  ```python
  for target_id in targets:
      best_params[target_id] = optimize_for_target(target_id)
  ```
- 不得用一个目标的参数直接套用到其他目标（每个目标的路径/约束可能不同）
- 不得只优化一个目标然后将结果复制到其他目标
- 输出中必须包含每个目标的独立结果和总结果汇总

# 遗传算法/进化算法初始化策略（关键！防止零值结果）
- **禁止纯随机初始化！** 在高维/窄可行域问题中，纯随机初始化几乎不可能命中有效解。必须结合问题结构进行智能初始化：
  - 分析问题的几何/物理约束，预计算可行域的大致范围
  - 基于预计算结果生成初始种群（至少 50% 个体来自智能初始化）
  - 示例（通用）：如果问题是路径规划，先计算目标位置的几何关系，再反推初始参数；如果问题是参数拟合，先用最小二乘得到粗略估计，再在其附近初始化
- **初始化时至少 30% 的个体使用随机扰动**（而非全部智能初始化），以增加搜索多样性
- **时间/顺序相关参数应基于关键时间节点反向推算**，而非随机采样
- **资源分配应覆盖所有目标**：确保每个目标都分配到至少最低限度的资源

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
- **⚠️ 扰动方式注意事项**：
  - 优先使用**加法扰动**：`param + U(-0.05*scale, 0.05*scale)`，其中 scale 是参数的量级
  - 乘法扰动 `param * (1.0 ± 0.05)` 在参数接近零时失效（扰动幅度趋近于零），导致 std≈0
  - 对于不同量级的参数，应使用各自的 scale 而非统一百分比
  - 扰动后必须裁剪到参数边界内：`np.clip(perturbed, LB, UB)`
- **⚠️ 控制台输出要求（必须！）**：蒙特卡洛验证完成后，必须 `print` 到控制台：
  ```
  print(f"蒙特卡洛验证: 完成（N=100，均值={{mean:.2f}}，标准差={{std:.2f}}，95%CI=[{{ci_lo:.2f}}, {{ci_hi:.2f}}]）")
  ```

# 自适应搜索精度策略（国赛关键 — 最重要！）
在生成搜索/优化代码前，必须先进行搜索精度分析：
1. 计算搜索空间的特征尺度（如：参数范围 [a, b]，变量个数 d）
2. 计算有效精度要求（如：目标函数对参数变化的敏感度，或题目隐含的精度要求）
3. 计算所需最小搜索点数 = 搜索范围 / 有效精度 × 安全系数(≥3~5)
4. 如果单层搜索点数过大（>10000），必须采用分层策略：
   - 第一层：粗搜索（500~1000 个点），找到最优解的大致区域
   - 第二层：在最优解附近 ±10% 范围内细搜索（2000~5000 个点），精化结果
   - 第三层：对最优解进行蒙特卡洛验证
5. 每层搜索必须输出当前最优结果，便于分析搜索收敛性
6. **关键原则：搜索分辨率必须由"有效精度"决定，不能随意设置！**

# 多轮自适应优化（国赛关键）
1. 第一轮：粗粒度搜索，快速定位最优解区域
2. 第二轮：在最优解附近进行细粒度搜索，精化结果
3. 第三轮：对精化后的最优解进行蒙特卡洛验证
4. 每轮必须输出当前最优结果，展示搜索收敛过程
5. 如果最优结果与理论最大值差距过大（< 理论最大值的 10%），必须自动触发更高精度搜索

# 遗传算法/进化算法约束处理（国赛关键 — 必须遵守！）
对于带约束的优化问题，**不可行解绝对不能直接返回固定值（0/1e6/1e9 等）**！
固定值惩罚（无论是 0.0 还是 1e6）都会在适应度景观中制造"悬崖"，
导致优化器无法区分"略微违反约束"和"严重违反约束"，最终无法找到可行解。

正确做法（必须使用**比例惩罚函数**法）：
```python
def fitness(params):
    raw_value = compute_objective(params)  # 原始目标值
    penalty = 0.0
    # 示例：速度约束处理
    if v < V_MIN:
        penalty += 1000.0 * (V_MIN - v)  # 大惩罚系数，比例于违反程度
    if v > V_MAX:
        penalty += 1000.0 * (v - V_MAX)
    # 示例：时间约束处理
    if t_start < 0:
        penalty += 10000.0 * abs(t_start)  # 时间约束惩罚更重
    if t_end < t_start:
        penalty += 10000.0 * (t_start - t_end)
    # 返回带惩罚的适应度（最大化问题）或原始值减惩罚（最小化问题）
    return raw_value - penalty
```

错误做法（禁止 — 以下所有写法都会导致优化器完全失效）：
```python
# 禁止写法 1：固定值返回 — 最严重的错误！
if constraint_violated:
    return 0.0   # 禁止！固定值，优化器无法区分违反程度

# 禁止写法 2：大固定值 — 和 return 0.0 一样的问题！
if constraint_violated:
    return 1e6   # 禁止！仍然是固定值，只是换了个大数

# 禁止写法 3：提前返回 — 跳过约束检测
if not valid:
    return 1e9   # 禁止！任何形式的固定值返回都是错误的

# 禁止写法 4：裁剪到 0 — 和 return 0.0 一样的问题！
T = max(0.0, T)  # 禁止！不可行解被裁剪到 0，优化器无法区分"略微偏离"和"完全不可行"
# 正确做法：在目标函数中保持原始值，通过惩罚项引导优化器
```

**为什么固定值惩罚会失败**：假设参数空间 99% 的区域违反约束，所有不可行解都返回 1e6。
优化器看到的景观是：99% 的区域都是 1e6 的"高原"，只有 1% 的区域有梯度。
优化器无法在高原上找到方向，最终随机收敛到某个全零解。

**比例惩罚的关键**：惩罚项必须 `∝` 违反程度。违反越严重，惩罚越大，
这样才能在不可行区域中形成指向可行域的梯度。

# 资源充分利用（国赛关键）
- 必须使用全部可用资源（如题目给定的所有设备、所有时间窗口、所有容量上限）
- 如果建模报告指定了资源上限，必须用满或给出未用满的合理理由
- **资源分配必须覆盖所有资源节点**：如果问题中有 N 个可用资源节点（如车辆、基站、设备），
  必须确保每个节点都分配到至少一项任务，不得将所有任务集中到单一节点。
  - 错误示例：5 个可用节点，3 个任务全部分配给节点 A → 其他 4 个节点闲置
  - 正确示例：5 个可用节点，3 个任务分配给节点 A/B/C → 充分利用资源
  - 实现方法：在分配算法中引入多样性约束（如每个节点至少分配 floor(N_tasks/N_nodes) 个任务）
- 多资源协同策略：同一平台的多个资源应形成连续或互补的使用窗口
- 输出中必须包含资源利用率统计

# 中文显示要求（关键）
- **所有图表的标题、轴标签、图例、注释必须使用中文**
- **代码开头必须配置中文字体**：
```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'
```
- 中文标题示例：`ax.set_title('模型A 与 模型B 对比')` 而非 `ax.set_title('3D Trajectory')`
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
                    [sys.executable, str(script), "--features", features],
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

        # 检测用户聚焦指令
        is_focused = "用户聚焦指令" in modeling_report
        if is_focused:
            prompt += "\n\n# ⚠️ 最高优先级指令\n用户明确要求只实现特定子问题。你必须严格遵守：只生成该子问题的代码，不要实现其他子问题的代码！\n\n# ⚠️ 防重复指令\n禁止在代码中重复输出相同的注释块！每段注释只写一次，不要大段重复！如果发现自己陷入重复，立即切换到下一个代码段。"

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

## P1 阶段核心原则：快速验证可行性，不追求最优解
- P1 的目标是验证代码能跑通，不是求解最优值
- 即使结果质量不高（如只有理论最优的 10%），只要代码能正确执行、结果合理（非零/非NaN），P1 就通过
- 优化结果质量是 P2 阶段的任务，不要在 P1 阶段追求完美

## 快速模式参数（P1 专用，与 P2 不同）
- **迭代算法参数**：种群大小 ≤ 50，迭代次数 ≤ 40，收敛停滞阈值 ≤ 10
- **时间步长**：仿真步长 ≥ 1.0s（P1 用大步长快速验证）
- **搜索点数**：粗搜索组合数 ≤ 1000，不需要精细搜索
- **优先使用贪心/启发式算法**：如果问题复杂，优先用贪心算法快速得到一个可行解，复杂优化留到 P2
- **代码总行数 ≤ 400 行**（P1 不需要完整的多算法对比和敏感性分析）

## 重要规则
- 代码可以分成多个文件，用 `# file: 文件名.py` 标记每个文件的起始，第一个文件为主入口文件
- 不要跨文件 import 本项目其他模块（所有 import 必须来自标准库或已安装的第三方库）
- 参数使用硬编码常量或在代码中定义，不要依赖外部 JSON/CSV 文件
- 如果题目没有附件数据，使用建模报告中的公式和参数构造示例数据
- **使用英文或拼音命名文件（如 'result.csv'、'sensitivity.csv'），避免中文文件名**
- **所有字符串字面量必须写在同一行内，禁止跨行字符串**
- **确保代码是语法正确的 Python，可直接运行**
- **⚠️ 常见 Python 语法陷阱（必须避免）：**
  - **生成器表达式中的 if/else**：`(x for x in seq if cond else y)` 是**错误**的！`if` 在生成器中是过滤器，不能带 `else`。正确写法：`(x if cond else y for x in seq)`（三元表达式放在 `for` 前面）
  - **f-string 中的复杂表达式**：f-string 中的 `:` 会被解析为格式说明符，不要在 f-string 内写带 `if/else` 的复杂表达式，先用变量存结果再放入 f-string
  - **缩进混用**：只使用 4 空格缩进，禁止混用 tab 和空格
- **代码块中只输出纯 Python 代码，禁止在代码块内混入 shell 命令（如 python xxx.py、pip install 等），shell 命令放在代码块外**
- **⚠️ 输出长度限制：代码总行数控制在 400 行以内。P1 阶段只需要验证可行性，不需要完整的多算法对比。**
- **⚠️ P1 阶段以文本输出为主，不强制要求生成图表。中文字体配置由系统自动注入，无需手动写 matplotlib 字体配置。如果需要画调试图，可以 import matplotlib 但不要写 rcParams 配置。**
- 使用 ```python 代码块包裹完整代码
- **禁止大段重复注释！每个注释块只写一次，不要重复粘贴相同的注释内容！**

## ⚠️ 适应度函数符号设计（P0 级 — 违者必错！）
- 差分进化/遗传算法默认**最小化**目标函数。如果目标是最大化（如遮蔽时间/覆盖率/收益），必须对目标值取反
- **正确写法**:
  ```python
  def fitness(params):
      raw = evaluate(params)  # 原始目标值（越大越好）
      penalty = check_constraints(params)  # 惩罚值（≥0，比例于违反程度）
      return -(raw) + penalty
  ```
- **错误写法**:
  ```python
  return -(raw - penalty)  # ❌ 当 penalty > raw 时，优化器会寻找约束违反最严重的解
  ```

只输出代码和运行说明，不需要生成完整图表。"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt)

    def implement_full(
        self,
        modeling_report: str,
        terminology_table: str,
        messages: List[BaseMessage],
        project_root: str,
        p1_feedback: str = "",
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)

        # 检测用户聚焦指令
        is_focused = "用户聚焦指令" in modeling_report
        if is_focused:
            prompt += "\n\n# ⚠️ 最高优先级指令\n用户明确要求只实现特定子问题。你必须严格遵守：只生成该子问题的代码，不要实现其他子问题的代码！\n\n# ⚠️ 防重复指令\n禁止在代码中重复输出相同的注释块！每段注释只写一次，不要大段重复！如果发现自己陷入重复，立即切换到下一个代码段。"

        # 根据是否聚焦调整图表数量
        fig_count = max(3, self.config.min_figure_count // 3) if is_focused else self.config.min_figure_count
        fig_per_class = max(1, fig_count // 3)

        # P1 质检反馈（如果有）
        p1_note = ""
        if p1_feedback:
            p1_note = f"""
## ⚠️ P1 质检反馈（必须针对性优化！）
P1 阶段通过了门禁，但质检发现了以下问题，P2 阶段必须重点解决：

{p1_feedback[:3000]}

请在 P2 代码中针对性优化上述问题，尤其关注：
- 结果质量提升（搜索精度、算法选择）
- 约束处理正确性
- 数值稳定性
"""

        user_msg = f"""P1已通过，现在进行全量实现（P2阶段）：{p1_note}

## 题目分析报告
{modeling_report[:5000]}

## 术语表格
{terminology_table[:3000]}

请完成：
1. 全量计算和参数扫描（敏感性分析）
2. 生成三类图（原始数据图、过程图、结果图），每类至少{fig_per_class}张，合计至少{fig_count}张
3. {"该子问题在三类图中各至少1张" if is_focused else "每个子问题在三类图中各至少1张"}
4. 生成结果表格
5. 生成复现清单

## 重要规则

### 禁止重复代码（最高优先级！）
- **禁止大段重复相同的注释块！** 每段注释只写一次，不要复制粘贴相同的注释内容！
- **如果你发现自己写了重复的注释，立即停止并删除重复部分！**
- 代码中的注释应简洁，每段注释不超过 2 行

### 中文显示（关键）
- **所有图表的标题、轴标签、图例、注释必须使用中文**
- **代码开头必须配置中文字体**：
```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'
```

### 代码可以多文件组织
- 可以用 `# file: 文件名.py` 标记每个文件的起始，第一个文件为主入口文件
- 文件间可以通过 `import` 互相引用（如 `from utils import helper_func`）
- 所有 import 必须来自标准库或已安装的第三方库，不要 import 本项目其他模块
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
- 图表至少生成 3 张（场景图、过程图、结果图各1张），不要为了凑数而增加代码量

### 精细搜索策略（国赛关键）
- 对网格搜索/参数扫描类任务，必须采用"先粗后精"两阶段搜索：
  1. **粗搜索**：大范围、大步长，确定最优解的大致区域
  2. **精细搜索**：在粗搜索最优解附近，将步长缩小 5~10 倍，进行局部加密
  3. 输出精细搜索前后的对比，展示优化幅度
  4. 例如：参数步长从 10 单位 → 1 单位，角度步长从 30° → 3°
- **搜索点数不能随意减少！必须根据建模报告中的精度要求计算最小搜索点数**

### ⚠️ 计算效率要求（关键！防止超时）
- **禁止使用超过 3 层的嵌套 for 循环**进行参数扫描。必须使用以下方法之一：
  1. `np.meshgrid()` + 向量化计算（推荐）
  2. `itertools.product()` + 单层循环
  3. `scipy.optimize.brute()` 的向量化版本
- **时间步长 ≥ 0.5s**：仿真时间步长不得小于 0.5s，禁止使用 0.1s
- **差分进化参数**: `popsize=8, maxiter=20, tol=0.01`（不要使用 popsize=10, maxiter=30）
- **粗搜索总组合数 ≤ 3000**：如 8 方向 × 4 参数值 × 8 时间点 × 4 参数值 = 1024 组合（良好）
- 在代码中输出进度信息，如 `print(f"粗搜索进度: {{i+1}}/{{n_total}}")`

### 约束处理方法（国赛关键）
- 对于迭代类优化算法（遗传算法、粒子群等），必须正确处理约束：
  1. **可行解初始化 + 可行解变异**（最优方案）：初始化时只生成满足约束的个体，变异操作确保子代仍然满足约束
  2. **惩罚函数法**（次优方案）：对违反约束的个体施加惩罚项，惩罚系数随进化代数动态增加
  3. **修复法**（备用方案）：对违反约束的个体进行修复，映射到最近可行解
- **禁止**：直接返回 0 作为不可行解的适应度值！这会导致算法完全失效。

### ⚠️ 适应度函数符号设计（P0 级 — 违者必错！）
- 差分进化/遗传算法默认**最小化**目标函数。如果目标是最大化（如遮蔽时间/覆盖率/收益），必须对目标值取反
- **正确写法**:
  ```python
  def fitness(params):
      raw = evaluate(params)  # 原始目标值（越大越好）
      penalty = check_constraints(params)  # 惩罚值（≥0，比例于违反程度）
      # 惩罚项不参与符号翻转，确保优化器在不可行区域也有梯度
      return -(raw) + penalty
  ```
- **错误写法**（会导致负值结果！）:
  ```python
  return -(raw - penalty)  # ❌ 当 penalty > raw 时，-(raw - penalty) 为正，
                            # 优化器会寻找 penalty 最大的解（即约束违反最严重的解）
  ```
- **为什么 `-(raw - penalty)` 会失败**:
  - 当 penalty > raw 时，`-(raw - penalty) = penalty - raw > 0`，而 `-(raw) < 0`
  - 优化器最小化 fitness → 会选择 `-(raw - penalty)` 更小的值 → 即 penalty 更大的参数
  - 结果: 优化器主动寻找违反约束的解，产生负值结果

### 资源充分利用（国赛关键）
- 必须使用全部可用资源（如：所有可用单元全部使用，每个单元使用全部容量）
- 如果资源未用满，必须在输出中说明原因
- 多资源协同策略：同一平台的多个资源应形成连续或重叠的有效时间窗口

### 输出格式
- 使用 ```python 代码块包裹完整代码
- 代码块后附上运行说明
- 图表保存到 figures/ 目录（同时保存 PNG 和 SVG 格式，确保矢量图可用于论文）
- 结果保存到 results/ 目录
- 必须输出资源利用率统计

请输出完整代码和所有结果。"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt)

    def fix_code(
        self,
        feedback: str,
        messages: List[BaseMessage],
        project_root: str,
    ) -> str:
        prompt = self.load_system_prompt().replace("{project_root}", project_root)

        # 分析失败类型，给出针对性指导
        feedback_lower = feedback.lower()
        if "超时" in feedback or "timeout" in feedback_lower:
            targeted_guide = """
## ⚠️ 超时专项修复
- **减少迭代算法参数**：种群大小降到 30，迭代次数降到 30，收敛停滞阈值降到 8
- **增大时间步长**：仿真步长 ≥ 2.0s
- **减少搜索组合数**：粗搜索组合数 ≤ 500
- **优先使用贪心/启发式算法**：放弃复杂的 GA/PSO/DE，改用贪心+局部搜索
- **移除不必要的计算**：去掉敏感性分析、蒙特卡洛验证（这些是 P2 的任务）
- **使用向量化替代嵌套循环**：如果仍有嵌套循环，用 np.meshgrid 替代"""
        elif "结果为零" in feedback or "全零" in feedback or "zero" in feedback_lower:
            targeted_guide = """
## ⚠️ 零值结果专项修复
- **检查适应度函数符号**：确认 `return -(raw) + penalty` 而非 `return -(raw - penalty)`
- **检查约束处理**：确认不可行解使用比例惩罚（penalty ∝ 违反程度），而非直接返回 0
- **检查初始种群**：确保所有初始个体在可行域内（使用约束感知的初始化）
- **添加调试输出**：在代码中 print 适应度值和惩罚值，确认优化过程正常"""
        elif "负值" in feedback or "negative" in feedback_lower:
            targeted_guide = """
## ⚠️ 负值结果专项修复
- **适应度函数符号错误**：最大化问题的适应度应为 `return -(raw) + penalty`，不是 `return -(raw - penalty)`
- **检查惩罚项设计**：当 penalty > raw 时，`-(raw - penalty)` 为正，优化器会寻找违反约束最严重的解
- **添加断言**：`assert penalty >= 0`，确保惩罚项非负
- **打印诊断信息**：在每次迭代中 print 原始值和惩罚值"""
        else:
            targeted_guide = """
## ⚠️ 通用修复
- 检查代码语法是否正确
- 确保所有 import 可用
- 确认所有变量在使用前已定义
- 检查是否有命名冲突或缩进问题"""

        user_msg = f"""质检反馈了以下问题，请修正代码：

## 反馈
{feedback}

{targeted_guide}

请修正代码并重新输出。

⚠️ **禁止大段重复注释！** 每段注释只写一次，不要复制粘贴相同的注释内容！
⚠️ P1 阶段代码总行数控制在 400 行以内，优先保证代码能跑通而非追求最优解。
⚠️ 所有图表标题、轴标签、图例、注释必须使用中文，代码开头必须配置中文字体。
⚠️ 如果反馈指出搜索精度不足或结果过低，优先使用贪心算法快速得到一个可行解，而非增加搜索点数。
⚠️ 代码可以分成多个文件，用 `# file: 文件名.py` 标记每个文件的起始。第一个文件为入口文件，其他文件可通过 import 引用。"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt)