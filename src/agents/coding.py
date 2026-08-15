import re
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

        # ====== 加载算法资产（关键！） ======
        algo_index = self._load_algorithm_index()
        if algo_index:
            parts.append(f"\n\n# 算法索引（编写代码前必须查阅）\n\n{algo_index}")

        # 加载优化算法详细说明（最常用）
        optimization_algo = self._load_algorithm("01-优化算法说明.md")
        if optimization_algo:
            parts.append(f"\n\n# 优化算法详细说明\n\n{optimization_algo}")

        # 加载遗传算法说明
        ga_algo = self._load_algorithm("08-遗传算法说明.md")
        if ga_algo:
            parts.append(f"\n\n# 遗传算法说明\n\n{ga_algo}")

        # 加载粒子群优化算法说明
        pso_algo = self._load_algorithm("09-粒子群优化算法说明.md")
        if pso_algo:
            parts.append(f"\n\n# 粒子群优化算法说明\n\n{pso_algo}")

        # 加载模拟退火算法说明
        sa_algo = self._load_algorithm("10-模拟退火算法说明.md")
        if sa_algo:
            parts.append(f"\n\n# 模拟退火算法说明\n\n{sa_algo}")

        # 提示：如需要其他算法类型，用 read_text 工具读取 assets/ 下的对应文件
        skill_root_str = str(self.config.skill_root).replace("\\", "/")
        parts.append(f"""
\n\n# ⚠️ 算法查阅规则（P0 级！必须遵守）
- **编写代码前，必须先查阅上方"算法索引"和对应的算法详细说明**
- 上方已预加载了优化、遗传算法、粒子群、模拟退火的完整说明
- 如果问题类型不是优化类（如预测、评价、图论等），请使用 `read_text` 工具读取对应的算法文件：
  - 预测类：`{skill_root_str}/assets/02-预测类算法说明.md`
  - 评价类：`{skill_root_str}/assets/03-评价类算法说明.md`
  - 图论类：`{skill_root_str}/assets/04-图论与网络分析算法说明.md`
  - 统计类：`{skill_root_str}/assets/05-统计分析与数据处理算法说明.md`
  - 综合类：`{skill_root_str}/assets/06-综合类算法说明.md`
  - 机器学习：`{skill_root_str}/assets/07-机器学习算法说明.md`
  - 假设检验：`{skill_root_str}/assets/11-假设检验说明.md`
  - 回归分析：`{skill_root_str}/assets/12-回归分析说明.md`
- **禁止凭记忆写算法代码！** 必须先查阅上方算法说明中的代码模板和参数建议
- 算法说明中包含了完整的代码模板、参数推荐、可视化方法和参考文献，直接参考使用
""")

        figure_category_count = self.config.min_figure_count // 3
        parts.append(f"""
\n\n# 当前任务配置
- 竞赛类型：{self.config.competition}
- 语言：{self.config.language}
- 项目根目录：{{project_root}}

# 核心任务
1. 读取题目分析报告和术语表格，**特别注意报告开头标注的问题类型**
2. 根据问题类型选择对应的代码实现策略（见下方"问题类型→代码策略"）
3. 数据读取、预处理和核心求解
4. 生成三类图（原始数据图、过程图、结果图），每类至少{figure_category_count}张，合计至少{self.config.min_figure_count}张
5. 生成结果表格和复现清单

# 问题类型→代码策略（根据建模报告中的问题类型自适应）
建模报告开头会标注问题类型（A~G），你必须根据类型调整代码实现策略：

## A. 优化类 → 优化求解器代码
- 实现至少 3 种优化算法对比（见下方"多算法对比要求"）
- 必须实现约束处理（比例惩罚函数法）
- 必须包含：收敛性分析、蒙特卡洛验证、敏感性分析
- 输出：最优解、最优值、收敛曲线、参数敏感性

## B. 预测/预报类 → 预测模型代码
- 划分为训练集（如 70%-80%）和测试集（如 20%-30%）
- 实现至少 2-3 种预测模型对比（如 ARIMA vs LSTM vs XGBoost）
- 必须报告：MAE、RMSE、MAPE、R²
- 必须包含：残差分析图、预测 vs 实际对比图、置信区间
- 输出：预测值、预测区间、模型对比表

## C. 评价/决策类 → 评价算法代码
- 实现至少 2 种评价方法对比（如 TOPSIS vs AHP vs 熵权法）
- 必须包含：权重敏感性分析（权重 ±10% 扰动，观察排序变化）
- 必须包含：指标相关性分析（避免冗余指标）
- 输出：评价得分、排序结果、权重敏感性图

## D. 分类/识别类 → 分类器代码
- 实现至少 2-3 种分类器对比（如 SVM vs 随机森林 vs 逻辑回归）
- 必须进行 K 折交叉验证（K≥5）
- 必须报告：混淆矩阵、精确率/召回率/F1、ROC/AUC
- 必须处理类别不平衡问题
- 输出：分类结果、混淆矩阵图、ROC 曲线

## E. 仿真/模拟类 → 仿真代码
- 必须验证仿真精度（网格收敛性/时间步长收敛性）
- 必须与解析解/已知特例对比验证
- 必须包含参数扫描和相图分析
- 输出：仿真结果、收敛性验证图、参数扫描图

## F. 机理/方程类 → 数值求解代码
- 选择合适的数值方法（odeint/solve_ivp/有限差分）
- 必须验证守恒律（能量/质量/动量守恒）
- 必须验证量纲一致性
- 输出：数值解、守恒律验证、参数影响分析

## G. 统计/数据分析类 → 统计分析代码
- 数据预处理（标准化/归一化/缺失值处理）
- 实现统计方法（回归/假设检验/方差分析）
- 必须进行残差分析和模型诊断（QQ图/残差图）
- 报告效应量和置信区间，而非仅 p 值
- 输出：统计结果表、诊断图、效应量

# 重要规则
- 所有结论必须来自真实代码输出，禁止编造
- 先用小实例跑通，再全量计算
- 图表必须使用出版级样式
- 代码必须可复现
- **禁止大段重复注释！** 每段注释只写一次，不要复制粘贴相同的注释块！

# 代码结构硬性要求（违反将被预执行扫描阻断！）
## 嵌套循环深度限制：优化循环最多3层嵌套
- 允许: for entity in range(N): for param in range(D): evaluate() (2层)
- 禁止: 5层及以上（会被阻断）
### 避免深层嵌套的方法
1. 向量化计算: I, J, K = np.meshgrid(range(N), range(M), range(K))
2. itertools.product()展平: for i, j, k in product(range(N), range(M), range(K)) (合并多个循环)
3. 函数分解: def 处理实体(id): ... ; results = [处理实体(i) for i in range(N)]

## 坐标下降维度限制：最多优化30个关键参数
## 高维问题(>20参数)强制策略
1. 逐实体分解优化（推荐）：外层遍历实体，每次DE只优化低维空间(4-6维)
2. 随机采样+局部优化：先随机生成50个解，取前10个局部优化

## 代码自检清单
- [ ] 循环嵌套<=3层, 坐标下降<=30维, DE popsize<=10且maxiter<=30
- [ ] PSO n_particles<=20且max_iter<=50, 总评估次数<=500
- [ ] 目标函数用比例惩罚(禁止return固定值如1e6), 时间步长DT>=0.2

## 实际案例参考(已验证通过扫描)
### 案例1: 逐导弹分解优化 - 外层遍历实体,每次DE只优化4维,安全!
### 案例2: itertools.product()展平 - 合并3个循环为1个迭代器,实际只有2层嵌套
### 案例3: 敏感性分析/蒙特卡洛 - 函数名含sensitivity关键词,自动豁免

## 🔴🔴🔴 几何可行性分析（物理问题强制第一步！）🔴🔴🔴

### ⚠️ 警告：不执行此步骤将导致结果全为0或NaN！

**识别标准**（满足任一即必须执行）：
- 题目涉及运动物体（无人机/导弹/车辆/机器人）
- 有初始位置和速度限制
- 有时间约束（必须在T秒内完成）
- 需要优化投放点/拦截点/相遇点

### ❌ 常见致命错误（会导致P2失败！）

**错误1：直接在目标位置构造解**
```python
# 错误代码 - 会导致所有解为None!
burst_pos = TARGET_REAL  # 例如 [0, 200, 0]
plan = make_bomb(burst_pos, t, uav)
# 结果: plan = None (因为距离太远无法到达)
```

**错误2：不检查速度约束**
```python
v = distance / time
# 如果 v > V_MAX 或 v < V_MIN → 解不可行!
```

**错误3：从全零解开始优化**
```python
# 如果贪心构造返回空列表 → 初始解=0 → 优化结果还是0!
```

### ✅ 正确流程（必须按顺序执行！）

#### Step 1: 计算每个运动实体的可达区域
```python
def calc_reachable_region(init_pos, v_min, v_max, t_max):
    # 计算实体在时间t_max内能到达的环形区域
    r_min = v_min * t_max  # 最小到达距离
    r_max = v_max * t_max  # 最大到达距离

    def can_reach(target_pos):
        dist = np.linalg.norm(target_pos - init_pos)
        return r_min <= dist <= r_max

    return dict(
        center=init_pos,
        r_min=r_min,
        r_max=r_max,
        can_reach=can_reach
    )

# 示例：计算所有无人机的可达区域
uav_regions = dict()
for uav_name in UAV_NAMES:
    uav_regions[uav_name] = calc_reachable_region(
        UAV_INIT[uav_name], V_UAV_MIN, V_UAV_MAX, T_MAX
    )
    # 打印诊断信息
    region = uav_regions[uav_name]
    target_dist = np.linalg.norm(TARGET_REAL - UAV_INIT[uav_name])
    print(# f-string示例: {{uav_name}}: 可达范围 [{{region['r_min']:.0f}}, {{region['r_max']:.0f}}]m, "
          f"到目标距离 {{target_dist:.0f}}m, "
          f"状态: {{'可达' if region['can_reach'](TARGET_REAL) else '不可达'}}")
```
**预期输出示例**：
```
FY1: 可达范围 [4900, 9800]m, 到目标距离 17801m, ❌ 不可达!
FY2: 可达范围 [4900, 9800]m, 到目标距离 12138m, ❌ 不可达!
FY3: 可达范围 [4900, 9800]m, 到目标距离 3066m, ✅ 可达!
...
```

#### Step 2: 如果目标不可达，寻找拦截点
```python
def find_interception_points(missile_name, uav_region, t_range, n_samples=50):
    # 在导弹轨迹上寻找无人机可达的拦截点
    points = []

    for t in np.linspace(t_range[0], t_range[1], n_samples):
        # 计算导弹在时刻t的位置
        m_pos = get_missile_position(missile_name, t)

        # 检查该点是否在无人机可达范围内
        if uav_region['can_reach'](m_pos):
            points.append(dict(
                time=t,
                position=m_pos,
                missile=missile_name
            ))

    return points

# 为每对(导弹,无人机)寻找拦截点
interception_map = dict()  # 存储每对(导弹,无人机)的拦截点列表

for missile in MISSILE_NAMES:
    interception_map[missile] = dict()
    for uav in UAV_NAMES:
        points = find_interception_points(
            missile, uav_regions[uav],
            t_range=(10, T_MAX-10),  # 避开边界
            n_samples=100
        )
        interception_map[missile][uav] = points

        if len(points) > 0:
            print(# f-string示例: {{missile}}-{{uav}}: 找到 {{len(points)}} 个拦截点")
        else:
            print(# f-string示例: {{missile}}-{{uav}}: ⚠️ 无可用拦截点!")
```

#### Step 3: 从拦截点生成可行初始解
```python
def generate_feasible_solution(interception_map, max_bombs_per_uav=3):
    # 基于拦截点生成可行的初始投放策略
    solution = []
    uav_usage = dict((u, 0) for u in UAV_NAMES)

    # 优先选择时间窗口较好的拦截点
    for missile in MISSILE_NAMES:
        best_points = []
        for uav in UAV_NAMES:
            if uav_usage[uav] >= max_bombs_per_uav:
                continue
            points = interception_map[missile][uav]
            if len(points) > 0:
                # 选择中间时间点的拦截（更稳定）
                mid_idx = len(points) // 2
                best_points.append((points[mid_idx], uav))

        if len(best_points) > 0:
            # 选择距离最优的点
            point, uav = best_points[0]  # 简单策略：取第一个

            # 反推无人机参数
            bomb_params = calc_uav_params_for_interception(
                UAV_INIT[uav], point['position'], point['time']
            )

            if bomb_params is not None:
                solution.append(dict(
                    uav=uav,
                    missile=missile,
                    theta=bomb_params['theta'],
                    v=bomb_params['v'],
                    t_drop=bomb_params['t_drop'],
                    t_burst=point['time'],
                    interception_point=point['position']
                ))
                uav_usage[uav] += 1
                print(# f-string示例: ✓ 添加弹: {{uav}} -> {{missile}}, t={{point['time']:.1f}}s")

    return solution

# 生成初始解
initial_solution = generate_feasible_solution(interception_map)

if len(initial_solution) == 0:
    print("❌ 错误: 无法生成任何可行解! 检查几何约束是否过严")
    # 兜底：使用宽松的随机采样
    initial_solution = fallback_random_sampling()
else:
    print(# f-string示例: \n成功生成 {{len(initial_solution)}} 枚可行弹的初始解")
```

#### Step 4: 两阶段优化（在可行解基础上精细优化）
```python
def two_phase_optimization(initial_sol):
    # Phase 1: 局部微调 -> Phase 2: 全局精优

    # Phase 1: 在每个初始解附近网格搜索
    refined_solutions = []
    for bomb in initial_sol:
        # 创建紧凑的搜索边界（围绕初始解±10%）
        bounds = [
            (bomb['theta'] - 0.1, bomb['theta'] + 0.1),
            (bomb['v'] * 0.9, bomb['v'] * 1.1),
            (bomb['t_drop'] - 1, bomb['t_drop'] + 1),
            (bomb['t_burst'] - 1, bomb['t_burst'] + 1)
        ]

        # 小规模DE优化
        result = differential_evolution(
            lambda x: -evaluate_single_bomb(x, bomb),
            bounds,
            maxiter=10,
            popsize=5,
            seed=42
        )
        refined_bomb = decode_to_bomb(result.x, bomb)
        refined_solutions.append(refined_bomb)

    # Phase 2: 协调优化（可选）
    final_result = coordinate_descent_refinement(refined_solutions)

    return final_result
```

### 📊 必须输出的诊断信息
```python
print("="*60)
print("几何可行性诊断报告")
print("="*60)
for uav in UAV_NAMES:
    region = uav_regions[uav]
    print(# f-string示例: \n{{uav}}:")
    print(# f-string示例:   初始位置: {{UAV_INIT[uav]}}")
    print(# f-string示例:   可达距离: [{{region['r_min']:.0f}}, {{region['r_max']:.0f}}]m")
    print(# f-string示例:   目标距离: {{np.linalg.norm(TARGET_REAL-UAV_INIT[uav]):.0f}}m")
    print(# f-string示例:   状态: {{'可达' if region['can_reach'](TARGET_REAL) else '不可达'}}")

print(# f-string示例: \n初始解质量: {{len(initial_solution)}} 枚可行弹")
print(# f-string示例: 初始遮蔽时间: {{evaluate_strategy(decode_solution(initial_solution)):.2f}}s")
```

### ✅ 验证清单（提交前必须确认）
- [ ] 已输出每个实体的可达区域诊断信息
- [ ] 已检测并处理了不可达情况
- [ ] 初始解包含至少5枚以上的可行弹
- [ ] 初始遮蔽时间 > 0（不为全零）
- [ ] 优化算法从非零初始解开始

# 可用的第三方库（已确认安装，可直接 import）
运行时环境中已安装以下第三方库，你可以直接 import 使用：
- **科学计算**: numpy (2.4), scipy (1.17), pandas (3.0)
- **优化算法**: scipy.optimize — 以下函数必须优先使用，禁止手写嵌套循环替代：
  - `differential_evolution` — 全局优化（差分进化），适合多峰、非凸、带约束问题
  - `minimize` — 局部优化（L-BFGS-B、SLSQP、Nelder-Mead 等）
  - `dual_annealing` — 模拟退火全局优化
  - `basinhopping` — 盆地跳跃全局优化
  - `brute` — 网格搜索（仅用于小规模粗搜索）
  - `linprog` — 线性规划
- **空间计算**: scipy.spatial — `distance`（距离矩阵）、`KDTree`（最近邻搜索）、`ConvexHull`（凸包）
- **数值计算**: scipy.integrate（积分）、scipy.interpolate（插值）、scipy.stats（统计分布）
- **可视化**: matplotlib (3.11)
- **文件读写**: openpyxl, PyPDF2, python-docx, requests
- **⚠️ 以下库未安装，不要 import**：sklearn、sympy、seaborn、pytorch/torch、tensorflow、xgboost、lightgbm、networkx、statsmodels、cvxpy、PuLP、plotly、folium、pyecharts、wordcloud、gensim

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
- [ ] 最大嵌套循环层数 ≤ 3？____（填写实际层数，跨函数调用链的嵌套也算！注意：`itertools.product()` 展平只算 1 层）
- [ ] 差分进化 popsize ≤ 10 且 maxiter ≤ 30？____
- [ ] 粒子群优化 粒子数 ≤ 20 且 迭代 ≤ 50？____
- [ ] 单次 evaluate 调用的总迭代次数 < 5000？____（填写估算值）
- [ ] 总评估成本 < 100,000？____（= 优化算法调用次数 × 单次 evaluate 迭代次数）
- [ ] 至少 3 种不同算法对比？____（填写算法名称，如：网格搜索+差分进化+粒子群优化）
- [ ] 至少 1 种全局优化算法？____（如差分进化/粒子群优化/遗传算法）
- [ ] 至少 1 种局部搜索算法？____（如贪心/爬山/坐标下降）
- [ ] 多目标全覆盖？____（每个目标都有独立优化结果，非零）
- [ ] 单点测试通过？____（选取一组手动参数，evaluate() 返回非零值）
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
  **正确做法**：将所有搜索参数放入参数向量，交给 `scipy.optimize.differential_evolution`！
  ```python
  from scipy.optimize import differential_evolution
  
  def objective(params):
      t1, dt1, t2, dt2, t3, dt3 = params  # 解包参数向量
      # 直接计算目标值，不做任何内部枚举
      return -compute_coverage(t1, dt1, t2, dt2, t3, dt3)
  
  bounds = [(0, T_max), (0, 5), (0, T_max), (0, 5), (0, T_max), (0, 5)]
  result = differential_evolution(objective, bounds, maxiter=30, popsize=10, seed=42)
  ```
  - 优化器负责搜索参数空间，目标函数只做单次计算
  - **禁止在目标函数内部做任何枚举或循环**
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
  - 正确示例 A（展平）：`from itertools import product; for t1, dt1, t2, dt2 in product(times, deltas, times, deltas):`  # 1 层
  - 正确示例 B（优化器）：将 (t1, dt1, t2, dt2, ...) 全部放入参数向量，交给 `differential_evolution` 或 PSO 搜索
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
  - 必须在代码中写入：`with open('results/sensitivity.csv', 'w', newline='', encoding='utf-8-sig') as f: csv_writer = ...`
  - **必须使用 `encoding='utf-8-sig'`**（带 BOM），否则中文列名在 Excel 中会乱码！
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

# ⚠️ 建模-编码转换常见错误（P0 级——直接在代码中检查！）

## 通用错误模式（适用于任何问题类型）

### 1. 初始条件/边界条件遗漏
- 建模报告中的初始条件（如初速度、初始位置、初始状态）必须在代码中显式使用
- 错误：`next_state = state + dt * dynamics(state)`（忽略了初速度传递）
- 正确：`next_state = state + v0 * dt + 0.5 * dt**2 * dynamics(state)`，其中 v0 来自建模报告

### 2. 硬编码常数代替模型参数
- 所有数值常数必须从建模报告的公式推导，不可手动硬编码
- 错误：`t_end = min(t_start + T_duration, 70.0)`（70.0 是凭空硬编码的）
- 正确：`t_end = min(t_start + T_duration, t_max_from_model)`（t_max_from_model 从建模报告推导）

### 3. 参数对称性破坏
- 如果建模报告中的 N 个实体具有不同参数，代码中必须保持差异
- 错误：所有实体的最优参数完全相同（说明优化器未真正工作）
- 正确：每个实体的参数独立优化，利用其不同的初始条件

### 4. 几何/空间关系判定错误
- 涉及空间位置判定的问题，必须验证几何逻辑：
  - 实体是否在目标的影响范围内？
  - 距离/角度判定条件的方向是否正确？
  - 时间窗口是否与目标活动时间重叠？
- 错误：判定条件取反（如"在前方"写成"在后方"）
- 正确：基于建模报告的几何关系图，逐项验证每个判定条件

### 5. 量纲/单位不一致
- 所有公式左右两侧的量纲必须一致
- 涉及角度的三角函数参数必须使用弧度制（除非明确标注为角度制）
- 不同来源的数据必须统一单位后再计算

### 6. 数值稳定性问题
- 大数除小数可能导致溢出，需使用对数变换或归一化
- 指数函数参数过大（>709）会导致 inf，需截断或重参数化
- 矩阵求逆前检查条件数，病态矩阵需正则化

### 7. 概率/统计模型常见错误
- 概率密度函数的参数化方式（如 scipy 使用 scale 而非 rate）
- 混淆概率和质量函数（PMF vs PDF）
- 假设检验中混淆单侧和双侧检验
- 置信区间使用正态近似而不检查样本量

### 8. 微分方程数值解常见错误
- ODE 求解器的时间步长与输出步长混淆
- 刚性问题（stiff）使用显式方法导致发散
- 边界条件数量与方程阶数不匹配
- 初值敏感性问题未做参数扫描

# 多算法/多模型对比要求（国赛关键 — 必须输出至少 3 种独立方法！）
**根据建模报告中的问题类型，选择对应的对比策略：**

## A. 优化类 → 至少 3 种优化算法对比
- 必须包含至少 1 种全局优化算法（差分进化/粒子群优化/遗传算法/模拟退火）和至少 1 种局部搜索算法（贪心/爬山/坐标下降）
- **注意**：粗搜索 + DE 精化 ≠ 两种算法！它们是一个流水线，不是独立算法对比。
- **符合要求的算法组合示例**：
  - 网格搜索 vs 差分进化 vs 粒子群优化（三个独立优化器）
  - 贪心法 vs 遗传算法 vs 模拟退火（三个独立策略）
- 在代码中输出对比表格，格式如下：
  ```
  print("算法对比:")
  print(f"  算法                     结果                   耗时(s)     ")
  print(f"  网格搜索                 xxx.xx                 x.xx      ")
  print(f"  差分进化                 xxx.xx                 x.xx      ")
  print(f"  粒子群优化               xxx.xx                 x.xx      ")
  ```

## B. 预测类 → 至少 2-3 种预测模型对比
- 选择不同原理的模型（如统计模型 vs 机器学习模型 vs 深度学习模型）
- **符合要求的模型组合示例**：
  - ARIMA vs Prophet vs LSTM（三个不同原理）
  - 线性回归 vs XGBoost vs 随机森林（三个不同复杂度）
  - 指数平滑 vs SARIMA vs LightGBM（三个不同方法）
- 在代码中输出对比表格，格式如下：
  ```
  print("模型对比:")
  print(f"  模型                     MAE         RMSE        R²         ")
  print(f"  ARIMA                   x.xx        x.xx        x.xx      ")
  print(f"  XGBoost                 x.xx        x.xx        x.xx      ")
  print(f"  LSTM                    x.xx        x.xx        x.xx      ")
  ```

## C. 评价类 → 至少 2 种评价方法对比
- 选择不同赋权/评价原理的方法
- 输出：TOPSIS vs AHP vs 熵权法 的排序对比表

## D. 分类类 → 至少 2-3 种分类器对比
- 选择不同原理的分类器（如基于距离 vs 基于树 vs 基于概率）
- 输出：SVM vs 随机森林 vs 逻辑回归 的精确率/召回率/F1 对比表

## E/F/G 类 → 至少 2 种方法对比
- 选择不同原理或不同精度的求解方法
- 输出对比结果表

# 国赛级结果质量要求（关键！不满足将导致 P2 自动 FAIL）
- **结果质量阈值**：优化结果必须 ≥ 理论最大值的 15%（国赛基本要求），争取 ≥ 30%（冲击国一）
  - 如果首次运行结果 < 15%，必须自动触发以下优化：
    1. 缩小搜索空间（基于物理约束去掉不可行区域）
    2. 增加搜索精度（步长缩小 5-10 倍）
    3. 尝试不同的算法组合
  - 如果 3 种算法结果都 < 15%，说明模型设计或搜索策略有根本性问题，需要重新分析
- **多目标全覆盖**：如果题目涉及多个目标（如多枚导弹、多个节点、多个任务），必须确保每个目标都有非零结果
  - 不得出现"仅部分目标有结果，其余为零"的情况
  - 每个目标的策略参数应独立优化，而非所有目标共享同一组参数
  - 输出中必须包含每个目标的独立结果统计
- **蒙特卡洛鲁棒性**：蒙特卡洛验证的均值应 ≥ 最优值的 50%，失败率（零值占比）应 ≤ 20%
  - **这是国赛最关键的分水岭！** 如果 MC 均值远低于最优值（如仅 30%），说明策略是"碰运气"而非"可靠优化"
  - 提高鲁棒性的核心方法：
    1. **在目标函数中加入鲁棒性惩罚项**：`penalty_robust = lambda_robust * (max_sensitivity)`，λ_robust 建议取 0.1~0.5
    2. **使用更保守的策略参数**：在最优解附近选择"平坦区域"（Hessian 矩阵条件数小的区域），而非仅追求峰值
    3. **引入参数安全余量**：将最优参数向可行域内部收缩 5%~10%，牺牲少量最优值换鲁棒性
    4. **使用鲁棒优化方法**：如 min-max 优化（worst-case optimization）或机会约束规划
  - 蒙特卡洛扰动应覆盖所有关键参数（包括物理参数和决策参数），而非仅部分参数
  - 扰动幅度建议：物理参数 ±5%，决策参数 ±3%，时间参数 ±2%

# 多目标全覆盖策略（国赛关键 — 防止部分目标结果为零）
- 如果题目涉及多个独立目标（如多个子任务、多个评价对象、多个时间段），必须为每个目标独立执行求解：
  ```python
  for target_id in targets:
      best_params[target_id] = solve_for_target(target_id)
  ```
- 不得用一个目标的参数直接套用到其他目标（每个目标的条件/约束可能不同）
- 不得只求解一个目标然后将结果复制到其他目标
- 输出中必须包含每个目标的独立结果和汇总

# 遗传算法/进化算法初始化策略（关键！防止零值结果）
- **禁止纯随机初始化！** 在高维/窄可行域问题中，纯随机初始化几乎不可能命中有效解。必须结合问题结构进行智能初始化：
  - 分析问题的几何/物理约束，预计算可行域的大致范围
  - 基于预计算结果生成初始种群（至少 50% 个体来自智能初始化）
  - 示例（通用）：如果问题是路径规划，先计算目标位置的几何关系，再反推初始参数；如果问题是参数拟合，先用最小二乘得到粗略估计，再在其附近初始化
- **初始化时至少 30% 的个体使用随机扰动**（而非全部智能初始化），以增加搜索多样性
- **时间/顺序相关参数应基于关键时间节点反向推算**，而非随机采样
- **资源分配应覆盖所有目标**：确保每个目标都分配到至少最低限度的资源

# 零结果诊断（关键！当所有算法返回零时必须检查）
如果 `evaluate()` 函数对所有输入都返回 0（或所有算法对比结果都是 0），说明存在**模型实现错误**而非优化算法问题：

1. **几何/空间位置验证**（如涉及空间关系）：
   - 核心实体是否在目标的影响范围内？
   - 距离/角度判定条件是否合理（如有效半径、阈值）？
   - 核心实体的有效时间窗口是否与目标活动时间有重叠？
2. **判定逻辑验证**（通用）：
   - 错误：判定条件取反或方向错误
   - 正确：基于建模报告的判定逻辑，逐项验证每个条件
3. **时间窗口验证**（如涉及时间）：
   - 实体的有效时间窗口：[t_start, t_start + T_duration]
   - 目标的活动时间窗口：需要显式计算
4. **强制单点测试**：在代码开头加入诊断输出（⚠️ 必须保留 `test_total = evaluate(test_params)` 赋值行，禁止省略后直接使用 `test_total`！）
   ```python
   # 单点诊断：测试一组参数是否产生非零结果
   # ⚠️ 必须定义 test_params（替换为实际参数），并确保 evaluate 函数已定义
   test_params = [/* 替换为实际测试参数，如初始位置、速度等 */]
   test_total = evaluate(test_params)  # ⚠️ 此行不可省略！必须先赋值再使用
   print(# f-string示例: [诊断] 单点测试结果: {{test_total:.2f}}")
   if test_total < 1e-6:
       print("[诊断] 警告：evaluate() 返回零！模型实现可能有误，请检查：")
       print("  1. 核心实体是否在目标影响范围内？")
       print("  2. 判定条件是否过于严格？")
       print("  3. 时间窗口是否有重叠？")
       print("  4. 初始条件/边界条件是否正确？")
       # ⚠️ 禁止在此处 return！继续执行后续代码（图表、算法对比等），
       # 让所有结果自然为 0，这样可以产出完整的图表和输出供评审
   # 注意：test_total 的作用域到此为止，后续代码如需使用请重新赋值
   ```
5. **可视化诊断（先画图再诊断）**：在单点诊断之前，先绘制 3D 场景图（轨迹/位置/几何关系），
   目视确认模型行为。这样即使后续诊断返回零，至少有场景图可供评审

# 几何模型正确性自检（关键！防止全零结果）
在编写任何优化代码之前，必须先完成以下几何验证：
- 选取一组"应该能产生作用"的手动参数（如：在目标路径正上方放置实体）
- 验证 `evaluate()` 对这组参数返回非零值
- 如果返回零，说明几何模型（非优化算法）有问题，打印警告但**不要 return**，继续执行后续代码
- 即使几何模型返回零，也要生成完整的图表（场景图、轨迹图等）和算法对比输出，作为工作量证明
- 修复几何模型是下一轮迭代的任务，本轮代码应尽可能产出完整的输出

# 🔴 全局优化算法使用指南（国赛关键 — 差分进化/PSO 必须使用！）
- **差分进化（differential_evolution）是最推荐的全局优化算法**，原因：
  1. 不需要梯度信息，适用于非光滑、非凸目标函数
  2. 自带种群多样性维护，不易陷入局部最优
  3. 可处理带约束优化（通过惩罚函数法）
  4. scipy 内置实现，无需额外安装
- **差分进化必须作为 3 种对比算法之一**（除非问题类型不适用）
- 自适应参数策略（根据搜索空间维度 D 调整）：
  - D ≤ 5: popsize=10, maxiter=30, tol=0.01
  - 5 < D ≤ 10: popsize=10, maxiter=25, tol=0.01
  - 10 < D ≤ 20: popsize=8, maxiter=20, tol=0.01
  - D > 20: 必须使用逐实体分解优化（外层 for 循环，内层 DE 每次 ≤ 6 维）
- **差分进化必须在目标函数外做约束处理**，不可在目标函数内部 return 固定值
- **差分进化结果验证**：必须检查 DE 返回的 success 标志和 fun 值
  - 如果 DE 返回 success=False 或 fun 接近 0，说明 DE 未找到有效解，必须在输出中标注
  - 如果 DE 和另一算法（如随机搜索）结果差异巨大，优先信任非零结果

# 收敛性分析要求（国赛关键）
- 对于迭代类算法（如坐标下降、梯度下降、遗传算法），必须输出收敛曲线
- 代码中自动判断是否收敛：连续 N 代改进量 < ε
- 输出收敛状态：已收敛 / 未收敛（达到最大迭代次数）
- **收敛曲线必须来自真实迭代记录**（在循环中每次 iter 记录 best_value），禁止合成直线

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

# 禁止写法 5：main() 中提前 return — 跳过图表生成！
if test_result <= 0:
    return  # 禁止！跳过了所有 savefig()，论文将没有图表可用
    # 正确做法：打印警告但继续执行，让所有图表正常生成（即使数据全零）
# 正确做法：在目标函数中保持原始值，通过惩罚项引导优化器
```

**为什么固定值惩罚会失败**：假设参数空间 99% 的区域违反约束，所有不可行解都返回 1e6。
优化器看到的景观是：99% 的区域都是 1e6 的"高原"，只有 1% 的区域有梯度。
优化器无法在高原上找到方向，最终随机收敛到某个全零解。

**比例惩罚的关键**：惩罚项必须 `∝` 违反程度。违反越严重，惩罚越大，
这样才能在不可行区域中形成指向可行域的梯度。

# 资源充分利用（国赛关键 — P0 级！）
- **必须使用全部可用资源**（如题目给定的所有设备、所有时间窗口、所有容量上限）
- 如果建模报告指定了资源上限，必须用满或给出未用满的合理理由
- **资源分配必须覆盖所有资源节点**：如果问题中有 N 个可用资源节点，
  必须确保每个节点都分配到至少一项任务，不得将所有任务集中到单一节点。
  - 错误示例：5 架无人机可用，只有 3 架被使用，FY1 和 FY4 完全闲置 → **P0 级错误！**
  - 正确示例：5 架无人机全部参与任务分配，每架至少 1 枚弹
  - 实现方法：在分配算法中引入多样性约束（如每个节点至少分配 floor(N_tasks/N_nodes) 个任务）
- **资源利用率自检**：代码输出中必须包含每个资源节点的使用统计，格式如下：
  ```
  print("资源利用率:")
  for name in NODES:
      print(f"  {{name}}: {{used}}/{{capacity}}")
  print(f"  总利用率: {{sum(used)/sum(capacity)*100:.0f}}%")
  ```
- 如果某资源节点确实无法使用（如物理上不可达），必须在输出中明确说明原因
- 多资源协同策略：同一平台的多个资源应形成连续或互补的使用窗口
- **资源利用率 < 100% 且无合理解释 → P2 自动降级**

# 中文显示要求（P2 阶段关键，P1 阶段由系统自动注入）
- **所有图表的标题、轴标签、图例、注释必须使用中文**
- **P2 阶段代码开头必须配置中文字体**：
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
- **⚠️ P1 阶段以文本输出为主，不强制要求生成图表。中文字体配置由系统自动注入，无需手动写 matplotlib 字体配置。如果需要画调试图，可以 import matplotlib 但不要写 rcParams 配置。此指令覆盖系统提示词中的字体配置要求。**
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
  1. **`itertools.product()` 展平**（最推荐用于构造性搜索）— 将 N 层嵌套改为单层循环，见下方示例
  2. **`scipy.optimize.differential_evolution`** — 将所有搜索变量放入参数向量，目标函数只做单次计算
  3. `scipy.optimize.minimize` — 局部优化（L-BFGS-B、SLSQP 等）
  4. `np.meshgrid()` + 向量化计算（仅适用于 ≤4 维小网格）
  5. `scipy.optimize.brute()` 的向量化版本

  **嵌套展平示例**：
  ```python
  # ❌ 5 层嵌套（会被阻断）
  for a in range(5):
      for b in range(3):
          for c in values:
              for d in items:
                  for e in range(10):
                      if check(a, b, c, d, e):
                          result = (a, b, c, d, e)
  
  # ✅ product() 展平为 1 层
  from itertools import product
  for a, b, c, d, e in product(range(5), range(3), values, items, range(10)):
      if check(a, b, c, d, e):
          result = (a, b, c, d, e)
  ```
- **时间步长 ≥ 0.5s**：仿真时间步长不得小于 0.5s，禁止使用 0.1s
- **差分进化参数**: `popsize=8, maxiter=20, tol=0.01`（严格遵守，不要增大）
- **粗搜索总组合数 ≤ 3000**：如 8 方向 × 4 参数值 × 8 时间点 × 4 参数值 = 1024 组合（良好）
- 在代码中输出进度信息，如 `print(f"粗搜索进度: {{i+1}}/{{n_total}}")`

### ⚠️ 核心模型诊断（关键！防止全零/NaN结果）
在优化前，必须先添加单点诊断测试，确认核心计算模型能返回非零值：
```python
# [诊断] 单点测试：手动构造一组合理参数，确认核心计算函数能否返回非零值
test_params = [合理的默认值, ...]  # 根据问题替换为实际参数
test_result = evaluate(test_params)  # 根据问题替换为实际函数名
print(# f-string示例: [诊断] 单点测试: {{test_result:.4f}}")
if test_result <= 0 or np.isnan(test_result):
    print("[诊断] 警告：核心计算模型返回零值/NaN！请检查数据/公式/判定条件")
    # 如果诊断失败，不要继续优化，先修复核心计算模型
```
**如果诊断返回 0 或 NaN，说明核心计算模型有误，不要使用优化器！先修复模型再优化。**

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
- 结果保存到 results/ 目录（JSON/CSV 文件必须使用 `encoding='utf-8-sig'` 打开，防止中文乱码）
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
- **使用向量化替代嵌套循环**：如果仍有嵌套循环，用 np.meshgrid 替代
- **⚠️ 最深嵌套不得超过 3 层**：将所有搜索变量放入参数向量，使用 DE/PSO 优化器搜索"""
        elif "嵌套" in feedback or "nest" in feedback_lower:
            targeted_guide = """
## ⚠️ 嵌套循环过深 — 必须彻底重构！（最高优先级）
你的代码被预执行扫描阻断，因为存在 **≥4 层嵌套 for 循环**，这会导致计算时间指数爆炸，300s 超时内根本无法完成。

### 方案 A：`itertools.product()` 展平（推荐用于构造性搜索）
将 N 层嵌套循环改为单层 `product()` 循环。**这是最直接、最不易出错的方法**：
```python
from itertools import product

# 之前：5 层嵌套（会被阻断）
# for t0 in time_steps:
#     for j in range(3):
#         for lam in lambdas:
#             for i in avail:
#                 # 计算...
#                 if found: break

# 之后：单层 product() 展平
for t0, j, lam, i in product(time_steps, range(3), lambdas, avail):
    # 同样的计算逻辑，但只是一层循环！
    # 用 continue 代替嵌套中的 break（需要外层配合）
    if not condition_ok:
        continue
    # 计算...
```
**关键技巧**：如需提前退出（原嵌套循环中的 `break`），用标志变量：
```python
found = False
for t0, j, lam, i in product(time_steps, range(3), lambdas, avail):
    if found:
        continue  # 跳过已找到解后的剩余组合
    if 满足条件:
        found = True
        result = (t0, j, lam, i)
```

### 方案 B：使用优化器 DE/PSO（推荐用于参数优化）
```python
from scipy.optimize import differential_evolution

def objective(params):
    a, b, c, d, e = params
    result = compute_result(int(a), int(b), c, d, e)
    return -result  # 负号因为 DE 默认最小化

bounds = [(0, 4), (0, 2), (0, 2), (0, 99), (0, 6.28)]
result = differential_evolution(objective, bounds, maxiter=20, popsize=8, seed=42)
```

### ⚠️ 注意区分
- **构造性搜索**（找初始可行解）→ 用方案 A（`product()`），因为搜索空间小且需要精确枚举
- **参数优化**（改进已有解）→ 用方案 B（DE/PSO），因为搜索空间大且连续
- **分析/验证**（蒙特卡洛/敏感性）→ 可以保留 3-4 层嵌套，这些不算违规

### 核心原则
- **最深嵌套 ≤ 3 层**：超过 3 层必须用 product() 或优化器替代
- **目标函数只做单次计算**：`objective(params)` 接收一个参数向量，内部不做任何枚举
- **禁止在目标函数内写 for 循环进行参数扫描**：参数扫描由优化器完成"""
        elif ("结果为零" in feedback or "全零" in feedback or "均为 0" in feedback or "均为0" in feedback
              or ("zero" in feedback_lower and "非零" not in feedback)
              or re.search(r'(?:均为|全是|所有|全部).*?(?:0\.0|为\s*0)', feedback)
              or re.search(r'总.*?(?:结果|值|时间|收益|得分).*?[=:：]\s*0\.0', feedback)):
            targeted_guide = """
## ⚠️ 零值结果专项修复（两阶段法）

### 第一阶段：诊断（必须先做！找到任意可行解）
在 main() 开头添加以下诊断代码，确认核心计算模型是否正确：
```python
# [诊断] 单点测试：手动构造一组合理参数，确认核心计算函数能否返回非零值
test_params = [合理的默认值, ...]  # 根据问题替换为实际参数
test_result = evaluate(test_params)  # 根据问题替换为实际函数名
print(# f-string示例: [诊断] 单点测试: {{test_result:.4f}}")
if test_result <= 0 or np.isnan(test_result):
    print("[诊断] 警告：核心计算模型返回零值/NaN！请检查：")
    print("  1. 输入数据是否正确加载（路径、列名、单位）")
    print("  2. 核心计算公式是否正确（公式推导、符号、量纲）")
    print("  3. 约束/判定条件是否过于严格（阈值、范围、边界）")
    print("  4. 中间变量是否意外为零（除数、初始化值、默认值）")
```

### 第二阶段：修复（找到可行解后再优化）
- **检查数据加载**：确认输入数据路径、列名、编码正确，数据形状符合预期
- **检查核心公式**：逐项验证公式推导，确认符号、量纲、单位一致
- **检查判定条件**：阈值/边界/范围是否合理，是否过于严格导致无可行解
- **先用简化策略**：在优化器无法找到可行解时，先用随机采样/贪心/穷举找一组可行解作为初始点
- **检查适应度函数符号**：最大化问题的适应度应为 `return -(raw) + penalty`，不是 `return -(raw - penalty)`
- **检查约束处理**：不可行解使用比例惩罚（penalty ∝ 违反程度），而非直接返回 0 或固定大值
- **⚠️ 如果所有算法都返回 0 或 NaN，优先怀疑核心计算模型有误，而非优化器参数问题**"""
        elif "结果质量" in feedback or "质量极低" in feedback or "质量过低" in feedback or "low" in feedback_lower:
            targeted_guide = """
## ⚠️ 结果质量过低修复（实际值 < 理论最大值 10%）

结果质量过低通常意味着搜索策略过于粗糙，而非模型本身有误。重点排查以下方向：

### 1. 候选解密度不足（离散化方法最常见问题）
- 如果使用网格/候选枚举方法，检查每个维度的候选数量是否足够
- 建议：每个连续参数至少 10-20 个候选值，离散参数覆盖所有可能取值
- 如果当前候选总数 < 500，扩展到 1000-5000（确保总计算量在 300s 内）
- 如果候选总数已足够但结果仍低，说明候选点分布不合理，需要调整参数范围

### 2. 搜索策略切换
- 离散化方法达到瓶颈时，切换到连续优化：
  - 将候选生成改为 `differential_evolution` 直接搜索连续参数空间
  - 或：粗搜索（离散化）→ 精搜索（局部优化 refine）的分阶段策略
- 禁止在连续优化中使用过大的离散步长

### 3. 目标函数扁平化
- 检查目标函数是否对参数变化不敏感（如 max(0, x) 在 x<0 时导数为 0）
- 将硬约束（max(0, ...)）改为软约束（比例惩罚）
- 确保目标函数在可行域内是连续的（没有阶梯状跳变）

### 4. 蒙特卡洛失败率过高
- 分析扰动后失败的原因（约束违反？计算溢出？）
- 如果失败原因是约束违反，说明当前解在可行域边界上，需要扩大搜索范围
- 在最优解附近做局部搜索，找一个更鲁棒（远离边界）的解"""
        elif "负值" in feedback or "negative" in feedback_lower:
            targeted_guide = """
## ⚠️ 负值结果专项修复
- **适应度函数符号错误**：最大化问题的适应度应为 `return -(raw) + penalty`，不是 `return -(raw - penalty)`
- **检查惩罚项设计**：当 penalty > raw 时，`-(raw - penalty)` 为正，优化器会寻找违反约束最严重的解
- **添加断言**：`assert penalty >= 0`，确保惩罚项非负
- **打印诊断信息**：在每次迭代中 print 原始值和惩罚值
- **检查嵌套循环深度**：如果代码有 ≥4 层嵌套循环，必须重构！将所有搜索变量放入参数向量，使用 DE/PSO 优化器搜索"""
        elif "p2" in feedback_lower or "编程终检" in feedback:
            targeted_guide = """
## ⚠️ P2 编程终检未通过（通用指导）
P2 门禁要求代码满足国赛标准的完整性和正确性。请仔细阅读反馈中的具体问题，逐项修复。

### ⚠️ 最高优先级：嵌套循环 ≤ 3 层（硬性约束）
修复任何问题时，**必须确保最深嵌套循环不超过 3 层**！如果修复引入新的嵌套循环，必须用 `itertools.product()` 展平或使用 DE/PSO 优化器替代。

### 通用要求
1. **代码可执行**：语法正确，无运行时错误，能在 300s 内完成
2. **三类图表**：场景图/数据图、过程图/分析图、结果图/对比图各至少 1 张
3. **复现清单**：列出所有依赖库版本、随机种子、运行步骤

### 常见失败原因及修复
- **结果质量低**：实际值 < 理论最大值的 10%，需要改进算法/模型
- **算法/模型数量不足**：根据问题类型补充到要求数量
- **缺少蒙特卡洛/交叉验证**：做扰动/重采样验证
- **图表英文标签**：所有 xlabel/ylabel/title/legend 必须使用中文
- **缺少 CSV 输出**：敏感性/预测/分类结果必须写入 CSV 文件"""
        else:
            targeted_guide = """
## ⚠️ 通用修复
- 检查代码语法是否正确
- 确保所有 import 可用
- 确认所有变量在使用前已定义
- 检查是否有命名冲突或缩进问题
- **⚠️ 检查嵌套循环深度**：最深嵌套不得超过 3 层！超过则必须用优化器替代手动循环"""

        # 判断当前是 P1 还是 P2 阶段
        is_p2 = "P2" in feedback or "P2" in str(messages[-1].content) if messages else False
        stage_note = (
            "⚠️ P2 阶段代码总行数控制在 600 行以内，必须包含至少3种算法对比、蒙特卡洛验证、敏感性分析。"
            if is_p2
            else "⚠️ P1 阶段代码总行数控制在 400 行以内，优先保证代码能跑通而非追求最优解。"
        )

        user_msg = f"""质检反馈了以下问题，请修正代码：

## 反馈
{feedback}

{targeted_guide}

请修正代码并重新输出。

⚠️ **禁止大段重复注释！** 每段注释只写一次，不要复制粘贴相同的注释内容！
{stage_note}
⚠️ 所有图表标题、轴标签、图例、注释必须使用中文，代码开头必须配置中文字体。
⚠️ 如果反馈指出搜索精度不足或结果过低，优先使用贪心算法快速得到一个可行解，而非增加搜索点数。
⚠️ 代码可以分成多个文件，用 `# file: 文件名.py` 标记每个文件的起始。第一个文件为入口文件，其他文件可通过 import 引用。"""
        return self.invoke(messages, user_input=user_msg, system_prompt=prompt, use_fix_model=True)