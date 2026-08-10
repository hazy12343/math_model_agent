# 项目规则与迭代记录

## 项目概述
数学建模 Agent —— 自动化数学建模竞赛解题流水线。
- 入口: `app.py` (Streamlit UI)
- 核心: `src/graph.py` (LangGraph 工作流)
- 配置: `src/config.py`
- 诊断: `scripts/diagnose.py` ← 每次跑完 agent 先运行这个

## 迭代流程
```
1. 跑 agent 生成 projects
2. 运行 .venv\Scripts\python.exe scripts/diagnose.py  获取诊断报告
3. 将报告粘贴到对话中
4. 根据报告定位问题，修改项目代码
5. 重复 1-4
```

## 工作流架构
```
init → modeling → m1_check → coding_p1 → p1_check → coding_full
  → code_exec → verify → p2_check → model_comparison → error_analysis
  → writing_w1 → w1_check → writing_full → w2_check → polish → done
                                         ↘ (P2耗尽重试) → failed → done
```

## 关键状态字段
- `code_exec_success: bool` — 代码是否执行成功（决定后续门禁）
- `exec_error: str` — 执行失败时的错误信息
- `focus_question: Optional[str]` — 用户指定的聚焦问题编号（如"5"、"1-3"）
- `quality_gates: dict` — 各阶段质量门禁状态

## 质量门禁
| 门禁 | 阶段 | 失败时行为 |
|------|------|------------|
| M1 | 建模终检 | 重试建模（最多3轮） |
| P1 | 最小代码门禁 | 重试代码生成 |
| P2 | 编程终检 | 代码执行失败时自动FAIL；验证器检测到结果异常（全零值/NaN/负值）时自动FAIL；重试代码生成 |
| W1 | 证据大纲门禁 | 代码执行失败时自动FAIL |
| W2 | 论文终检 | 代码执行失败时自动FAIL |

## 已修复问题记录

### 2026-08-09: 聚焦问题 + 全零值结果放行 + 误差分析幻觉 + 提示词泛化
- 问题1: Agent 忽略用户"只做第X问"的要求
- 修复: 添加 `focus_question` 状态字段 + UI 输入框 + 3 个 Agent 的聚焦指令注入
- 涉及: state.py, chat.py, app.py, graph.py, modeling.py, coding.py, writing.py
- 问题2: LLM 生成代码时大段重复注释
- 修复: 所有 Agent 的系统提示词和 user_msg 添加"禁止大段重复"指令
- 涉及: modeling.py, coding.py, writing.py
- 问题3: 优化返回全零但 P2 门禁放行 → 论文生成
- 修复: `_parse_quality_status()` 扫描全部行，任一 FAIL 即返回 FAIL；P2 门禁新增零值/NaN 硬性检测；P2 耗尽重试后路由到 `failed_node` 而非继续生成论文
- 涉及: graph.py
- 问题4: 误差分析在结果全零时编造大量数值（"0.5~2.0s"、"2.1×10¹²"、"5300倍"）
- 修复: error_analysis_node 新增全零/NaN 检测，异常时输出诊断建议而非幻觉数据；LLM 提示词移除"必须给出具体数值估计"改为"数据不足标注"
- 涉及: graph.py
- 问题5: 模型对比提取器无法处理 `模型A: 0.00 s` 格式
- 修复: `_extract_comparison_table` 新增冒号分隔格式匹配，扩展关键词列表
- 涉及: graph.py
- 问题6: coding.py 提示词含导弹/无人机/烟幕弹等特定问题语言
- 修复: 泛化 GA 初始化策略、自适应搜索精度、资源充分利用章节，移除特定领域术语
- 涉及: coding.py

### 2026-08-10: 全面泛化审查 — 移除所有特定问题语言
- 问题: 用户反馈"你这只是针对这个问题，我要的是泛化能力"。经全面审查，发现 modeling.py、writing.py、quality.py、coding.py、graph.py、verifier.py 共 6 个文件中存在 18 处问题特定语言（导弹/拦截/遮蔽/烟幕/无人机/牛顿力学/运动学/3枚导弹/5架无人机/多弹协同等）
- 修复: 逐文件审查并替换为通用语言，确保 Agent 可处理任意数学建模问题而非仅限导弹拦截
- 涉及: modeling.py（搜索空间示例）、writing.py（参考文献示例、关键词列表保留但已足够泛化）、quality.py（M1/P2 判定标准示例）、coding.py（图表标题示例、进度变量名、资源利用示例）、graph.py（误差分析关键词、失败报告建议、论文规则示例）、verifier.py（NaN 上下文检测关键词）

### 2026-08-10: 超时诊断误报 + LLM 无视效率约束 + P2 截断误判
- 问题1: 超时诊断报告"35 层 for 循环"，实际代码最多 3-4 层——`nested_loop_count` 简单计数所有 `for ` 行，未计算实际嵌套深度
- 修复: 基于缩进分析实际最大嵌套深度，同时报告总循环数和嵌套深度；新增 PSO 参数检测（粒子数、迭代数）；时间步长检测扩展为 DT/dt/TIME_STEP 等多种变量名
- 涉及: graph.py (code_exec_node)
- 问题2: LLM 系统性忽略计算效率约束（DT=0.05 vs 要求≥0.2；popsize=15 vs 要求≤10；maxiter=100 vs 要求≤30；PSO 50粒子×100迭代 vs 无约束）
- 修复: 计算效率规范升级为"⚠️ 硬性约束！违反将导致超时 → P2 门禁 FAIL"，新增违反示例/正确示例对比，新增 PSO 参数硬性上限（粒子≤20/迭代≤50），新增单次 evaluate 调用成本估算公式
- 涉及: coding.py
- 问题3: P2 门禁代码截断 8000 字符 → 复杂代码后半部分（PSO/蒙特卡洛/敏感性分析）被截断 → LLM 误判"缺少算法特征"
- 修复: 代码截断从 8000 提升到 16000 字符；新增"超时场景特别说明"指令：代码超时时从代码文本分析算法完整性，不因输出截断而判算法缺失
- 涉及: quality.py

### 2026-08-10: LLM 仍然生成 8 层嵌套循环 + 双层防线
- 问题: 上轮修复后 LLM 仍生成 8 层嵌套 for 循环（50×30×30×30×20×20×3 = 16 亿次迭代），DE 仍用 popsize=15/maxiter=50。LLM 读完约束后直接写代码，没有"自查"环节
- 修复1（提示词层）: coding.py 新增**"代码生成前自检清单"**——6 项硬性检查，LLM 必须逐项确认后才能输出代码，任一项不满足则重新设计；新增**"禁止的代码模式"**章节——直接展示 LLM 反复犯的 6 层嵌套网格搜索模式并标注"必定超时"
- 修复2（运行时层）: graph.py code_exec_node 新增**"预执行安全扫描"**——代码执行前自动检测嵌套深度、DE/PSO 参数、时间步长违规，输出警告信息
- 涉及: coding.py, graph.py

### 2026-08-10: 固定值惩罚导致优化器全零 + 资源分配不均匀
- 问题1: LLM 用 `return 1e6` 代替 `return 0.0` 作为不可行解惩罚，但本质上仍是固定值惩罚——优化器无法区分"略微违反约束"和"严重违反约束"，在适应度景观中形成"高原"，最终收敛到全零解
- 问题2: 任务分配全给 FY1，其他 4 架无人机闲置——贪心分配只考虑距离，未考虑资源多样性
- 修复1（提示词层）: coding.py 约束处理章节升级——明确列出 `return 0.0`、`return 1e6`、`return 1e9` 三种禁止写法，新增"为什么固定值惩罚会失败"的解释（高原效应），强调比例惩罚 `∝` 违反程度
- 修复2（提示词层）: coding.py 资源分配章节新增"覆盖所有资源节点"要求——给出错误示例（5架无人机全给FY1）和正确示例，建议实现方法
- 修复3（运行时层）: graph.py 预执行扫描新增固定值惩罚检测（`return 1e6`/`1e9` 等）
- 修复4（门禁层）: quality.py P2 门禁新增固定值惩罚代码检测——如果代码中存在 `return 1e6` 标记为 P0 错误并判 FAIL
- 涉及: coding.py, graph.py, quality.py

### 2026-08-10: 多算法缺失 + 敏感性CSV缺失 + 误报修复
- 问题1: LLM 认为"粗搜索 + DE 精化"就是两种算法，只生成一个流水线而非独立算法对比
- 问题2: 敏感性分析只打印到控制台，未写入 CSV 文件
- 问题3: 代码输出 `[NaN检查] 所有计算结果无NaN`，verifier 把"NaN"关键词误判为数值异常
- 问题4: 3 个任务中 1 个未收敛，graph.py 误判为"算法未收敛"P0 错误
- 修复1（提示词层）: coding.py 自检清单新增 3 项——多算法对比、敏感性CSV输出、资源覆盖；多算法对比章节明确"粗搜索+DE ≠ 两种算法"，列出正确/错误示例
- 修复2（提示词层）: 敏感性CSV输出要求从"建议"升级为"必须在代码中写入文件"，给出具体代码示例
- 修复3（verifier）: NaN 检测新增"无NaN"、"未检测到NaN"等否定表述的排除逻辑
- 修复4（graph.py）: "未收敛"检测从一刀切改为分级——全部未收敛→P0；≥3个未收敛→P0；个别未收敛→P1
- 涉及: coding.py, verifier.py, graph.py

### 2026-08-10: 第三轮泛化审查 — 移除所有残留问题特定语言
- 问题: 上轮泛化后，编码提示词和技能参考文档中仍有 10 处残留的问题特定语言（无人机、导弹、投放时刻、起爆时间、FY1、遮蔽时长、三维轨迹判定等）
- 修复: 逐文件替换为通用术语——禁止模式2示例改为"资源节点/目标任务/参数维度"；嵌套循环调用链示例改为"node/task/simulate"；惩罚函数示例改为"t_start/t_end"；资源分配示例改为"节点A/B/C"；"轨迹图"→"场景图"；"投放时刻×延迟"→"时间点×参数值"；技能文档同步泛化
- 涉及: coding.py（6处）, quality.py（1处）, 工作流程.md（1处）, 章节模板.md（1处）

### 2026-08-07: 代码执行失败 → 论文幻觉数据
- 问题: 代码崩溃后，论文手编造了所有数值
- 修复: 添加 `code_exec_success` 状态，P2/W1/W2 门禁在代码失败时自动 FAIL
- 修复: 论文手收到"严重警告"标记，禁止编造数值
- 涉及: state.py, graph.py, writing.py

### 2026-08-07: 防御性编程缺失
- 问题: NumPy 广播错误导致代码崩溃（`td**2 * np.array([0,0,1])`）
- 修复: 在 coding.py 系统提示词中添加"防御性编程规范"章节
- 涉及: coding.py

### 2026-08-07: 配置硬编码
- 问题: 图表数量、算法数量等硬编码在提示词中
- 修复: 添加 `min_figure_count`, `min_algorithm_count` 等可配置参数
- 涉及: config.py, coding.py

### 2026-08-07: 异常静默吞没
- 问题: `except Exception: pass` 导致调试困难
- 修复: 改为 `except Exception as e: output += f"[修复异常: {e}]"`
- 涉及: graph.py (code_exec_node)

### 2026-08-07: verifier.py 死代码
- 问题: `ast.Num` 在 Python 3.8+ 中已废弃，对应分支永不执行
- 修复: 移除 `elif isinstance(node.value, (ast.Num,))` 分支
- 涉及: tools/verifier.py

### 2026-08-07: config.py 空环境变量崩溃
- 问题: `float(os.getenv("TEMPERATURE", "0.1"))` 在环境变量为空字符串时崩溃
- 修复: `float(os.getenv("TEMPERATURE", "0.1") or "0.1")`
- 涉及: config.py

### 2026-08-08: 代码执行超时（120s → 300s）
- 问题: 4 层嵌套循环 + 700 时间步 + 6144 组合导致 120s 内无法完成
- 修复: 超时时间从 120s 增加到 300s
- 涉及: config.py

### 2026-08-08: 超时诊断信息缺失
- 问题: 超时时只输出简单的"超时"消息，无法定位性能瓶颈
- 修复: 超时时自动分析代码特征（嵌套循环层数、时间步长、差分进化参数），给出针对性修复建议
- 涉及: graph.py (code_exec_node)

### 2026-08-08: 计算效率规范缺失
- 问题: coding.py 提示词缺少计算效率约束，LLM 仍生成 4 层嵌套循环
- 修复: 添加"计算效率规范"章节（禁止 3+ 层嵌套循环、时间步长 ≥0.5s、差分进化参数限制等）
- 涉及: coding.py

### 2026-08-08: 误差分析在代码失败时生成幻觉数据
- 问题: 代码超时后 error_analysis_node 仍生成详细的蒙特卡洛/敏感性分析数值
- 修复: 在 error_analysis_node 开头检查 code_exec_success，失败时输出诊断建议而非幻觉数据
- 涉及: graph.py (error_analysis_node)

### 2026-08-08: 论文中"待计算"占位符
- 问题: writing_full_node 在代码失败时传入"请在论文中用'待计算'标注"的指令
- 修复: 改为"不得在结果表中填入任何数值（包括'待计算'），直接留空或标注'见代码输出'"
- 涉及: graph.py (writing_w1_node, writing_full_node), writing.py (write_paper)

## 常见问题排查指南

### 代码执行失败
1. 检查 `projects/output/代码执行结果.txt` 中的错误类型
2. 常见错误:
   - `ValueError: operands could not be broadcast` → NumPy 广播问题，在 coding.py 提示词中强调 np.outer()
   - `SyntaxError` → LLM 截断输出，增加 max_tokens
   - `TimeoutExpired` → 代码死循环或计算量过大
   - `ModuleNotFoundError` → 环境缺少依赖

### 论文数值不可信
1. 先检查 `代码执行结果.txt` 是否成功
2. 如果失败，检查 `diagnose.py` 报告的幻觉检测
3. 修复代码后重新运行，不要在代码失败时生成论文

### 图表不足
1. 检查 `config.min_figure_count` 配置
2. 检查 coding.py 提示词中的图表要求
3. 检查代码中是否包含 `plt.savefig()` 或 `fig.savefig()`

## 修改项目时的注意事项
- graph.py 中的所有节点函数都是闭包，通过 `create_workflow()` 创建
- 新增状态字段需要同时在 state.py 和 init_node 中添加
- 质量门禁的返回值通过 `_parse_quality_status()` 解析
- 修改 coding.py 提示词时注意 {project_root} 是占位符，会被 .replace() 替换
- 运行 `python scripts/diagnose.py` 验证修改效果