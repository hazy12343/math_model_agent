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

### 2026-08-14: 代码审查 — 13 处 Bug 修复
- 问题1: `_build_initial_state()` 缺少 `code_exec_success` 和 `exec_error` 字段，单阶段工作流状态不完整
- 修复: app.py 新增两个字段初始值
- 问题2/3: `_check_result_plausibility` 零值检测正则是问题特定的（`总[遮蔽时]`），且非零值正则需要小数点，遗漏整数结果
- 修复: 零值检测移除问题特定关键词 `总[遮蔽时]`，扩展为 `target|objective`；非零值检测改为 `[1-9]\d*(?:\.\d+)?` 支持整数
- 涉及: graph.py
- 问题4: detector.py `detect_result_anomalies` 有相同的零值检测正则问题
- 修复: 同步修复，与 graph.py 保持一致
- 涉及: detector.py
- 问题5: verifier.py `cross_validation` NaN 否定表述过滤不完整，缺少 "nan not found"、"no NaN values"、"all values are valid" 等
- 修复: 扩展否定表述列表至 15 项
- 涉及: verifier.py, detector.py
- 问题6: `error_analysis_node` 全零检测 regex `\d+\.\d+` 只匹配浮点数，遗漏整数零
- 修复: 改为 `\d+(?:\.\d+)?` 同时匹配整数和浮点数
- 涉及: graph.py
- 问题7: `_check_result_plausibility` 警告编号重复（两个"10."）
- 修复: 重新编号为 10/11/12
- 涉及: graph.py
- 问题8: `model_comparison_node` 和 `error_analysis_node` 中 `raw_exec = state.get("raw_exec_output", exec_output)` 若值为 None 则后续 `.lower()` 崩溃
- 修复: 改为 `state.get("raw_exec_output") or exec_output`
- 涉及: graph.py
- 问题9: `model_comparison_node` 对比表格硬编码 "s" 单位
- 修复: 移除单位后缀，仅输出数值
- 涉及: graph.py
- 问题10: `code_exec_node` PSO 迭代检测使用 `for...range(N)` 匹配任意循环，且 `pso_iters_match` 结果未被使用
- 修复: 改为匹配变量名 `n_iterations|max_iter|pso_iters|iterations|n_iter`，移除通用 range 回退
- 涉及: graph.py
- 问题11: coding.py 使用中文变量名 `图分类`
- 修复: 改为 `figure_category_count`
- 涉及: coding.py
- 问题12: diagnose.py 算法检测仅检查中文名，遗漏 DE/PSO/GA/SA/ACO/TS 等
- 修复: 扩展关键词列表至 20 项（含英文缩写和中文名）
- 涉及: diagnose.py
- 问题13: detector.py NaN 检测缺少否定表述过滤（之前只在 verifier.py 修复）
- 修复: 新增自检标签行跳过和否定表述过滤
- 涉及: detector.py

### 2026-08-14: 基于 projects 产物的优化 — 负值检测 + 适应度函数 + 模型对比解析
- 问题1: P2 门禁未检测到负值结果（-56170.97s），`_check_result_plausibility` 缺少负值检测
- 修复: 新增负值检测（#2），使用正则匹配 `(?:最优|best|total|...).*?[=:：]\s*-\d+`，全部负值→P0，部分负值→P1
- 涉及: graph.py
- 问题2: detector.py 负值检测仅检查关键词"负值"，不检测实际负值数字
- 修复: 新增 #3.5 实际负值数字检测，使用与 graph.py 一致的正则
- 涉及: detector.py
- 问题3: 误差分析节点在负值结果时仍生成详细数值估计（`pass  # negative values are fine`）
- 修复: 新增负值检测分支，检测到负值时输出诊断建议而非编造数据；移除"negative values are fine"注释
- 涉及: graph.py (error_analysis_node)
- 问题4: 模型对比表解析错误——"s" 单独成列（如 `| 模型A | -56170.97 | s |`）
- 修复: `_extract_comparison_table` 格式1新增数字+单位合并逻辑，将 "s" 与前面的数字合并
- 涉及: graph.py
- 问题5: 适应度函数 `-(raw - penalty)` 符号错误未被检测，导致优化器走向错误方向
- 修复: coding.py 新增"适应度函数符号设计"章节，展示正确/错误写法及原理；预执行扫描新增 `return -(var - var)` 模式检测；quality.py P2 门禁新增符号错误检测
- 涉及: coding.py, graph.py, quality.py
- 问题6: 图表仅输出 PNG 格式，缺少 SVG 矢量图
- 修复: coding.py 输出格式要求改为"同时保存 PNG 和 SVG 格式"；自检清单新增 SVG 项
- 涉及: coding.py

### 2026-08-14: 二轮产物审查 — NaN 误报 + 收敛检测 + MC std=0 + 对比表清洗
- 问题1: 误差分析误报"全部 NaN"——`[NaN/Inf] False` 行含 "NaN" 触发 `all_nan=True`，实际结果有效
- 修复1: `error_analysis_node` 和 `_check_result_plausibility` 的 NaN 检测增加自检标签排除逻辑（`[xxx检查]`/`[xxx检测]` + 否定表述 `False/false/无/未检测/not found`）
- 涉及: graph.py
- 问题2: 验证器不认"收敛代数=10"——`convergence_check` 的 `converge_state_keywords` 缺少"收敛代数"
- 修复2: 新增"收敛代数"/"收敛于"/"最终适应度"/"final fitness" 关键词
- 涉及: verifier.py
- 问题3: 蒙特卡洛 std=0.0000 未告警——所有扰动结果相同，说明扰动无效或目标函数平坦
- 修复3: `_check_result_plausibility` 新增 MC std=0 检测（正则匹配 `std=0.0`）
- 涉及: graph.py
- 问题4: 对比表"结果=0.000334"含标签前缀——`_extract_comparison_table` 格式1未剥离值前缀
- 修复4: 新增值前缀清洗逻辑（剥离"结果="/"值="/"最优值="/"耗时="/"收敛代数="等前缀）；格式2新增 `=` 分隔符支持
- 涉及: graph.py
- 问题5: 模型对比节点数值提取失败——`结果=0.000334` 使用 `=` 分隔符，正则只匹配 `[：:]`
- 修复5: `model_comparison_node` 的 5 个提取正则全部新增 `=` 分隔符选项
- 涉及: graph.py
- 问题6: 算法对比输出格式不灵活——prompt 只规定"耗时(s)"列，实际代码输出"收敛代数"
- 修复6: coding.py 对比格式说明新增"收敛代数"作为合法替代列，并给出格式示例
- 涉及: coding.py
- 问题7: MC 乘法扰动在参数≈0 时失效——`param * (1.0 ± 0.05)` 在零值附近扰动幅度趋近于零
- 修复7: coding.py MC 验证章节新增扰动方式注意事项（优先加法扰动、不同量级用不同 scale）
- 涉及: coding.py
- 问题8: 自检清单缺少 MC 扰动方式和算法对比格式检查
- 修复8: 自检清单新增 2 项——MC 扰动方式、算法对比格式
- 涉及: coding.py

### 2026-08-14: 论文公式格式规范 — \frac12 / \text / 范数 / 编号
- 问题1: `\frac12` 简写不规范——`\frac12 g\tau^2` 虽能编译但不符 LaTeX 规范，跨编译器兼容性差
- 修复1: 写作规范新增"分数必须使用完整形式 `\frac{分子}{分母}`"规则
- 问题2: `\text{...}` 需要 amsmath 宏包——`\mathbf P^{\text{drop}}` 可移植性差
- 修复2: 统一要求使用 `\mathrm{...}` 替代 `\text{...}`（纯数学上下文中）
- 问题3: 范数用 `||x||` 双竖线——LaTeX 中双竖线间距不正确
- 修复3: 统一要求使用 `\lVert ... \rVert` 或 `\left\| ... \right\|`
- 问题4: 全文独立公式缺少编号——无法在正文中引用"由式(3)可得..."
- 修复4: 新增"所有独立公式必须编号"规则
- 涉及: writing.py, 写作规范.md, LaTeX格式规范.md

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

### 2026-08-10: 算法检测误报 + 假验证模式检测 + 浮点比较误报
- 问题1: P1-多算法 误报"未检测到多种算法对比"——LLM 输出"双层优化(GA+PSO)"和"差分进化(DE)"，但算法关键词列表缺少 GA/PSO/DE/差分进化/双层优化等缩写
- 问题2: P1-浮点比较 误报——正则 `==\s*[\d.]+` 误匹配 `len(t_points) == 0` 等整数比较
- 问题3: 敏感性分析是假的——`change_pct = (shield - 10.0) / 10.0 * 100` 硬编码基线值 10.0
- 问题4: 收敛曲线是假的——`history = [best_shield * (i/10) for i in range(1, 11)]` 合成直线
- 问题5: 蒙特卡洛验证只覆盖部分参数——`monte_carlo_validation(best_params[:4])` 只取前4个参数
- 修复1（graph.py）: 3 处算法关键词列表扩展——新增 GA/PSO/DE/SA/ACO/TS/差分进化/双层优化/蚁群/禁忌搜索/爬山 等缩写和中文名；`_extract_comparison_table` 格式1关键词新增差分/进化/双层/两层/蚁群/禁忌
- 修复2（detector.py）: 浮点比较检测改为只匹配浮点数字面量（含小数点），排除纯整数比较
- 修复3（graph.py）: `_check_result_plausibility` 新增假敏感性分析检测（正则匹配硬编码基线值）和假收敛曲线检测（正则匹配合成直线）
- 修复4（coding.py）: 新增"禁止的验证模式"章节（V1-硬编码基线/V2-合成收敛曲线/V3-部分参数蒙特卡洛）；自检清单新增 3 项（基线值/收敛记录/参数覆盖）
- 涉及: graph.py, detector.py, coding.py

### 2026-08-11: 零值检测误报 — 部分零值误判为算法全部失败
- 问题: 3 个导弹中 M1 产出 5.00s 遮蔽，但 M2/M3 为零 → P0-算法失败 误报。当前检测逻辑是"发现任一零值即 P0"，应改为"全部零值→P0，部分零值→P1"
- 修复: graph.py `_check_result_plausibility` 和 detector.py `detect_result_anomalies` 的零值检测改为分级——同时统计零值和非零值结果数，全部零→P0，零值多于非零值→P1，多数非零→不告警
- 涉及: graph.py, detector.py

### 2026-08-11: 诊断词污染 + 零值跳过输出缺失
- 问题1: "负值检查"误报 — `[负值检查] 遮蔽时间均为正: False` 是自检标签，detector 因含"负值"关键词误判为数值异常
- 问题2: 收敛性分析误报 — `[结果合理性检测]` 的诊断警告 "P1-收敛性: 未检测到收敛性判断，**迭代**类算法..." 含"迭代"关键词，verifier 误判为"有收敛性分析但不完整"
- 问题3: 零值跳过导致输出缺失 — 代码用 `if base_shield > 0:` 跳过敏感性分析（无 CSV）、蒙特卡洛验证、收敛曲线图，导致图表数不足、无 sensitivity.csv
- 修复1（detector.py）: 负值检测跳过 `[xxx检查]`/`[xxx检测]` 格式的自检标签行
- 修复2（verifier.py）: convergence_check 过滤掉 `⚠️ P0-`/`⚠️ P1-` 诊断警告行和 `[结果合理性检测]` 节标题，仅分析代码实际输出
- 修复3（coding.py）: 新增"零值保护与输出完整性"要求——禁止 `if base > 0:` 跳过敏感性/蒙特卡洛/收敛曲线，即使结果为零也须生成全部图表和 CSV；自检清单新增零值保护项
- 涉及: detector.py, verifier.py, coding.py

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