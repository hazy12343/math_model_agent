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
```

## 关键状态字段
- `code_exec_success: bool` — 代码是否执行成功（决定后续门禁）
- `exec_error: str` — 执行失败时的错误信息
- `quality_gates: dict` — 各阶段质量门禁状态

## 质量门禁
| 门禁 | 阶段 | 失败时行为 |
|------|------|------------|
| M1 | 建模终检 | 重试建模（最多3轮） |
| P1 | 最小代码门禁 | 重试代码生成 |
| P2 | 编程终检 | 代码执行失败时自动FAIL，重试代码生成 |
| W1 | 证据大纲门禁 | 代码执行失败时自动FAIL |
| W2 | 论文终检 | 代码执行失败时自动FAIL |

## 已修复问题记录

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