# 🎯 数学建模智能助手 (Math Model Agent)

<div align="center">

**基于 LangChain + LangGraph + Streamlit 的智能数学建模 Agent 系统**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)

</div>

---

## 📖 项目简介

**数学建模智能助手** 是一个面向数学建模竞赛（CUMCM 国赛 / MCM-ICM 美赛）的 LLM Agent 系统。它通过 **建模手 → 编程手 → 论文手** 三角色协作，配合 **五道独立质量门禁**、**数值验证引擎** 和 **陷阱检测系统**，将完整的数学建模流程自动化，目标达到国赛级别。

核心能力矩阵：

| 能力 | 说明 | 技术创新 |
|------|------|---------|
| 🧠 **智能建模分析** | 自动读题、拆分子问题、选择模型、设计求解方案 | 模型创新引导 + 陷阱检测 |
| 💻 **自动代码生成与修复** | 生成可运行 Python 代码，8 层修复管道确保语法正确 | 跨行字符串合并、全局声明修复、截断回退 |
| ✅ **数值验证引擎** | 量纲分析、边界条件检查、交叉验证、敏感性分析 | 基于 AST 的变量提取 + 量纲推断 |
| 🪤 **陷阱检测系统** | 数据异常检测、隐式约束识别、单位陷阱、数值陷阱 | CSV 智能解析 + 中文关键字匹配 |
| 📊 **中文图表输出** | 自动检测 CJK 内容并注入 matplotlib 字体配置 | 8 个 Unicode 区块扫描 + 字体回退链 |
| 📝 **论文自动撰写** | 基于真实代码结果生成论文，支持 Claim-Evidence 映射 | 证据大纲 → 完整论文两阶段 |
| 🛡️ **五道质量门禁** | M1/P1/P2/W1/W2 独立质检，每道最多 3 次重试 | 独立 Subagent 只读质检 |
| 🔄 **灵活工作模式** | 完整流程 / 单阶段（建模/编程/论文） / 自由对话 | 统一输入 → 路由分发 |

---

## 🏗️ 系统架构

```
                            ┌──────────────────────┐
                            │    Streamlit UI       │
                            │  ┌────┐┌────┐┌────┐  │
                            │  │上传││描述││执行│  │
                            │  └────┘└────┘└────┘  │
                            └──────────┬───────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                        LangGraph 工作流引擎                                  │
│                                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ ① 建模   │──▶│ ② M1质检 │──▶│ ③ 最小代码│──▶│ ④ P1质检 │──▶│ ⑤ 全量代码│  │
│  │ modeling │   │  m1_check│   │ coding_p1│   │ p1_check│   │coding_full│  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └─────┬────┘  │
│                                                                      │       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────▼────┐  │
│  │ ⑩ 完成  │◀──│ ⑨ W2质检 │◀──│ ⑧ 完整论文│◀──│ ⑦ W1质检 │◀──│ ⑥ 代码执行│  │
│  │   done   │   │ w2_check│   │writing_ful│   │ w1_check│   │code_exec │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                  │               │          │
│                                          ┌───────┘        ┌──────┘          │
│                                    ┌─────▼─────┐   ┌─────▼─────┐            │
│                                    │ 证据大纲   │   │ 数值验证   │            │
│                                    │writing_w1 │   │  verify   │            │
│                                    └───────────┘   └───────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                         LLM 服务层 (DeepSeek / OpenAI)                       │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ 建模手 Agent  │  │ 编程手 Agent  │  │ 论文手 Agent  │  │ 质检 Subagent │    │
│  │ ModelingAgent│  │ CodingAgent  │  │ WritingAgent │  │QualityCheckAg│    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │              │
│  ┌──────▼─────────────────▼─────────────────▼─────────────────▼──────┐      │
│  │                    math-modeling-skill 知识库                      │      │
│  │  建模设计理论 │ 算法索引 │ 可视化规范 │ 写作规范 │ 章节模板 │ 质检协议 │      │
│  └───────────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 工作流详解

```
题目输入
  │
  ▼
初始化 ──▶ 建模分析 ──▶ M1建模终检
              │ 失败重试        │ 通过
              └────────────────┘
                                 ▼
                          最小可运行代码(P1) ──▶ P1门禁
                              │ 失败重试             │ 通过
                              └─────────────────────┘
                                                      ▼
                                               全量代码(P2) ──▶ 代码执行
                                                  │ 失败重试
                                                  └──────────┘
                                                       │
                                                       ▼
                                                 数值验证(verify)
                                                       │
                                                       ▼
                                                 P2编程终检
                                                       │ 通过
                                                       ▼
                                               证据大纲(W1) ──▶ W1门禁
                                                       │ 通过
                                                       ▼
                                               完整论文(W2) ──▶ W2终检
                                                       │ 通过
                                                       ▼
                                                     完成
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- DeepSeek 或 OpenAI API Key
- Windows / Linux / macOS

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/hazy12343/math_model_agent.git
cd math_model_agent

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
copy .env.example .env    # Windows
# cp .env.example .env    # Linux/Mac
```

编辑 `.env` 文件：

```ini
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
COMPETITION=cumcm
LANGUAGE=chinese
TEMPERATURE=0.1
```

### 启动

```bash
streamlit run app.py
```

### 使用三步走

1. **📁 上传题目** — 支持 PDF、Excel、CSV、TXT、DOCX、PNG/JPG
2. **📝 描述题目** — 粘贴题目文本
3. **🚀 选择模式** — 点击执行：

| 模式 | 说明 |
|------|------|
| 🔄 完整流程 | 建模 → 编程 → 论文，全自动 |
| 🧠 仅建模分析 | 只做题目分析和模型设计 |
| 💻 仅编程实现 | 只生成和执行代码 |
| 📝 仅撰写论文 | 基于已有结果撰写论文 |

---

## 📁 项目结构

```
math-model-agent/
├── app.py                          # Streamlit 应用入口
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量模板
├── .gitignore                      # Git 忽略规则
│
├── src/                            # 核心源码
│   ├── config.py                   # 应用配置（dataclass + 环境变量加载）
│   ├── state.py                    # 工作流状态定义（TypedDict）
│   ├── graph.py                    # LangGraph 工作流引擎（~1360行）
│   │   ├── 代码修复管道            # 8 层自动修复（合并字符串→括号→全局声明→迭代→截断→空块→中文→main）
│   │   ├── 代码执行节点            # 隔离执行 + 自动收集图表/结果文件
│   │   ├── 数值验证节点            # 集成 NumericalVerifier + TrapDetector
│   │   ├── 质检节点 ×5             # M1/P1/P2/W1/W2 独立门禁
│   │   └── 路由逻辑                # 条件边 + 重试控制
│   │
│   ├── agents/                     # 多智能体模块
│   │   ├── base.py                 # 基础 Agent（LLM 流式调用、文件读取、中断检测）
│   │   ├── modeling.py             # 建模手 — 题目分析、模型选择、模型创新引导
│   │   ├── coding.py               # 编程手 — 代码生成、环境检查、中文图表规范
│   │   ├── writing.py              # 论文手 — 证据大纲、完整论文、论文修复
│   │   └── quality.py              # 质检 Subagent — 五道门禁独立检查
│   │
│   ├── tools/                      # 工具模块
│   │   ├── verifier.py             # 数值验证引擎（量纲分析/边界条件/交叉验证/敏感性/图表格式）
│   │   ├── detector.py             # 陷阱检测系统（数据异常/缺失值/隐式约束/单位陷阱/模型风险/数值陷阱）
│   │   ├── skill_loader.py         # 技能库加载器（SKILL.md / 参考文档 / 算法说明）
│   │   ├── file_reader.py          # 文件读取工具（PDF/Excel/Text）
│   │   └── paper_search.py         # 学术论文搜索（hybrid_scholar + OpenAlex）
│   │
│   └── ui/                         # Streamlit 前端
│       ├── chat.py                 # 三步界面 + 结果展示 + 对话模式
│       ├── components.py           # 进度条/阶段指示器/交付物/质量门禁/下载按钮
│       └── sidebar.py              # 侧边栏（API配置/模型选择/清空操作）
│
└── math-modeling-skill/            # 数学建模技能知识库（submodule）
    ├── SKILL.md                    # 技能主入口 + 强制执行协议
    ├── references/
    │   ├── Subagent调度.md         # 质检 Subagent 调度协议
    │   ├── 算法索引.md             # 7 类算法分类索引
    │   └── roles/
    │       ├── 建模手/              # 建模设计理论、工作流程、常见模式、质检清单、前置合同
    │       ├── 编程手/              # 可视化规范、工作流程、常见模式、MATLAB规范、质检清单
    │       └── 论文手/              # 章节模板、写作规范、工作流程、LaTeX格式、英文化工作流
    ├── assets/                     # 7 类算法详细说明文档
    └── tools/                      # 工具链（DOCX/LaTeX/PDF/论文搜索/XLSX）
```

---

## 🔧 核心技术详解

### 1. 四角色协作

| 角色 | 类 | 职责 |
|------|-----|------|
| **建模手** | `ModelingAgent` | 读题→拆子问题→选模型→输出报告，要求每子问题≥1个创新改进点 |
| **编程手** | `CodingAgent` | P1最小实现→P2全量计算，生成图表和结果文件，强制中文标签 |
| **论文手** | `WritingAgent` | W1证据大纲（Claim-Evidence映射）→W2完整论文 |
| **质检员** | `QualityCheckAgent` | 独立只读质检，不参与产物编写，返回 PASS/FAIL/BLOCKED |

每个 Agent 都从 `math-modeling-skill` 知识库加载角色专属的 SKILL.md、工作流程、参考文档和算法说明。

### 2. 代码修复管道（8 层）

LLM 生成的代码可能因上下文窗口截断导致语法错误。系统内置 8 层修复管道，按优先级执行：

| 层 | 修复器 | 职责 |
|----|--------|------|
| 1 | `_merge_broken_strings` | 合并跨行字符串，自动闭合三引号 |
| 2 | `_fix_last_line_brackets` | 移除末尾不完整的括号/花括号行 |
| 3 | `_repair_global_declaration` | 将 `global` 声明移至函数体顶部 |
| 4 | `_iterative_repair` | 最多 15 轮迭代修复（全局声明→截断→空块） |
| 5 | `_remove_empty_functions` | 移除空函数体/空循环/空条件块 |
| 6 | `_truncate_to_valid` | 逐行删除末尾直到语法正确（最多 300 行） |
| 7 | `_inject_chinese_font` | 检测 CJK 内容→注入 matplotlib 中文字体配置 |
| 8 | `_ensure_main_block` | 注入 `if __name__ == "__main__"` 入口 |

### 3. 数值验证引擎

`NumericalVerifier` 在代码执行后自动运行四项验证：

| 验证项 | 方法 | 关键逻辑 |
|--------|------|---------|
| **量纲分析** | `dimensional_analysis()` | AST 提取变量赋值 → 根据变量名推断量纲（speed→m/s, time→s 等） → 与预期单位对比 |
| **边界条件** | `boundary_condition_check()` | 从题目提取约束条件（"不能超过X"、"必须等于Y"）→ 在结果中精确匹配 |
| **交叉验证** | `cross_validation()` | 检测 NaN/Inf/None 异常值、数值范围跨度过大（>1e6）、负值警告 |
| **敏感性分析** | `sensitivity_check()` | 检查是否包含敏感性分析关键词和 CSV 输出 |

### 4. 陷阱检测系统

`TrapDetector` 在建模阶段和代码阶段分别检测：

| 检测阶段 | 检测内容 | 方法 |
|---------|---------|------|
| **数据异常** | CSV 列数不一致、常量列、缺失值 | `csv.reader` 解析 + 统计检查 |
| **隐式约束** | 物理限制、整数约束、单调性、对称性 | 中文正则匹配 |
| **单位陷阱** | 非标准单位（km/cm/mm）、非秒时间单位 | 正则提取 + 警告 |
| **模型风险** | 线性假设、正态假设、K-means 的 K 值、神经网络可解释性、蒙特卡洛收敛性 | 模型名匹配 |
| **数值陷阱** | 浮点 `==` 比较、硬编码容差、大循环性能、循环中 append | 正则匹配 |

### 5. 中文图表显示

`_inject_chinese_font()` 的实现细节：

- 扫描 8 个 CJK Unicode 区块（基本汉字 + 扩展 A~F + 兼容汉字）
- 支持中文标点符号检测（，。！？；：""''（）【】《》…—～·）
- 字体回退链：`Microsoft YaHei → SimHei → DejaVu Sans`
- 自动设置 `axes.unicode_minus = False` 防止负号乱码
- SVG 文本保持可编辑（`svg.fonttype = 'none'`）

### 6. 五道质量门禁

| 门禁 | 阶段 | 检查要点 |
|------|------|---------|
| **M1** | 建模终检 | 子问题覆盖、假设依据、公式自洽、单位约束、模型可实现性、验证方案、模型创新 |
| **P1** | 最小可运行 | 代码可执行性、退出码、输入输出追溯、单位数值范围、关键约束满足 |
| **P2** | 编程终检 | 代码完整性、结果合理性、图表覆盖度、中文标签、数值验证结果 |
| **W1** | 证据大纲 | 主张-证据映射、摘要数值一致性、图表公式引用路径 |
| **W2** | 论文终检 | 规则合规、主张-证据一致、数值单位正确、图表引用、文献可追溯 |

### 7. 模型创新引导

建模手被要求对每个子问题提出至少 1 个创新改进点，创新来源包括：

- **模型简化**：巧妙的近似方法
- **算法改进**：收敛加速、数值稳定性提升
- **问题转化**：将复杂约束转化为等价形式
- **多目标权衡**：非平凡权重设计

创新点必须有明确的数学表达，在报告中用"创新点"标签标注。

---

## 🌐 支持的问题类型

| 类型 | 示例场景 | 核心方法 |
|------|---------|---------|
| 🦠 **微分方程/生物数学** | 传染病模型（SIR） | ODE 求解、参数拟合、敏感性分析 |
| 📦 **优化/运筹学** | 物流配送中心选址 | Weber 问题、K-means、成本分析 |
| 💰 **金融经济** | 投资组合优化 | Markowitz 均值-方差、有效前沿 |
| ⚙️ **物理工程** | 弹簧-质量-阻尼系统 | 二阶 ODE、模态分析、频率响应 |
| 📈 **统计预测** | 时间序列预测 | 回归分析、数据拟合 |
| 🗺️ **图论/网络** | 最短路径、网络流 | 图算法 |
| 🧬 **综合评价** | 多种指标体系 | AHP、TOPSIS、熵权法 |
| 🤖 **机器学习** | 分类/回归/聚类 | 集成学习、神经网络 |

算法知识库包含 7 大类共 50+ 种算法的详细说明文档。

---

## ⚙️ 配置参考

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | `deepseek` 或 `openai` | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | — |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `OPENAI_API_KEY` | OpenAI API Key | — |
| `OPENAI_BASE_URL` | OpenAI API 地址 | — |
| `LLM_MODEL` | 模型名称 | `deepseek-chat` |
| `COMPETITION` | 竞赛类型：`cumcm` / `mcm-icm` / `other` | `cumcm` |
| `LANGUAGE` | 语言：`chinese` / `english` | `chinese` |
| `TEMPERATURE` | 生成温度 (0.0–1.0) | `0.1` |

### AppConfig 高级参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_tokens` | LLM 最大输出 Token | 16384 |
| `max_retries` | 质检失败最大重试次数 | 3 |
| `code_exec_timeout` | 代码执行超时（秒） | 120 |
| `enable_verification` | 启用数值验证引擎 | `True` |
| `enable_trap_detection` | 启用陷阱检测 | `True` |
| `enable_innovation_guidance` | 启用模型创新引导 | `True` |
| `enable_multi_model_compare` | 启用多模型对比（实验性） | `False` |

### 可选 Subagent

| 名称 | 功能 |
|------|------|
| `attachment_inventory` | 多附件复杂文件盘点 |
| `literature_survey` | 按主张簇搜索文献 |
| `algorithm_prototype` | 验证候选算法可实现性 |
| `independent_experiment` | 并行敏感性/消融实验 |
| `dual_language` | Python/MATLAB 双语言对照 |
| `terminology_check` | 专业术语核验 |

---

## 📋 输出文件

完整流程运行后在 `projects/output/` 生成：

| 文件 | 说明 |
|------|------|
| `题目分析报告.md` | 问题分析、子问题拆解、模型选择、创新点标注 |
| `术语表格.md` | 符号、含义、单位对照表 |
| `solution_p1.py` | P1 阶段最小可运行代码 |
| `solution_full.py` | P2 阶段完整代码（含中文图表） |
| `代码执行结果.txt` | 代码运行输出 + 数值验证结果 |
| `证据大纲.md` | Claim-Evidence 映射 |
| `完整论文.md` | 完整论文正文 |

---

## ⚠️ 注意事项

- **API Key 安全**：`.env` 已加入 `.gitignore`，不会被提交到仓库
- **论文仅供参考**：生成的论文需人工核对后使用，不可直接提交竞赛
- **中文字体**：Windows 默认支持，Linux 需安装 `fonts-wqy-microhei`
- **网络要求**：使用 LLM API 需要稳定的网络连接
- **代码执行**：所有代码在临时目录中隔离执行，执行后自动清理

---

## 📄 许可证

本项目基于 [math-modeling-skill](https://github.com/XiaoMaColtAI/math-modeling-skill) 构建，仅供学习参考。

---

## 🙏 致谢

- [math-modeling-skill](https://github.com/XiaoMaColtAI/math-modeling-skill) — 数学建模技能知识库
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 应用框架
- [LangGraph](https://github.com/langchain-ai/langgraph) — 有状态工作流引擎
- [Streamlit](https://streamlit.io) — Python Web 界面框架