# 🎯 数学建模智能助手 (Math Model Agent)

<div align="center">

**基于 LangChain + LangGraph + Streamlit 的智能数学建模工作流系统**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)

</div>

---

## 📖 项目简介

**数学建模智能助手** 是一个基于大语言模型（LLM）的智能体系统，专为数学建模竞赛和数学建模项目设计。它将完整的数学建模工作流程自动化，分为 **建模分析 → 代码实现 → 论文撰写** 三个阶段，每个阶段都有独立的质量门禁（Subagent 质检），确保输出质量。

系统集成了 [math-modeling-skill](https://github.com/XiaoMaColtAI/math-modeling-skill) 知识库，包含建模手、编程手、论文手三个角色的专业知识、算法索引和工具链。

### ✨ 核心能力

| 能力 | 说明 |
|------|------|
| 🧠 **智能建模分析** | 自动读题、拆分子问题、选择模型、设计求解方案 |
| 💻 **自动代码生成** | 生成可运行的 Python 代码，含语法自动检查和修复 |
| ▶️ **代码自动执行** | 在隔离环境执行代码，捕获输出和结果 |
| 📊 **结果可视化** | 自动生成出版级图表（PNG/SVG），支持中文显示 |
| 📝 **论文自动撰写** | 基于真实结果生成完整论文，结构完整 |
| 🛡️ **质量门禁系统** | 五道独立质检关卡（M1/P1/P2/W1/W2），每道独立重试 |
| 🔄 **灵活工作模式** | 完整流程 / 单阶段执行 / 自由对话 |
| 🔧 **代码自动修复** | 多层修复：跨行字符串合并、全局声明修复、空块移除、截断回退、中文注入 |
| 🎨 **中文图表输出** | 自动检测并注入中文字体配置，确保图表标签正确渲染 |
| 🧩 **多问题类型支持** | 微分方程、优化问题、统计分析、物理仿真、金融经济等 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 题目上传  │  │ 题目描述  │  │ 模式选择/执行    │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              LangGraph 工作流引擎                     │
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │
│  │ 建模分析  │→│ 代码生成  │→│ 代码执行  │→│ 论文撰写│ │
│  │ modeling │  │ coding   │  │ exec    │  │ writing│ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └───┬────┘ │
│       │            │            │            │       │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐ │
│  │ M1 质检  │  │P1/P2质检│  │ 图审计   │  │W1/W2质检│ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │       代码修复管道 (Code Repair Pipeline)     │   │
│  │  跨行合并 → 括号修复 → 全局声明 → 迭代修复 → │   │
│  │  截断回退 → 空块移除 → 中文注入 → main块注入 │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              LLM 服务层 (DeepSeek / OpenAI)          │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ 建模手    │  │ 编程手    │  │ 论文手    │           │
│  │ (Agent)  │  │ (Agent)  │  │ (Agent)  │           │
│  └──────────┘  └──────────┘  └──────────┘           │
└──────────────────────────────────────────────────────┘
```

### 工作流流程

```
题目输入 → 建模分析 → M1质检 → 代码生成(P1) → P1质检 →
全量代码(P2) → 代码执行 → P2质检 → 证据大纲(W1) → W1质检 →
论文撰写(W2) → W2质检 → 完成交付
```

每个阶段质检失败会自动触发重试（最多 3 次），使用质反馈进行针对性修复，确保输出质量。

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- 一个 LLM API Key（DeepSeek 或 OpenAI）

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/hazy12343/math_model_agent.git
cd math_model_agent
```

2. **创建虚拟环境**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **配置环境变量**

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
# Windows
copy .env.example .env
# Linux/Mac
cp .env.example .env
```

编辑 `.env` 文件：

```ini
# 选择 LLM 提供商：deepseek / openai
LLM_PROVIDER=deepseek

# DeepSeek 配置
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 模型选择
LLM_MODEL=deepseek-chat

# 竞赛配置
COMPETITION=cumcm
LANGUAGE=chinese
TEMPERATURE=0.1
```

5. **启动应用**

```bash
streamlit run app.py
```

### 使用方式

1. **上传题目文件**：支持 PDF、Excel、CSV、TXT、DOCX 等格式
2. **输入题目描述**：粘贴题目文本
3. **选择工作模式**：
   - 🔄 **完整流程**：自动完成建模→编程→论文全流程
   - 🧠 **仅建模分析**：只做题目分析和模型设计
   - 💻 **仅编程实现**：只生成和执行代码
   - 📝 **仅撰写论文**：只撰写论文
4. **点击执行**，系统自动完成对应流程
5. **查看结果**：在交付物面板下载所有输出文件

---

## 📁 项目结构

```
math-model-agent/
├── app.py                     # Streamlit 应用主入口
├── requirements.txt           # Python 依赖清单
├── .env.example               # 环境变量模板
├── .gitignore                 # Git 忽略规则
│
├── src/                       # 核心源码
│   ├── __init__.py
│   ├── config.py              # 应用配置管理
│   ├── state.py               # 工作流状态定义
│   ├── graph.py               # LangGraph 工作流引擎
│   │                          # 包含代码提取、修复管道、执行、质检逻辑
│   │
│   ├── agents/                # 多智能体模块
│   │   ├── base.py            # 基础 Agent 类（LLM 交互、文件读取）
│   │   ├── modeling.py        # 建模手 Agent - 题目分析与模型设计
│   │   ├── coding.py          # 编程手 Agent - 代码生成
│   │   ├── writing.py         # 论文手 Agent - 论文撰写
│   │   └── quality.py         # 质检 Subagent - 质量门禁检查
│   │
│   ├── tools/                 # 工具模块
│   │   ├── file_reader.py     # 文件读取工具（PDF/Excel/Text）
│   │   ├── paper_search.py    # 学术论文搜索工具
│   │   └── skill_loader.py    # 技能库加载器
│   │
│   └── ui/                    # 前端界面
│       ├── chat.py            # 聊天界面和步骤渲染
│       ├── components.py      # UI 组件（进度条、交付物、下载）
│       └── sidebar.py         # 侧边栏配置
│
└── math-modeling-skill/       # 数学建模技能知识库
    ├── SKILL.md               # 技能主文件
    ├── references/            # 角色知识库
    │   ├── roles/
    │   │   ├── 建模手/        # 建模手知识
    │   │   ├── 编程手/        # 编程手知识
    │   │   └── 论文手/        # 论文手知识
    │   ├── Subagent调度.md    # 质检调度协议
    │   └── 算法索引.md        # 算法索引
    ├── assets/                # 算法说明文档
    ├── tools/                 # 工具链（LaTeX/DOCX/PDF/论文搜索/XLSX）
    └── tests/                 # 技能测试
```

---

## 🔧 核心技术

### 1. 多智能体协作

| Agent | 角色 | 职责 | 输出 |
|-------|------|------|------|
| **建模手** | 建模分析师 | 理解题目、拆分子问题、选择模型 | `题目分析报告.md`、`术语表格.md` |
| **编程手** | 算法工程师 | 编写代码、生成结果和图表 | `.py` 代码、结果表格、图表 |
| **论文手** | 学术写手 | 撰写论文 | `完整论文.md`、证据大纲 |
| **质检** | 独立审查员 | 五道门禁检查 | 质量回执（PASS/FAIL） |

### 2. 代码自动修复管道

LLM 生成的代码可能因上下文窗口截断导致语法错误，系统内置了多层修复管道，按顺序执行：

| 步骤 | 修复器 | 说明 |
|------|--------|------|
| 1 | `_merge_broken_strings` | 合并跨行字符串，自动关闭未闭合的三引号 |
| 2 | `_fix_last_line_brackets` | 移除末尾括号/花括号不完整的截断行 |
| 3 | `_repair_global_declaration` | 将 `global` 声明移动至函数体顶部 |
| 4 | `_iterative_repair` | 迭代修复：最多 15 轮，循环执行全局声明修复→截断回退→空块移除 |
| 5 | `_remove_empty_functions` | 检测并移除空函数体、空循环、空条件块 |
| 6 | `_truncate_to_valid` | 逐行删除末尾代码直到语法有效，最多删除 300 行 |
| 7 | `_inject_chinese_font` | 自动检测中文内容，注入 matplotlib 中文字体配置 |
| 8 | `_ensure_main_block` | 注入 `if __name__ == "__main__"` 入口，包裹顶层代码 |

### 3. 中文图表显示

系统自动检测代码中的中文内容，并注入 matplotlib 字体配置：

- 覆盖 8 个 CJK Unicode 区块（基本汉字 + 扩展 A/B/C/D/E/F + 兼容汉字）
- 支持中文标点符号检测
- 字体回退链：`Microsoft YaHei → SimHei → DejaVu Sans`
- 自动设置 `axes.unicode_minus = False` 防止负号乱码
- SVG 文本保持可编辑（`svg.fonttype = 'none'`）

### 4. 五道质量门禁

| 门禁 | 阶段 | 检查内容 |
|------|------|----------|
| **M1** | 建模终检 | 子问题覆盖、假设依据、公式一致性、模型可实现性 |
| **P1** | 最小可运行 | 代码可执行性、退出码、输入输出追溯 |
| **P2** | 编程终检 | 代码完整性、结果正确性、图表质量、中文标签检查 |
| **W1** | 证据大纲 | 结论与证据路径、数值一致性、图表引用 |
| **W2** | 论文终检 | 规则合规、主张-证据一致、格式规范 |

---

## 🌐 支持的数学建模问题类型

系统经过测试验证，可处理以下类型的问题：

| 类型 | 示例 | 核心方法 |
|------|------|---------|
| 🦠 **微分方程/生物数学** | 传染病模型（SIR） | ODE 求解、参数拟合、敏感性分析 |
| 📦 **优化/运筹学** | 物流配送中心选址 | Weber 问题、K-means、成本分析 |
| 💰 **金融经济** | 投资组合优化 | Markowitz 均值-方差、有效前沿 |
| ⚙️ **物理工程** | 弹簧-质量-阻尼系统 | 二阶 ODE、模态分析、频率响应 |
| 📈 **统计预测** | 时间序列预测 | 回归分析、数据拟合 |
| 🗺️ **图论/网络** | 最短路径、网络流 | 图算法 |

---

## ⚙️ 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供商：`deepseek` 或 `openai` | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | - |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `LLM_MODEL` | 模型名称 | `deepseek-chat` |
| `COMPETITION` | 竞赛类型 | `cumcm` |
| `LANGUAGE` | 语言 | `chinese` |
| `TEMPERATURE` | 生成温度 (0.0-1.0) | `0.1` |

### 应用配置

在 `src/config.py` 中可调整：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_tokens` | LLM 最大输出 Token | 16384 |
| `max_retries` | 质检失败最大重试次数 | 3 |
| `code_exec_timeout` | 代码执行超时秒数 | 120 |

---

## 📋 输出文件

完整流程运行后，在 `projects/output/` 目录生成以下文件：

| 文件 | 说明 |
|------|------|
| `题目分析报告.md` | 问题分析、模型选择、求解方案 |
| `术语表格.md` | 符号、含义、单位对照表 |
| `solution_full.py` | 完整可运行 Python 代码（含中文图表） |
| `代码执行结果.txt` | 代码运行输出 |
| `证据大纲.md` | Claim-Evidence 映射 |
| `完整论文.md` | 完整论文正文 |
| 图表文件 | PNG/SVG 格式的结果图（中文标签） |

---

## ⚠️ 注意事项

- **API Key 安全**：`.env` 文件包含你的 API Key，已加入 `.gitignore` 不会上传
- **论文仅供参考**：生成的论文内容仅供参考，提交竞赛前请仔细核对
- **代码结果验证**：建议对关键数值结果进行人工复核
- **网络要求**：使用 LLM 服务需要稳定的网络连接
- **中文字体**：图表中文显示需要系统安装中文字体（Windows 默认支持，Linux 需安装 `fonts-wqy-microhei`）

---

## 📄 许可证

本项目基于 [math-modeling-skill](https://github.com/XiaoMaColtAI/math-modeling-skill) 构建，仅供学习参考。

---

## 🙏 致谢

- [math-modeling-skill](https://github.com/XiaoMaColtAI/math-modeling-skill) - 数学建模技能知识库
- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用框架
- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流引擎
- [Streamlit](https://streamlit.io) - Web 界面框架