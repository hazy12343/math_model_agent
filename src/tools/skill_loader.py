from pathlib import Path
from typing import Optional


class SkillLoader:
    def __init__(self, skill_root: Path):
        self.skill_root = Path(skill_root)

    def load_skill(self, relative_path: str) -> str:
        path = self.skill_root / relative_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")

    def load_reference(self, relative_path: str) -> str:
        return self.load_skill(relative_path)

    def load_algorithm(self, filename: str) -> str:
        return self.load_skill(f"assets/{filename}")

    def load_tool_skill(self, tool_name: str) -> str:
        return self.load_skill(f"tools/{tool_name}/SKILL.md")

    def load_algorithm_index(self) -> str:
        return self.load_skill("references/算法索引.md")

    def load_subagent_protocol(self) -> str:
        return self.load_skill("references/Subagent调度.md")

    def get_all_algorithms(self) -> dict:
        return {
            "优化": "01-优化算法说明.md",
            "预测": "02-预测类算法说明.md",
            "评价": "03-评价类算法说明.md",
            "图论": "04-图论与网络分析算法说明.md",
            "统计": "05-统计分析与数据处理算法说明.md",
            "综合": "06-综合类算法说明.md",
            "机器学习": "07-机器学习算法说明.md",
        }