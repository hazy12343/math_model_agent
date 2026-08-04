import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent
SKILL_ROOT = PROJECT_ROOT / "math-modeling-skill"


@dataclass
class AppConfig:
    skill_root: Path = SKILL_ROOT
    project_root: Path = PROJECT_ROOT / "projects"
    competition: str = "cumcm"
    language: str = "chinese"
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 8192
    max_retries: int = 3
    code_exec_timeout: int = 120
    subagents: dict = field(default_factory=lambda: {
        "official_rules": False,
        "attachment_inventory": False,
        "literature_survey": False,
        "algorithm_prototype": False,
        "independent_experiment": False,
        "dual_language": False,
        "terminology_check": False,
    })

    @classmethod
    def from_env(cls) -> "AppConfig":
        provider = os.getenv("LLM_PROVIDER", "deepseek")
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            model = os.getenv("LLM_MODEL", "deepseek-chat")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            model = os.getenv("LLM_MODEL", "gpt-4o")

        return cls(
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key,
            llm_base_url=base_url,
            competition=os.getenv("COMPETITION", "cumcm"),
            language=os.getenv("LANGUAGE", "chinese"),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )

    def ensure_project_root(self) -> Path:
        self.project_root.mkdir(parents=True, exist_ok=True)
        (self.project_root / "data").mkdir(exist_ok=True)
        (self.project_root / "results").mkdir(exist_ok=True)
        (self.project_root / "figures").mkdir(exist_ok=True)
        return self.project_root