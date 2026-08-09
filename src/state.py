from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class WorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    current_stage: str
    stage_history: List[str]

    quality_gates: Dict[str, str]

    project_root: str
    skill_root: str
    competition: str
    language: str

    problem_description: str
    problem_files: List[str]
    attachment_hashes: Dict[str, str]

    modeling_report: Optional[str]
    terminology_table: Optional[str]

    code_files: List[str]
    result_files: List[str]
    figure_files: List[str]
    reproducibility_manifest: Optional[str]
    code_exec_output: Optional[str]
    raw_exec_output: Optional[str]
    verification_output: Optional[str]

    code_exec_success: bool
    exec_error: Optional[str]

    evidence_outline: Optional[str]
    paper_output: Optional[str]

    error_analysis: Optional[str]
    model_comparison: Optional[str]
    polished_paper: Optional[str]

    subagent_config: Dict[str, bool]

    error: Optional[str]
    retry_counts: Dict[str, int]
    stage_output: Optional[str]

    user_input: Optional[str]
    uploaded_files: List[str]