from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, Mapping, Optional
import uuid


def _normalize_parent_ids(values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        text = values.strip()
        return (text,) if text else ()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result.append(text)
    return tuple(result)


@dataclass
class RunContext:
    run_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    task_id: str = ""
    step_id: int = 0
    tool_name: str = ""
    parent_record_ids: tuple[str, ...] = field(default_factory=tuple)
    selection_stage: str = ""
    checkpoint_name: str = ""
    phase: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Optional[Any] = None, **updates: Any) -> "RunContext":
        if mapping is None:
            data: Dict[str, Any] = {}
        elif isinstance(mapping, RunContext):
            data = mapping.to_dict()
        elif isinstance(mapping, Mapping):
            data = dict(mapping)
        else:
            data = {}

        data.update(updates)
        known = {
            "run_id",
            "agent_id",
            "agent_name",
            "task_id",
            "step_id",
            "tool_name",
            "parent_record_ids",
            "selection_stage",
            "checkpoint_name",
            "phase",
            "extra",
        }
        extra = dict(data.get("extra", {}) or {})
        for key, value in data.items():
            if key not in known:
                extra[key] = value
        return cls(
            run_id=str(data.get("run_id", "") or ""),
            agent_id=str(data.get("agent_id", "") or ""),
            agent_name=str(data.get("agent_name", "") or ""),
            task_id=str(data.get("task_id", "") or ""),
            step_id=int(data.get("step_id", 0) or 0),
            tool_name=str(data.get("tool_name", "") or ""),
            parent_record_ids=_normalize_parent_ids(data.get("parent_record_ids", ())),
            selection_stage=str(data.get("selection_stage", "") or ""),
            checkpoint_name=str(data.get("checkpoint_name", "") or ""),
            phase=str(data.get("phase", "") or ""),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "parent_record_ids": list(self.parent_record_ids),
            "selection_stage": self.selection_stage,
            "checkpoint_name": self.checkpoint_name,
            "phase": self.phase,
            "extra": dict(self.extra),
        }

    def with_updates(self, **updates: Any) -> "RunContext":
        data = self.to_dict()
        data.update(updates)
        return RunContext.from_mapping(data)

    def next_step(
        self,
        *,
        tool_name: Optional[str] = None,
        task_id: Optional[str] = None,
        parent_record_ids: Optional[Any] = None,
        phase: Optional[str] = None,
        selection_stage: Optional[str] = None,
        **extra: Any,
    ) -> "RunContext":
        merged_extra = dict(self.extra)
        merged_extra.update(extra)
        return replace(
            self,
            step_id=self.step_id + 1,
            tool_name=str(tool_name if tool_name is not None else self.tool_name or ""),
            task_id=str(task_id if task_id is not None else self.task_id or ""),
            parent_record_ids=_normalize_parent_ids(
                parent_record_ids if parent_record_ids is not None else self.parent_record_ids
            ),
            phase=str(phase if phase is not None else self.phase or ""),
            selection_stage=str(selection_stage if selection_stage is not None else self.selection_stage or ""),
            extra=merged_extra,
        )

    def provenance_event(self, event_type: str, **details: Any) -> Dict[str, Any]:
        event = {
            "event_type": str(event_type),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "parent_record_ids": list(self.parent_record_ids),
            "selection_stage": self.selection_stage,
            "checkpoint_name": self.checkpoint_name,
            "phase": self.phase,
        }
        if self.extra:
            event["context_extra"] = dict(self.extra)
        for key, value in details.items():
            if value is not None:
                event[key] = value
        return event


_current_run_context: ContextVar[RunContext] = ContextVar("run_context", default=RunContext())


def get_run_context() -> RunContext:
    return _current_run_context.get()


def set_run_context(context: Optional[Any] = None, **updates: Any) -> RunContext:
    new_context = RunContext.from_mapping(context, **updates)
    _current_run_context.set(new_context)
    return new_context


def update_run_context(**updates: Any) -> RunContext:
    current = get_run_context()
    new_context = current.with_updates(**updates)
    _current_run_context.set(new_context)
    return new_context


@contextmanager
def run_context_scope(context: Optional[Any] = None, **updates: Any):
    new_context = RunContext.from_mapping(context, **updates)
    token = _current_run_context.set(new_context)
    try:
        yield new_context
    finally:
        _current_run_context.reset(token)


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
