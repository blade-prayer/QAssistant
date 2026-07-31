from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.utils.run_context import RunContext, get_run_context, make_run_id

try:
    import dill  # type: ignore
except Exception:  # pragma: no cover - fallback for lean environments
    dill = None


@dataclass
class MemoryRecord:
    """Structured wrapper around raw agent output stored in shared memory."""

    id: str
    memory_type: str
    subtype: Optional[str]
    title: str
    content: str
    source: str
    url: str
    query: str
    semantic_key: str
    content_hash: str
    source_agent_id: Optional[str]
    task_id: Optional[str]
    tool_name: Optional[str]
    created_at: str
    updated_at: str
    quality_score: float
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


class Memory:
    """Persistent shared state for the report-generation pipeline."""

    def __init__(self, config):
        self.config = config
        self.working_dir = getattr(config, "working_dir", os.path.abspath("./outputs"))
        self.memory_dir = os.path.join(self.working_dir, "memory")
        self.memory_file = os.path.join(self.memory_dir, "memory.pkl")

        self.data: List[Any] = []
        self.data_signatures: set[str] = set()
        self.records: Dict[str, MemoryRecord] = {}
        self.record_order: List[str] = []
        self.index_by_semantic_key: Dict[str, str] = {}
        self.index_by_type: Dict[str, set[str]] = defaultdict(set)
        self.index_by_url: Dict[str, str] = {}
        self.index_by_task: Dict[str, set[str]] = defaultdict(set)
        self.index_by_agent: Dict[str, set[str]] = defaultdict(set)
        self.task_mapping: List[Dict[str, Any]] = []
        self.dependency: Dict[str, set[str]] = defaultdict(set)
        self.log: List[Dict[str, Any]] = []
        self.generated_collect_tasks: List[str] = []
        self.generated_analysis_tasks: List[str] = []
        self.selection_traces: List[Dict[str, Any]] = []
        self.run_id = self._resolve_run_id()
        self.metadata: Dict[str, Any] = {
            "schema_version": 3,
            "created_at": self._now(),
            "updated_at": self._now(),
            "run_id": self.run_id,
        }
        self._prompt_loader = None

        os.makedirs(self.memory_dir, exist_ok=True)

    def _resolve_run_id(self) -> str:
        config_run_id = getattr(self.config, "run_id", None)
        if config_run_id:
            return str(config_run_id)

        config_dict = getattr(self.config, "config", {}) or {}
        if config_dict.get("run_id"):
            return str(config_dict.get("run_id"))

        return make_run_id()

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _target_type(self) -> str:
        config_dict = getattr(self.config, "config", {}) or {}
        target_type = str(config_dict.get("target_type", "general")).strip().lower()
        if target_type == "company":
            return "financial_company"
        return target_type or "general"

    def _target_name(self) -> str:
        config_dict = getattr(self.config, "config", {}) or {}
        return str(config_dict.get("target_name", "")).strip()

    def _normalize_text_list(self, values: Optional[Iterable[Any]]) -> List[str]:
        if not values:
            return []
        if isinstance(values, str):
            text = values.strip()
            return [text] if text else []
        result: List[str] = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                result.append(text)
        return result

    def _canonical_object(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._canonical_object(val)
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [self._canonical_object(item) for item in value]
        if isinstance(value, set):
            return sorted(self._canonical_object(item) for item in value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        descriptor = {
            "class": value.__class__.__name__,
        }
        if hasattr(value, "name"):
            descriptor["name"] = getattr(value, "name")
        if hasattr(value, "title"):
            descriptor["title"] = getattr(value, "title")
        if hasattr(value, "AGENT_NAME"):
            descriptor["agent_name"] = getattr(value, "AGENT_NAME")
        if hasattr(value, "short_description"):
            descriptor["short_description"] = getattr(value, "short_description")
        if hasattr(value, "source"):
            descriptor["source"] = getattr(value, "source")
        return descriptor

    def _json_dumps(self, value: Any) -> str:
        return json.dumps(
            self._canonical_object(value),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _item_text(self, item: Any) -> str:
        if hasattr(item, "brief_str") and callable(getattr(item, "brief_str")):
            try:
                return str(item.brief_str())
            except Exception:
                pass
        if hasattr(item, "__str__"):
            try:
                return str(item)
            except Exception:
                pass
        return repr(item)

    def _item_signature(self, item: Any) -> str:
        try:
            return self._normalize_item(item).content_hash
        except Exception:
            pass

        payload: Dict[str, Any] = {
            "class": item.__class__.__name__,
            "name": getattr(item, "name", None),
            "title": getattr(item, "title", None),
            "description": getattr(item, "description", None),
            "source": getattr(item, "source", None),
            "query": getattr(item, "query", None),
        }
        if hasattr(item, "content"):
            content = getattr(item, "content")
            if isinstance(content, str):
                payload["content"] = content[:1500]
            else:
                payload["content"] = str(content)[:1500]
        if hasattr(item, "data"):
            data = getattr(item, "data")
            if isinstance(data, (str, int, float, bool)) or data is None:
                payload["data"] = data
            else:
                payload["data"] = str(data)[:1500]
        return hashlib.sha1(self._json_dumps(payload).encode("utf-8")).hexdigest()

    def _classify_item(self, item: Any) -> str:
        if isinstance(item, MemoryRecord):
            if item.memory_type == "analysis":
                return "analysis"
            if item.memory_type == "search":
                return "search"
            if item.memory_type == "document" or item.subtype == "web_page":
                return "click"
            return item.memory_type or "collect"

        class_name = item.__class__.__name__.lower()
        type_name = str(getattr(item, "type", "")).lower()
        source = str(getattr(item, "source", "")).lower()

        if "analysisresult" in class_name:
            return "analysis"
        if "clickresult" in class_name:
            return "click"
        if "searchresult" in class_name or "deepsearchresult" in class_name:
            return "search"
        if "click" in type_name:
            return "click"
        if "search" in type_name:
            return "search"
        if "deepsearch agent" in source:
            return "search"
        return "collect"

    def _safe_str(self, value: Any, max_chars: int = 10000) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()[:max_chars]
        if hasattr(value, "get_full_string") and callable(getattr(value, "get_full_string")):
            try:
                return str(value.get_full_string()).strip()[:max_chars]
            except Exception:
                pass
        try:
            return str(value).strip()[:max_chars]
        except Exception:
            return repr(value)[:max_chars]

    def _first_attr(self, item: Any, names: Iterable[str]) -> Any:
        for name in names:
            if hasattr(item, name):
                value = getattr(item, name)
                if value is not None and str(value).strip():
                    return value
        return None

    def _clean_key_part(self, value: Any, max_chars: int = 160) -> str:
        text = self._safe_str(value, max_chars=max_chars).lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_url(self, url: Any) -> str:
        raw_url = self._safe_str(url, max_chars=2000)
        if not raw_url:
            return ""

        parsed = urlsplit(raw_url)
        if not parsed.scheme or not parsed.netloc:
            return raw_url.rstrip("/")

        dropped_params = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "utm_id",
            "fbclid",
            "gclid",
            "yclid",
            "mc_cid",
            "mc_eid",
            "spm",
        }
        query_pairs = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            key_lower = key.lower()
            if key_lower.startswith("utm_") or key_lower in dropped_params:
                continue
            query_pairs.append((key, value))

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        if path != "/":
            path = path.rstrip("/")
        query = urlencode(query_pairs, doseq=True)
        return urlunsplit((scheme, netloc, path, query, ""))

    def _extract_url(self, item: Any) -> str:
        raw_url = self._first_attr(item, ("link", "url", "href", "page_url"))
        if raw_url:
            return self._normalize_url(raw_url)

        for value in (
            getattr(item, "source", None),
            getattr(item, "description", None),
            getattr(item, "data", None),
        ):
            text = self._safe_str(value, max_chars=4000)
            match = re.search(r"https?://[^\s\]\)>'\"]+", text)
            if match:
                return self._normalize_url(match.group(0).rstrip(".,;"))
        return ""

    def _record_id(self, semantic_key: str) -> str:
        digest = hashlib.sha1(semantic_key.encode("utf-8")).hexdigest()[:16]
        return f"memory_{digest}"

    def _content_hash(self, record: MemoryRecord, item: Any = None) -> str:
        payload = {
            "class": (item or record.raw).__class__.__name__ if (item or record.raw) is not None else "",
            "memory_type": record.memory_type,
            "subtype": record.subtype,
            "title": record.title,
            "content": record.content[:10000],
            "source": record.source,
            "url": record.url,
            "query": record.query,
            "data": self._safe_str(getattr(item, "data", None), max_chars=10000) if item is not None else "",
        }
        return hashlib.sha1(self._json_dumps(payload).encode("utf-8")).hexdigest()

    def _semantic_key(self, item: Any, record: MemoryRecord) -> str:
        if record.url:
            return f"url:{record.url}"

        class_name = item.__class__.__name__ if item is not None else "MemoryRecord"
        title = self._clean_key_part(record.title)
        source = self._clean_key_part(record.source)

        if record.memory_type == "analysis":
            parts = [
                self._clean_key_part(record.task_id or ""),
                title,
                self._clean_key_part(record.metadata.get("input_hash", "")),
                self._clean_key_part(record.metadata.get("model_name", "")),
                self._clean_key_part(record.source_agent_id or ""),
            ]
            key_body = ":".join(part for part in parts if part)
            return f"analysis:{key_body or record.content_hash}"

        metric = self._first_attr(item, ("metric", "indicator", "name", "title")) if item is not None else title
        period = self._first_attr(item, ("period", "date", "year", "report_date", "end_date")) if item is not None else ""
        if record.memory_type == "collect" and (metric or period or source):
            parts = [
                class_name,
                self._clean_key_part(metric),
                self._clean_key_part(period),
                source,
            ]
            return "collect:" + ":".join(part for part in parts if part)

        fallback_parts = [
            record.memory_type,
            record.subtype or "",
            class_name,
            title,
            source,
        ]
        key_body = ":".join(self._clean_key_part(part) for part in fallback_parts if self._clean_key_part(part))
        return key_body or f"content:{record.content_hash}"

    def _current_context(self) -> RunContext:
        return get_run_context()

    def _context_value(self, context: RunContext, attr: str) -> Optional[str]:
        value = getattr(context, attr, "")
        if value in ("", "N/A", None):
            return None
        return str(value)

    def _provenance_event(
        self,
        *,
        event_type: str,
        kind: str,
        record: Optional[MemoryRecord] = None,
        item: Any = None,
        source_agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = self._current_context()
        event = context.provenance_event(
            event_type,
            kind=kind,
            source_agent_id=source_agent_id or self._context_value(context, "agent_id"),
            task_id=task_id or self._context_value(context, "task_id"),
            tool_name=tool_name or self._context_value(context, "tool_name"),
            record_id=record.id if record else None,
            semantic_key=record.semantic_key if record else None,
            content_hash=record.content_hash if record else None,
            memory_type=record.memory_type if record else None,
            subtype=record.subtype if record else None,
            query=(record.query if record else getattr(item, "query", None)),
            url=(record.url if record else self._extract_url(item) if item is not None else None),
            source=(record.source if record else getattr(item, "source", None)),
        )
        if extra:
            event.update({key: value for key, value in extra.items() if value is not None})
        if not event.get("run_id"):
            event["run_id"] = self.run_id
        event["event_id"] = f"prov_{hashlib.sha1(self._json_dumps(event).encode('utf-8')).hexdigest()[:16]}"
        return event

    def _merge_event_list(self, *event_lists: Any) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw_events in event_lists:
            if not raw_events:
                continue
            events = raw_events if isinstance(raw_events, list) else [raw_events]
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_key = str(event.get("event_id") or self._json_dumps(event))
                if event_key in seen:
                    continue
                merged.append(event)
                seen.add(event_key)
        return merged

    def _apply_provenance_metadata(
        self,
        metadata: Dict[str, Any],
        event: Dict[str, Any],
        *,
        prefer_origin: bool = False,
    ) -> Dict[str, Any]:
        updated = dict(metadata or {})
        existing_provenance = updated.get("provenance", [])
        updated["provenance"] = self._merge_event_list(existing_provenance, event)

        if prefer_origin or not updated.get("origin"):
            updated["origin"] = event
        else:
            derivation = updated.get("derivation", [])
            updated["derivation"] = self._merge_event_list(derivation, event)
        return updated

    def _metadata_parent_record_ids(self, metadata: Dict[str, Any]) -> List[str]:
        parent_ids: List[str] = []
        for event in metadata.get("provenance", []) or []:
            if not isinstance(event, dict):
                continue
            for parent_id in event.get("parent_record_ids", []) or []:
                text = str(parent_id).strip()
                if text and text not in parent_ids:
                    parent_ids.append(text)
        for parent_id in metadata.get("parent_record_ids", []) or []:
            text = str(parent_id).strip()
            if text and text not in parent_ids:
                parent_ids.append(text)
        return parent_ids

    def _score_record(self, record: MemoryRecord) -> tuple[float, Dict[str, float]]:
        type_base = {
            "search": 0.30,
            "document": 0.55,
            "collect": 0.50,
            "analysis": 0.75,
        }.get(record.memory_type, 0.40)

        title_score = 0.10 if record.title else 0.0
        source_score = 0.10 if record.source else 0.0
        url_score = 0.10 if record.url else 0.0
        query_score = 0.03 if record.query else 0.0
        provenance_events = record.metadata.get("provenance", []) or []
        provenance_score = 0.10 if provenance_events else 0.0
        parent_score = 0.05 if self._metadata_parent_record_ids(record.metadata) else 0.0
        directness_score = 0.05 if record.memory_type in {"document", "collect"} and record.url else 0.0
        content_length = len(record.content or "")
        if content_length >= 5000:
            content_score = 0.25
        elif content_length >= 1000:
            content_score = 0.20
        elif content_length >= 300:
            content_score = 0.15
        elif content_length > 0:
            content_score = 0.06
        else:
            content_score = -0.15

        score = (
            type_base
            + title_score
            + source_score
            + url_score
            + query_score
            + content_score
            + provenance_score
            + parent_score
            + directness_score
        )
        if record.memory_type == "search":
            score = min(score, 0.65)
        if record.memory_type == "analysis":
            score = max(score, 0.75)
        score = max(0.0, min(1.0, round(score, 4)))
        scores = {
            "type_base": type_base,
            "title": title_score,
            "source": source_score,
            "url": url_score,
            "query": query_score,
            "content": content_score,
            "provenance": provenance_score,
            "parent_lineage": parent_score,
            "directness": directness_score,
        }
        return score, scores

    def _normalize_item(
        self,
        item: Any,
        source_agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        context = self._current_context()
        source_agent_id = source_agent_id or self._context_value(context, "agent_id")
        task_id = task_id or self._context_value(context, "task_id")
        tool_name = tool_name or self._context_value(context, "tool_name")

        if isinstance(item, MemoryRecord):
            item.metadata = self._merge_metadata(item.metadata, metadata or {})
            event = self._provenance_event(
                event_type="add_data",
                kind="derivation",
                record=item,
                source_agent_id=source_agent_id,
                task_id=task_id,
                tool_name=tool_name,
            )
            item.metadata = self._apply_provenance_metadata(item.metadata, event)
            item.quality_score, item.scores = self._score_record(item)
            return item

        legacy_type = self._classify_item(item)
        if legacy_type == "analysis":
            memory_type = "analysis"
            subtype = "analysis_result"
        elif legacy_type == "search":
            memory_type = "search"
            subtype = "web_search"
        elif legacy_type == "click":
            memory_type = "document"
            subtype = "web_page"
        else:
            memory_type = "collect"
            subtype = "tool_result" if hasattr(item, "data") else "unknown"

        title = self._safe_str(self._first_attr(item, ("title", "name")), max_chars=500)
        if not title:
            title = item.__class__.__name__
        source = self._safe_str(getattr(item, "source", ""), max_chars=1000)
        query = self._safe_str(getattr(item, "query", ""), max_chars=1000)
        url = self._extract_url(item)

        content_value = None
        if hasattr(item, "content"):
            content_value = getattr(item, "content")
        elif hasattr(item, "data"):
            content_value = getattr(item, "data")
        elif hasattr(item, "description"):
            content_value = getattr(item, "description")
        content = self._safe_str(content_value, max_chars=30000)
        description = self._safe_str(getattr(item, "description", ""), max_chars=4000)
        if not content and description:
            content = description

        now = self._now()
        record_metadata: Dict[str, Any] = dict(metadata or {})
        record_metadata.setdefault("item_class", item.__class__.__name__)
        record_metadata.setdefault("legacy_type", legacy_type)
        if description:
            record_metadata.setdefault("description", description)
        if memory_type == "search" and title:
            record_metadata.setdefault("search_title", title)

        record = MemoryRecord(
            id="",
            memory_type=memory_type,
            subtype=subtype,
            title=title,
            content=content,
            source=source,
            url=url,
            query=query,
            semantic_key="",
            content_hash="",
            source_agent_id=source_agent_id,
            task_id=task_id,
            tool_name=tool_name,
            created_at=now,
            updated_at=now,
            quality_score=0.0,
            scores={},
            metadata=record_metadata,
            raw=item,
        )
        record.content_hash = self._content_hash(record, item)
        record.semantic_key = self._semantic_key(item, record)
        record.id = self._record_id(record.semantic_key)
        parent_ids = list(context.parent_record_ids)
        for parent_id in record_metadata.get("parent_record_ids", []) or []:
            text = str(parent_id).strip()
            if text and text not in parent_ids:
                parent_ids.append(text)
        event_kind = "derivation" if memory_type == "analysis" or parent_ids else "origin"
        event = self._provenance_event(
            event_type="add_data",
            kind=event_kind,
            record=record,
            item=item,
            source_agent_id=source_agent_id,
            task_id=task_id,
            tool_name=tool_name,
            extra={"parent_record_ids": parent_ids} if parent_ids else None,
        )
        record_metadata = self._apply_provenance_metadata(
            record_metadata,
            event,
            prefer_origin=event_kind == "origin",
        )
        record.metadata = record_metadata
        record.quality_score, record.scores = self._score_record(record)
        return record

    def _coerce_record(self, raw_record: Any) -> Optional[MemoryRecord]:
        if isinstance(raw_record, MemoryRecord):
            return raw_record
        if not isinstance(raw_record, dict):
            return None

        try:
            record = MemoryRecord(
                id=str(raw_record.get("id", "")),
                memory_type=str(raw_record.get("memory_type", "collect")),
                subtype=raw_record.get("subtype"),
                title=str(raw_record.get("title", "")),
                content=str(raw_record.get("content", "")),
                source=str(raw_record.get("source", "")),
                url=str(raw_record.get("url", "")),
                query=str(raw_record.get("query", "")),
                semantic_key=str(raw_record.get("semantic_key", "")),
                content_hash=str(raw_record.get("content_hash", "")),
                source_agent_id=raw_record.get("source_agent_id"),
                task_id=raw_record.get("task_id"),
                tool_name=raw_record.get("tool_name"),
                created_at=str(raw_record.get("created_at", self._now())),
                updated_at=str(raw_record.get("updated_at", self._now())),
                quality_score=float(raw_record.get("quality_score", 0.0)),
                scores=dict(raw_record.get("scores", {}) or {}),
                metadata=dict(raw_record.get("metadata", {}) or {}),
                raw=raw_record.get("raw"),
            )
        except Exception:
            return None

        if not record.content_hash:
            record.content_hash = self._content_hash(record, record.raw)
        if not record.semantic_key:
            record.semantic_key = self._semantic_key(record.raw, record)
        if not record.id:
            record.id = self._record_id(record.semantic_key)
        if not record.quality_score:
            record.quality_score, record.scores = self._score_record(record)
        return record

    def _record_labels(self, record: MemoryRecord) -> set[str]:
        labels = {
            str(record.memory_type or "").lower(),
            str(record.subtype or "").lower(),
        }
        legacy_type = str(record.metadata.get("legacy_type", "")).lower()
        if legacy_type:
            labels.add(legacy_type)
        if record.memory_type == "document" or record.subtype == "web_page":
            labels.add("click")
        labels.discard("")
        return labels

    def _merge_metadata(self, existing: Optional[Dict[str, Any]], incoming: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(existing or {})
        for key, value in (incoming or {}).items():
            if value in (None, "", [], {}):
                continue
            if key in ("provenance", "versions", "derivation", "selection_traces"):
                current = list(merged.get(key, []) or [])
                incoming_items = value if isinstance(value, list) else [value]
                seen = {self._json_dumps(item) for item in current}
                for item in incoming_items:
                    item_key = self._json_dumps(item)
                    if item_key not in seen:
                        current.append(item)
                        seen.add(item_key)
                merged[key] = current
            elif key == "origin":
                if not merged.get("origin"):
                    merged["origin"] = value
            elif key not in merged or merged.get(key) in (None, "", [], {}):
                merged[key] = value
            elif key == "search_title":
                continue
            elif merged.get(key) != value:
                conflicts = merged.setdefault("conflicts", {})
                conflicts.setdefault(key, [])
                value_key = self._json_dumps(value)
                if value_key not in {self._json_dumps(item) for item in conflicts[key]}:
                    conflicts[key].append(value)
        return merged

    def _record_version_info(self, record: MemoryRecord) -> Dict[str, Any]:
        return {
            "content_hash": record.content_hash,
            "memory_type": record.memory_type,
            "subtype": record.subtype,
            "title": record.title,
            "source": record.source,
            "url": record.url,
            "quality_score": record.quality_score,
            "updated_at": record.updated_at,
        }

    def _merge_records(self, existing: MemoryRecord, incoming: MemoryRecord) -> MemoryRecord:
        merged_metadata = self._merge_metadata(existing.metadata, incoming.metadata)
        if existing.memory_type == "search" and existing.title:
            merged_metadata.setdefault("search_title", existing.title)
        if incoming.memory_type == "search" and incoming.title:
            merged_metadata.setdefault("search_title", incoming.title)

        if not merged_metadata.get("origin"):
            merged_metadata["origin"] = existing.metadata.get("origin") or incoming.metadata.get("origin")

        merged_metadata["provenance"] = self._merge_event_list(
            existing.metadata.get("provenance", []),
            incoming.metadata.get("provenance", []),
        )
        if merged_metadata.get("origin"):
            merged_metadata["provenance"] = self._merge_event_list(
                [merged_metadata["origin"]],
                merged_metadata["provenance"],
            )

        if existing.content_hash != incoming.content_hash:
            versions = merged_metadata.setdefault("versions", [])
            seen_hashes = {str(item.get("content_hash")) for item in versions if isinstance(item, dict)}
            for record in (existing, incoming):
                if record.content_hash not in seen_hashes:
                    versions.append(self._record_version_info(record))
                    seen_hashes.add(record.content_hash)

        prefer_incoming = incoming.quality_score > existing.quality_score
        base = incoming if prefer_incoming else existing
        merged = MemoryRecord(
            id=existing.id,
            memory_type=base.memory_type,
            subtype=base.subtype,
            title=base.title or existing.title or incoming.title,
            content=base.content or existing.content or incoming.content,
            source=base.source or existing.source or incoming.source,
            url=base.url or existing.url or incoming.url,
            query=base.query or existing.query or incoming.query,
            semantic_key=existing.semantic_key,
            content_hash=base.content_hash,
            source_agent_id=base.source_agent_id or existing.source_agent_id or incoming.source_agent_id,
            task_id=base.task_id or existing.task_id or incoming.task_id,
            tool_name=base.tool_name or existing.tool_name or incoming.tool_name,
            created_at=existing.created_at,
            updated_at=self._now(),
            quality_score=base.quality_score,
            scores=base.scores,
            metadata=merged_metadata,
            raw=base.raw,
        )
        merged.metadata["selection_traces"] = self._merge_event_list(
            existing.metadata.get("selection_traces", []),
            incoming.metadata.get("selection_traces", []),
        )
        return merged

    def _add_record_to_indexes(self, record: MemoryRecord) -> None:
        self.index_by_semantic_key[record.semantic_key] = record.id
        for label in self._record_labels(record):
            self.index_by_type[label].add(record.id)
        if record.url:
            self.index_by_url[record.url] = record.id
        if record.task_id:
            self.index_by_task[str(record.task_id)].add(record.id)
        if record.source_agent_id:
            self.index_by_agent[str(record.source_agent_id)].add(record.id)

    def _rebuild_indexes(self) -> None:
        self.index_by_semantic_key = {}
        self.index_by_type = defaultdict(set)
        self.index_by_url = {}
        self.index_by_task = defaultdict(set)
        self.index_by_agent = defaultdict(set)

        ordered_ids: List[str] = []
        seen: set[str] = set()
        for record_id in self.record_order:
            if record_id in self.records and record_id not in seen:
                ordered_ids.append(record_id)
                seen.add(record_id)
        for record_id in self.records:
            if record_id not in seen:
                ordered_ids.append(record_id)
                seen.add(record_id)
        self.record_order = ordered_ids

        self.data = []
        self.data_signatures = set()
        for record_id in self.record_order:
            record = self.records[record_id]
            self._add_record_to_indexes(record)
            if record.raw is not None:
                self.data.append(record.raw)
            self.data_signatures.add(record.semantic_key)
            self.data_signatures.add(record.content_hash)

    def _insert_or_merge_record(self, record: MemoryRecord) -> MemoryRecord:
        existing_id = self.index_by_semantic_key.get(record.semantic_key)
        if existing_id and existing_id in self.records:
            merged = self._merge_records(self.records[existing_id], record)
            self.records[existing_id] = merged
            stored = merged
        else:
            self.records[record.id] = record
            if record.id not in self.record_order:
                self.record_order.append(record.id)
            stored = record

        self._rebuild_indexes()
        self.metadata["updated_at"] = self._now()
        return stored

    def _default_checkpoint_name(self, agent_class: Any) -> str:
        agent_name = str(getattr(agent_class, "AGENT_NAME", "")).lower()
        if "report_generator" in agent_name or "report generator" in agent_name:
            return "report_latest.pkl"
        if "deepsearch" in agent_name:
            return "deepsearch_latest.pkl"
        if "outline" in agent_name:
            return "outline_latest.pkl"
        if "section" in agent_name:
            return "section_latest.pkl"
        return "latest.pkl"

    def _get_prompt_loader(self):
        if self._prompt_loader is not None:
            return self._prompt_loader

        try:
            from src.utils.prompt_loader import get_prompt_loader
        except Exception:
            return None

        try:
            self._prompt_loader = get_prompt_loader("memory", report_type=self._target_type())
        except Exception:
            self._prompt_loader = None
        return self._prompt_loader

    def _resolve_llm(self, use_llm_name: Optional[str]):
        llm_dict = getattr(self.config, "llm_dict", {}) or {}
        if not llm_dict:
            return None

        if use_llm_name and use_llm_name in llm_dict:
            return llm_dict[use_llm_name]

        for llm in llm_dict.values():
            if getattr(llm, "model_name", None) == use_llm_name:
                return llm

        return next(iter(llm_dict.values()))

    def _task_signature(self, agent_class: Any, task_input: Dict[str, Any], agent_kwargs: Dict[str, Any]) -> str:
        runtime_keys = {"memory", "config", "resume", "priority", "checkpoint_name", "agent_id"}
        sanitized_kwargs = {
            key: value
            for key, value in agent_kwargs.items()
            if key not in runtime_keys
        }
        payload = {
            "agent_class_name": getattr(agent_class, "__name__", str(agent_class)),
            "agent_name": getattr(agent_class, "AGENT_NAME", None),
            "task_input": self._canonical_object(task_input or {}),
            "agent_kwargs": self._canonical_object(sanitized_kwargs),
        }
        return hashlib.sha1(self._json_dumps(payload).encode("utf-8")).hexdigest()

    def _task_summary(self, task_input: Dict[str, Any]) -> str:
        if not task_input:
            return ""
        nested_input = task_input.get("input_data")
        if isinstance(nested_input, dict):
            for key in ("analysis_task", "task", "query", "task_content"):
                value = nested_input.get(key)
                if value:
                    return str(value)
        for key in ("task", "analysis_task", "query", "task_content"):
            value = task_input.get(key)
            if value:
                return str(value)
        return self._json_dumps(task_input)[:200]

    def _upsert_task_mapping(self, entry: Dict[str, Any]) -> None:
        for idx, existing in enumerate(self.task_mapping):
            if existing.get("task_key") == entry.get("task_key"):
                merged = dict(existing)
                merged.update(entry)
                merged["updated_at"] = self._now()
                self.task_mapping[idx] = merged
                return
        self.task_mapping.append(entry)

    def _serialize_dependency(self) -> Dict[str, List[str]]:
        return {parent: sorted(children) for parent, children in self.dependency.items()}

    def _deserialize_dependency(self, raw_dependency: Any) -> Dict[str, set[str]]:
        dependency: Dict[str, set[str]] = defaultdict(set)
        if isinstance(raw_dependency, dict):
            for parent, children in raw_dependency.items():
                if isinstance(children, (list, tuple, set)):
                    dependency[str(parent)].update(str(child) for child in children)
                elif children is not None:
                    dependency[str(parent)].add(str(children))
        return dependency

    def _snapshot(self) -> Dict[str, Any]:
        self._rebuild_indexes()
        updated_at = self._now()
        self.metadata["schema_version"] = 3
        self.metadata["run_id"] = self.run_id
        self.metadata["updated_at"] = updated_at
        return {
            "schema_version": 3,
            "created_at": self.metadata.get("created_at", self._now()),
            "updated_at": updated_at,
            "metadata": dict(self.metadata),
            "records": self.records,
            "record_order": list(self.record_order),
            "selection_traces": self.selection_traces,
            "data": self.data,
            "data_signatures": sorted(self.data_signatures),
            "task_mapping": self.task_mapping,
            "dependency": self._serialize_dependency(),
            "log": self.log,
            "generated_collect_tasks": self.generated_collect_tasks,
            "generated_analysis_tasks": self.generated_analysis_tasks,
        }

    def _load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.task_mapping = list(snapshot.get("task_mapping", []))
        self.dependency = self._deserialize_dependency(snapshot.get("dependency", {}))
        self.log = list(snapshot.get("log", []))
        self.generated_collect_tasks = self._normalize_text_list(snapshot.get("generated_collect_tasks", []))
        self.generated_analysis_tasks = self._normalize_text_list(snapshot.get("generated_analysis_tasks", []))
        self.selection_traces = list(snapshot.get("selection_traces", []))
        raw_metadata = dict(snapshot.get("metadata", {}) or {})
        self.run_id = str(raw_metadata.get("run_id") or snapshot.get("run_id") or self.run_id or make_run_id())
        self.metadata = {
            **raw_metadata,
            "schema_version": 3,
            "created_at": raw_metadata.get("created_at", snapshot.get("created_at", self._now())),
            "updated_at": raw_metadata.get("updated_at", snapshot.get("updated_at", self._now())),
            "run_id": self.run_id,
        }

        self.records = {}
        self.record_order = []
        raw_records = snapshot.get("records")
        if isinstance(raw_records, dict) and raw_records:
            for fallback_id, raw_record in raw_records.items():
                record = self._coerce_record(raw_record)
                if record is None:
                    continue
                if not record.id:
                    record.id = str(fallback_id)
                self.records[record.id] = record

            raw_order = snapshot.get("record_order", [])
            if isinstance(raw_order, (list, tuple)):
                self.record_order = [str(record_id) for record_id in raw_order if str(record_id) in self.records]
            for record_id in self.records:
                if record_id not in self.record_order:
                    self.record_order.append(record_id)
            self._rebuild_indexes()
            return

        for item in list(snapshot.get("data", [])):
            try:
                record = self._normalize_item(item)
                self._insert_or_merge_record(record)
            except Exception:
                self.data.append(item)

        self._rebuild_indexes()

    def _dump_state(self, path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        serializers = []
        if dill is not None:
            serializers.append(dill)
        serializers.append(pickle)

        last_error: Optional[Exception] = None
        for serializer in serializers:
            try:
                with open(tmp_path, "wb") as f:
                    serializer.dump(payload, f)
                os.replace(tmp_path, path)
                return
            except Exception as exc:
                last_error = exc
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        if last_error is not None:
            raise last_error

    def _load_state(self, path: str) -> Dict[str, Any]:
        loaders = []
        if dill is not None:
            loaders.append(dill)
        loaders.append(pickle)

        last_error: Optional[Exception] = None
        for loader in loaders:
            try:
                with open(path, "rb") as f:
                    return loader.load(f)
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise FileNotFoundError(path)

    def _load_task_candidates(self, items: List[Any]) -> List[Dict[str, Any]]:
        return [
            {
                "index": idx,
                "text": self._item_text(item)[:1000],
                "class": item.__class__.__name__,
                "name": getattr(item, "name", getattr(item, "title", "")),
            }
            for idx, item in enumerate(items)
        ]

    def _record_selection_trace(
        self,
        *,
        operation: str,
        query: str,
        candidate_records: List[MemoryRecord],
        selected_records: List[MemoryRecord],
        selected_by: str,
        prompt: str = "",
        response: Any = None,
        model_name: Optional[str] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        context = self._current_context()
        trace_seed = {
            "operation": operation,
            "query": query,
            "selected_by": selected_by,
            "candidate_ids": [record.id for record in candidate_records],
            "selected_ids": [record.id for record in selected_records],
            "timestamp": self._now(),
            "run_id": self._context_value(context, "run_id") or self.run_id,
        }
        trace_id = f"selection_{hashlib.sha1(self._json_dumps(trace_seed).encode('utf-8')).hexdigest()[:16]}"
        trace = {
            "trace_id": trace_id,
            **trace_seed,
            "agent_id": self._context_value(context, "agent_id"),
            "agent_name": self._context_value(context, "agent_name"),
            "task_id": self._context_value(context, "task_id"),
            "step_id": context.step_id,
            "tool_name": self._context_value(context, "tool_name"),
            "model_name": model_name,
            "candidate_count": len(candidate_records),
            "selected_count": len(selected_records),
            "prompt_preview": self._safe_str(prompt, max_chars=2000),
            "response_preview": self._safe_str(response, max_chars=2000),
            "note": note,
        }

        if trace_id not in {item.get("trace_id") for item in self.selection_traces}:
            self.selection_traces.append(trace)

        event = context.provenance_event(
            "select_records",
            kind="selection",
            selection_trace_id=trace_id,
            operation=operation,
            query=query,
            selected_by=selected_by,
            selected_ids=[record.id for record in selected_records],
            candidate_count=len(candidate_records),
        )
        event["event_id"] = f"prov_{hashlib.sha1(self._json_dumps(event).encode('utf-8')).hexdigest()[:16]}"

        for record in selected_records:
            trace_ids = list(record.metadata.get("selection_trace_ids", []) or [])
            if trace_id not in trace_ids:
                trace_ids.append(trace_id)
            record.metadata["selection_trace_ids"] = trace_ids
            record.metadata["selection_traces"] = self._merge_event_list(
                record.metadata.get("selection_traces", []),
                trace,
            )
            record.metadata = self._apply_provenance_metadata(record.metadata, event)
            record.quality_score, record.scores = self._score_record(record)

        self.metadata["updated_at"] = self._now()
        return trace

    def save(self, state: Optional[Dict[str, Any]] = None) -> None:
        if state:
            self.metadata["extra_state"] = state
        payload = self._snapshot()
        if state is not None:
            payload["extra_state"] = state
        self._dump_state(self.memory_file, payload)

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.memory_file):
            return None
        snapshot = self._load_state(self.memory_file)
        if not isinstance(snapshot, dict):
            return None
        self._load_snapshot(snapshot)
        return snapshot

    def add_data(
        self,
        item: Any,
        *,
        source_agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if item is None:
            return None

        if isinstance(item, (list, tuple, set)):
            for sub_item in item:
                self.add_data(
                    sub_item,
                    source_agent_id=source_agent_id,
                    task_id=task_id,
                    tool_name=tool_name,
                    metadata=metadata,
                )
            return item

        record = self._normalize_item(
            item,
            source_agent_id=source_agent_id,
            task_id=task_id,
            tool_name=tool_name,
            metadata=metadata,
        )
        self._insert_or_merge_record(record)
        return item

    def add_log(self, *args, **kwargs) -> Dict[str, Any]:
        context = self._current_context()
        if args:
            agent_id = args[0] if len(args) > 0 else kwargs.get("agent_id") or kwargs.get("id")
            agent_type = args[1] if len(args) > 1 else kwargs.get("type")
            input_data = args[2] if len(args) > 2 else kwargs.get("input_data", {})
            output_data = args[3] if len(args) > 3 else kwargs.get("output_data", {})
        else:
            agent_id = kwargs.get("agent_id") or kwargs.get("id")
            agent_type = kwargs.get("type")
            input_data = kwargs.get("input_data", {})
            output_data = kwargs.get("output_data", {})

        agent_id = agent_id or self._context_value(context, "agent_id")
        log_event = context.provenance_event(
            "add_log",
            kind="runtime",
            log_agent_id=agent_id,
            log_type=agent_type,
            error=bool(kwargs.get("error", False)),
            note=kwargs.get("note", ""),
        )
        log_event["event_id"] = f"prov_{hashlib.sha1(self._json_dumps(log_event).encode('utf-8')).hexdigest()[:16]}"

        entry = {
            "timestamp": self._now(),
            "agent_id": agent_id,
            "agent_name": self._context_value(context, "agent_name"),
            "run_id": kwargs.get("run_id") or self._context_value(context, "run_id") or self.run_id,
            "task_id": kwargs.get("task_id") or self._context_value(context, "task_id"),
            "step_id": context.step_id,
            "tool_name": kwargs.get("tool_name") or self._context_value(context, "tool_name"),
            "parent_record_ids": list(context.parent_record_ids),
            "type": agent_type,
            "input_data": input_data,
            "output_data": output_data,
            "error": bool(kwargs.get("error", False)),
            "note": kwargs.get("note", ""),
            "provenance": log_event,
        }
        self.log.append(entry)
        self.metadata["updated_at"] = entry["timestamp"]
        return entry

    def add_dependency(self, child_id: str, parent_id: str) -> None:
        if not parent_id or not child_id:
            return
        self.dependency[str(parent_id)].add(str(child_id))
        self.metadata["updated_at"] = self._now()

    def get_log(self, parent_id: str, key: str = None) -> List[Dict[str, Any]]:
        child_list = self.dependency.get(str(parent_id), set())
        return_log = []
        for child_id in child_list:
            if key is not None and key not in child_id:
                continue
            child_log = [
                item for item in self.log
                if item.get("agent_id") == child_id or item.get("id") == child_id
            ]
            return_log.extend(child_log)
        return return_log

    def get_log_by_type(self, input_type: str) -> List[Dict[str, Any]]:
        return [item for item in self.log if input_type in str(item.get("type", ""))]

    def get_url_title(self, url: str) -> str:
        normalized_url = self._normalize_url(url)
        record_id = self.index_by_url.get(normalized_url)
        if record_id and record_id in self.records:
            record = self.records[record_id]
            title = record.metadata.get("search_title")
            if title:
                return str(title)
            if "search" in self._record_labels(record):
                return record.title

        for record in self.get_records():
            if record.url != normalized_url:
                continue
            title = record.metadata.get("search_title")
            if title:
                return str(title)
            if "search" in self._record_labels(record):
                return record.title
        return ""

    def _normalize_type_filter(self, values: Optional[Any]) -> set[str]:
        if not values:
            return set()
        if isinstance(values, str):
            values = [values]
        return {
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        }

    def get_records(
        self,
        memory_type: Optional[Any] = None,
        exclude_type: Optional[Any] = None,
        min_quality: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryRecord]:
        include_labels = self._normalize_type_filter(memory_type)
        exclude_labels = self._normalize_type_filter(exclude_type)

        records = []
        for record_id in self.record_order:
            record = self.records.get(record_id)
            if record is None:
                continue
            labels = self._record_labels(record)
            if include_labels and not (labels & include_labels):
                continue
            if exclude_labels and (labels & exclude_labels):
                continue
            if min_quality is not None and record.quality_score < min_quality:
                continue
            records.append(record)

        if top_k is not None and top_k >= 0:
            order_index = {record_id: idx for idx, record_id in enumerate(self.record_order)}
            records = sorted(
                records,
                key=lambda record: (-record.quality_score, order_index.get(record.id, 0)),
            )[:top_k]
        return records

    def _resolve_record_reference(self, record_ref: Any) -> Optional[MemoryRecord]:
        if isinstance(record_ref, MemoryRecord):
            return record_ref
        if isinstance(record_ref, str):
            record = self.get_record(record_ref)
            if record:
                return record
            return self.get_record_by_semantic_key(record_ref)
        for record in self.records.values():
            if record.raw is record_ref:
                return record
        return None

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        return self.records.get(str(record_id))

    def get_record_by_semantic_key(self, semantic_key: str) -> Optional[MemoryRecord]:
        record_id = self.index_by_semantic_key.get(str(semantic_key))
        if not record_id:
            return None
        return self.records.get(record_id)

    def get_provenance(self, record_ref: Any) -> List[Dict[str, Any]]:
        record = self._resolve_record_reference(record_ref)
        if record is None:
            return []
        return self._merge_event_list(
            [record.metadata.get("origin")] if record.metadata.get("origin") else [],
            record.metadata.get("provenance", []),
            record.metadata.get("derivation", []),
        )

    def get_lineage(self, record_ref: Any, max_depth: int = 5) -> Dict[str, Any]:
        record = self._resolve_record_reference(record_ref)
        if record is None:
            return {}

        visited: set[str] = set()

        def _walk(current: MemoryRecord, depth: int) -> Dict[str, Any]:
            visited.add(current.id)
            parent_ids = self._metadata_parent_record_ids(current.metadata)
            parents = []
            if depth < max_depth:
                for parent_id in parent_ids:
                    parent = self.get_record(parent_id)
                    if parent is None or parent.id in visited:
                        continue
                    parents.append(_walk(parent, depth + 1))
            return {
                "record_id": current.id,
                "semantic_key": current.semantic_key,
                "memory_type": current.memory_type,
                "title": current.title,
                "quality_score": current.quality_score,
                "parent_record_ids": parent_ids,
                "provenance": self.get_provenance(current),
                "parents": parents,
            }

        return _walk(record, 0)

    def get_selection_traces(self, record_ref: Any = None) -> List[Dict[str, Any]]:
        if record_ref is None:
            return list(self.selection_traces)
        record = self._resolve_record_reference(record_ref)
        if record is None:
            return []
        trace_ids = set(record.metadata.get("selection_trace_ids", []) or [])
        return [trace for trace in self.selection_traces if trace.get("trace_id") in trace_ids]

    def get_collect_data(self, exclude_type: Optional[Any] = None) -> List[Any]:
        exclude_labels = self._normalize_type_filter(exclude_type)
        exclude_labels.add("analysis")
        return [
            record.raw
            for record in self.get_records(exclude_type=exclude_labels)
            if record.raw is not None
        ]

    def get_analysis_result(self) -> List[Any]:
        return [
            record.raw
            for record in self.get_records(memory_type="analysis")
            if record.raw is not None
        ]

    def get_formatted_analysis_result(self, analysis_result_list: Optional[List[Any]] = None) -> str:
        if analysis_result_list is None:
            analysis_result_list = self.get_analysis_result()
        formatted = ""
        for idx, item in enumerate(analysis_result_list):
            formatted += f"Analysis report {idx + 1}:\n"
            formatted += str(item)
            formatted += "\n\n"
        return formatted

    def get_formatted_data_description(self, data_list: Optional[List[Any]] = None) -> str:
        if data_list is None:
            data_list = self.get_collect_data(exclude_type=["search"])
        formatted = ""
        for item in data_list:
            formatted += self._item_text(item)
            formatted += "\n\n"
        return formatted

    async def generate_collect_tasks(
        self,
        query: str,
        use_llm_name: Optional[str] = None,
        max_num: int = 5,
        existing_tasks: Optional[Iterable[Any]] = None,
    ) -> List[str]:
        existing = self._normalize_text_list(existing_tasks)
        fallback = self._fallback_collect_tasks(query, max_num=max_num, existing_tasks=existing)
        tasks = await self._generate_tasks(
            prompt_keys=["generate_collect_tasks", "generate_collect_task", "generate_industry_collect_task"],
            query=query,
            use_llm_name=use_llm_name,
            max_num=max_num,
            existing_tasks=existing,
            fallback_tasks=fallback,
            storage_attr="generated_collect_tasks",
        )
        return tasks

    async def generate_analyze_tasks(
        self,
        query: str,
        use_llm_name: Optional[str] = None,
        max_num: int = 5,
        existing_tasks: Optional[Iterable[Any]] = None,
    ) -> List[str]:
        existing = self._normalize_text_list(existing_tasks)
        fallback = self._fallback_analyze_tasks(query, max_num=max_num, existing_tasks=existing)
        tasks = await self._generate_tasks(
            prompt_keys=["generate_analyze_tasks", "generate_task"],
            query=query,
            use_llm_name=use_llm_name,
            max_num=max_num,
            existing_tasks=existing,
            fallback_tasks=fallback,
            storage_attr="generated_analysis_tasks",
        )
        return tasks

    async def _generate_tasks(
        self,
        prompt_keys: Any,
        query: str,
        use_llm_name: Optional[str],
        max_num: int,
        existing_tasks: List[str],
        fallback_tasks,
        storage_attr: str,
    ) -> List[str]:
        llm = self._resolve_llm(use_llm_name)
        loader = self._get_prompt_loader()
        tasks: List[str] = []

        if llm is not None and loader is not None:
            try:
                if isinstance(prompt_keys, str):
                    prompt_keys = [prompt_keys]
                prompt_template = None
                for prompt_key in prompt_keys:
                    prompt_template = loader.get_prompt(prompt_key)
                    if prompt_template:
                        break
                if prompt_template:
                    prompt = prompt_template.format(
                        query=query,
                        target_type=self._target_type(),
                        target_name=self._target_name(),
                        current_time=self._now(),
                        existing_tasks="\n".join(
                            f"{idx + 1}. {task}" for idx, task in enumerate(existing_tasks)
                        ) or "(none)",
                        max_num=max_num,
                    )
                    response = await llm.generate(messages=[{"role": "user", "content": prompt}])
                    tasks = self._parse_task_response(response)
            except Exception:
                tasks = []

        if not tasks:
            tasks = fallback_tasks

        tasks = self._dedupe_tasks(tasks, existing_tasks)
        tasks = tasks[:max_num]
        setattr(self, storage_attr, tasks)
        self.save()
        return tasks

    def _parse_task_response(self, response: Any) -> List[str]:
        if response is None:
            return []
        text = str(response).strip()
        if not text:
            return []

        fenced_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fenced_matches:
            text = fenced_matches[-1].strip()

        parsed: Any = None
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                import json_repair

                parsed = json_repair.loads(text)
            except Exception:
                parsed = None

        if isinstance(parsed, dict):
            for key in ("tasks", "collect_tasks", "analysis_tasks", "items", "data"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break

        result: List[str] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        result.append(cleaned)
                elif isinstance(item, dict):
                    for key in ("task", "content", "text", "name"):
                        if item.get(key):
                            cleaned = str(item[key]).strip()
                            if cleaned:
                                result.append(cleaned)
                                break

        if result:
            return result

        for line in text.splitlines():
            cleaned = line.strip()
            cleaned = re.sub(r"^[\-\*\u2022]\s*", "", cleaned)
            cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
            if cleaned:
                result.append(cleaned)

        return result

    def _dedupe_tasks(self, tasks: Iterable[Any], existing_tasks: Iterable[str]) -> List[str]:
        seen = {task.strip().lower() for task in existing_tasks if str(task).strip()}
        deduped: List[str] = []
        for task in tasks:
            text = str(task).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped

    def _fallback_collect_tasks(
        self,
        query: str,
        max_num: int,
        existing_tasks: Optional[Iterable[str]] = None,
    ) -> List[str]:
        target_type = self._target_type()
        target_name = self._target_name() or query
        templates = {
            "financial_company": [
                f"Collect company profile and business model for {target_name}",
                f"Collect latest financial statements and core metrics for {target_name}",
                f"Collect stock price, valuation, and trading activity for {target_name}",
                f"Collect shareholder structure and governance details for {target_name}",
                f"Collect peer and industry comparison data for {target_name}",
            ],
            "macro": [
                f"Collect the latest macro indicators related to {target_name}",
                f"Collect historical trend data and key policy changes for {target_name}",
                f"Collect region-specific supporting evidence for {target_name}",
            ],
            "industry": [
                f"Collect industry size and growth data for {target_name}",
                f"Collect competitive landscape and peer data for {target_name}",
                f"Collect demand, supply, and policy drivers for {target_name}",
            ],
            "general": [
                f"Collect authoritative background sources for {target_name}",
                f"Collect evidence that explains the main subtopics in {target_name}",
                f"Collect recent source material and cross-checks for {target_name}",
            ],
        }
        tasks = templates.get(target_type, templates["general"])
        return self._dedupe_tasks(tasks, existing_tasks or [])[:max_num]

    def _fallback_analyze_tasks(
        self,
        query: str,
        max_num: int,
        existing_tasks: Optional[Iterable[str]] = None,
    ) -> List[str]:
        target_type = self._target_type()
        target_name = self._target_name() or query
        templates = {
            "financial_company": [
                f"Analyze revenue, margin, and growth trends for {target_name}",
                f"Analyze balance-sheet strength and cash-flow quality for {target_name}",
                f"Analyze valuation, peers, and investment thesis for {target_name}",
                f"Analyze major risks, catalysts, and governance issues for {target_name}",
                f"Analyze long-term strategic positioning for {target_name}",
            ],
            "macro": [
                f"Analyze the directional trend of the key macro indicators for {target_name}",
                f"Analyze the policy and structural drivers behind {target_name}",
                f"Analyze implications for markets, sectors, and timing for {target_name}",
            ],
            "industry": [
                f"Analyze industry growth and demand drivers for {target_name}",
                f"Analyze competitive positioning and profit pools for {target_name}",
                f"Analyze policy, supply chain, and cyclical risks for {target_name}",
            ],
            "general": [
                f"Analyze the main evidence clusters in {target_name}",
                f"Analyze contradictions, uncertainties, and confidence levels in {target_name}",
                f"Analyze implications and next-step questions for {target_name}",
            ],
        }
        tasks = templates.get(target_type, templates["general"])
        return self._dedupe_tasks(tasks, existing_tasks or [])[:max_num]

    def is_agent_finished(self, agent_id: str) -> bool:
        cache_dir = os.path.join(self.working_dir, "agent_working", agent_id, ".cache")
        if not os.path.isdir(cache_dir):
            return False

        mapped_checkpoint = None
        for task_entry in self.task_mapping:
            if task_entry.get("agent_id") == agent_id:
                mapped_checkpoint = task_entry.get("checkpoint_name")
                break

        checkpoint_files = []
        for file_name in [
            mapped_checkpoint,
            "report_latest.pkl",
            "deepsearch_latest.pkl",
            "latest.pkl",
        ]:
            if file_name and file_name not in checkpoint_files:
                checkpoint_files.append(file_name)

        for file_name in checkpoint_files:
            checkpoint_path = os.path.join(cache_dir, file_name)
            if not os.path.exists(checkpoint_path):
                continue
            try:
                state = self._load_state(checkpoint_path)
            except Exception:
                continue

            if isinstance(state, dict) and (state.get("finished") or "return_dict" in state):
                return True
        return False

    async def get_or_create_agent(
        self,
        agent_class: Any,
        task_input: Dict[str, Any],
        resume: bool = True,
        priority: int = 0,
        checkpoint_name: Optional[str] = None,
        **agent_kwargs,
    ):
        task_input = task_input or {}
        agent_kwargs = dict(agent_kwargs)
        requested_agent_id = agent_kwargs.pop("agent_id", None)
        runtime_checkpoint = agent_kwargs.pop("checkpoint_name", None)
        checkpoint_name = checkpoint_name or runtime_checkpoint or self._default_checkpoint_name(agent_class)

        task_key = self._task_signature(agent_class, task_input, agent_kwargs)
        existing_entry = next((item for item in self.task_mapping if item.get("task_key") == task_key), None)
        agent_id = requested_agent_id
        if resume and existing_entry and not agent_id:
            agent_id = existing_entry.get("agent_id")

        if resume and existing_entry and agent_id:
            restored_agent = None
            try:
                restored_agent = await agent_class.from_checkpoint(
                    config=self.config,
                    memory=self,
                    agent_id=agent_id,
                    checkpoint_name=existing_entry.get("checkpoint_name", checkpoint_name),
                    restored_agents={},
                    **agent_kwargs,
                )
            except Exception:
                restored_agent = None

            if restored_agent is not None:
                agent = restored_agent
            else:
                agent = agent_class(
                    config=self.config,
                    memory=self,
                    agent_id=agent_id,
                    **agent_kwargs,
                )
        else:
            agent = agent_class(
                config=self.config,
                memory=self,
                agent_id=agent_id,
                **agent_kwargs,
            )

        entry = {
            "task_key": task_key,
            "agent_class_name": getattr(agent_class, "__name__", str(agent_class)),
            "agent_name": getattr(agent_class, "AGENT_NAME", getattr(agent_class, "__name__", str(agent_class))),
            "agent_id": agent.id,
            "priority": priority,
            "checkpoint_name": checkpoint_name,
            "task_input": task_input,
            "task_summary": self._task_summary(task_input),
            "agent_kwargs": self._canonical_object(agent_kwargs),
            "status": "restored" if resume and existing_entry else "created",
            "created_at": existing_entry.get("created_at", self._now()) if existing_entry else self._now(),
            "updated_at": self._now(),
        }
        self._upsert_task_mapping(entry)
        self.save()
        return agent

    async def retrieve_relevant_data(
        self,
        query: str,
        top_k: int = 5,
        embedding_model: Optional[str] = None,
    ) -> List[Any]:
        candidates = self.data
        if not candidates:
            return []

        embedding_model = embedding_model or self._infer_embedding_model_name()

        try:
            from src.utils.index_builder import IndexBuilder

            if embedding_model:
                index = IndexBuilder(
                    config=self.config,
                    embedding_model=embedding_model,
                    working_dir=self.memory_dir,
                )
                if len(getattr(index, "embeddings", [])) != len(candidates):
                    await index._build_index([self._item_text(item) for item in candidates])
                results = await index.search(query, top_k=top_k)
                selected: List[Any] = []
                seen: set[int] = set()
                for result in results:
                    idx = int(result.get("id", -1))
                    if 0 <= idx < len(candidates) and idx not in seen:
                        selected.append(candidates[idx])
                        seen.add(idx)
                if selected:
                    return selected
        except Exception:
            pass

        return self._fallback_rank(query, candidates, top_k)

    def _infer_embedding_model_name(self) -> Optional[str]:
        llm_dict = getattr(self.config, "llm_dict", {}) or {}
        if not llm_dict:
            return None
        for key, llm in llm_dict.items():
            model_name = getattr(llm, "model_name", None)
            if model_name and "embedding" in str(model_name).lower():
                return key
        return next(iter(llm_dict.keys()), None)

    def _fallback_rank(self, query: str, candidates: List[Any], top_k: int) -> List[Any]:
        query_tokens = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", query.lower()))
        scored = []
        for idx, item in enumerate(candidates):
            text = self._item_text(item).lower()
            item_tokens = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text))
            overlap = len(query_tokens & item_tokens)
            bonus = 2 if query.lower() in text else 0
            score = overlap + bonus
            scored.append((score, idx, item))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item for _, _, item in scored[:top_k]]

    async def select_data_by_llm(
        self,
        query: str,
        top_k: int = 5,
        use_llm_name: Optional[str] = None,
        max_k: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> tuple[List[Any], str]:
        if max_k is not None and max_k > 0:
            top_k = max_k
        if model_name is not None:
            use_llm_name = model_name

        candidate_records = [record for record in self.get_records(exclude_type=["analysis"]) if record.raw is not None]
        candidates = [record.raw for record in candidate_records]
        if not candidates:
            return [], "No data available"

        loader = self._get_prompt_loader()
        llm = self._resolve_llm(use_llm_name)
        if loader is None or llm is None:
            selected = self._fallback_rank(query, candidates, top_k)
            selected_raw_ids = {id(item) for item in selected}
            selected_records = [record for record in candidate_records if id(record.raw) in selected_raw_ids]
            self._record_selection_trace(
                operation="select_data_by_llm",
                query=query,
                candidate_records=candidate_records,
                selected_records=selected_records,
                selected_by="fallback_similarity",
                model_name=use_llm_name,
                note="Prompt loader or LLM unavailable; used similarity fallback.",
            )
            return selected, "Prompt loader or LLM unavailable; used similarity fallback."

        prompt = ""
        response: Any = ""
        try:
            prompt_template = loader.get_prompt("select_data_by_llm")
            if prompt_template:
                candidate_lines = []
                for candidate in self._load_task_candidates(candidates[: min(len(candidates), 20)]):
                    record = candidate_records[candidate["index"]]
                    candidate_lines.append(
                        f"{candidate['index']}. [{record.id}] {candidate['class']}: {candidate['text']}"
                    )
                prompt = prompt_template.format(
                    query=query,
                    candidate_items="\n\n".join(candidate_lines),
                    current_time=self._now(),
                )
            else:
                prompt_template = loader.get_prompt("select_data")
                prompt = prompt_template.format(
                    data_description=self.get_formatted_data_description(candidates),
                    section_description=query,
                    query=query,
                    current_time=self._now(),
                )
            response = await llm.generate(messages=[{"role": "user", "content": prompt}])
            indices = self._parse_selected_indices(response, len(candidates))
            if indices:
                selected_records = [candidate_records[idx] for idx in indices[:top_k]]
                selected = [record.raw for record in selected_records]
                self._record_selection_trace(
                    operation="select_data_by_llm",
                    query=query,
                    candidate_records=candidate_records,
                    selected_records=selected_records,
                    selected_by="llm_indices",
                    prompt=prompt,
                    response=response,
                    model_name=use_llm_name,
                )
                return selected, str(response)
            names = self._parse_selected_names(response, "selected_data_list")
            if names:
                selected_records = [
                    record
                    for record in candidate_records
                    if getattr(record.raw, "name", None) in names or record.title in names
                ][:top_k]
                selected = [record.raw for record in selected_records]
                if selected:
                    self._record_selection_trace(
                        operation="select_data_by_llm",
                        query=query,
                        candidate_records=candidate_records,
                        selected_records=selected_records,
                        selected_by="llm_names",
                        prompt=prompt,
                        response=response,
                        model_name=use_llm_name,
                    )
                    return selected, self.get_formatted_data_description(selected)
        except Exception as exc:
            selected = self._fallback_rank(query, candidates, top_k)
            selected_raw_ids = {id(item) for item in selected}
            selected_records = [record for record in candidate_records if id(record.raw) in selected_raw_ids]
            self._record_selection_trace(
                operation="select_data_by_llm",
                query=query,
                candidate_records=candidate_records,
                selected_records=selected_records,
                selected_by="fallback_after_error",
                prompt=prompt,
                response=response,
                model_name=use_llm_name,
                note=str(exc),
            )
            return selected, f"LLM selection failed and fallback was used: {exc}"

        selected = self._fallback_rank(query, candidates, top_k)
        selected_raw_ids = {id(item) for item in selected}
        selected_records = [record for record in candidate_records if id(record.raw) in selected_raw_ids]
        self._record_selection_trace(
            operation="select_data_by_llm",
            query=query,
            candidate_records=candidate_records,
            selected_records=selected_records,
            selected_by="fallback_no_parse",
            prompt=prompt,
            response=response,
            model_name=use_llm_name,
        )
        return selected, str(response)

    async def select_analysis_result_by_llm(
        self,
        query: str,
        max_k: int = -1,
        model_name: Optional[str] = None,
    ) -> tuple[List[Any], str]:
        candidate_records = [record for record in self.get_records(memory_type="analysis") if record.raw is not None]
        candidates = [record.raw for record in candidate_records]
        if not candidates:
            return [], ""

        loader = self._get_prompt_loader()
        llm = self._resolve_llm(model_name)
        if loader is None or llm is None:
            limit = len(candidates) if max_k is None or max_k < 0 else max_k
            selected = candidates[:limit]
            self._record_selection_trace(
                operation="select_analysis_result_by_llm",
                query=query,
                candidate_records=candidate_records,
                selected_records=candidate_records[:limit],
                selected_by="fallback_all",
                model_name=model_name,
                note="Prompt loader or LLM unavailable.",
            )
            return selected, self.get_formatted_analysis_result(selected)

        prompt = ""
        response: Any = ""
        try:
            prompt_template = loader.get_prompt("select_analysis")
            if not prompt_template:
                limit = len(candidates) if max_k is None or max_k < 0 else max_k
                selected = candidates[:limit]
                self._record_selection_trace(
                    operation="select_analysis_result_by_llm",
                    query=query,
                    candidate_records=candidate_records,
                    selected_records=candidate_records[:limit],
                    selected_by="fallback_no_prompt",
                    model_name=model_name,
                )
                return selected, self.get_formatted_analysis_result(selected)
            prompt = prompt_template.format(
                analysis_description=self.get_formatted_analysis_result(candidates),
                section_description=query,
                query=query,
                current_time=self._now(),
            )
            response = await llm.generate(messages=[{"role": "user", "content": prompt}])
            names = self._parse_selected_names(response, "selected_analysis_list")
            selected_records = [record for record in candidate_records if getattr(record.raw, "title", None) in names]
            if max_k is not None and max_k > 0:
                selected_records = selected_records[:max_k]
            selected = [record.raw for record in selected_records]
            if selected:
                self._record_selection_trace(
                    operation="select_analysis_result_by_llm",
                    query=query,
                    candidate_records=candidate_records,
                    selected_records=selected_records,
                    selected_by="llm_names",
                    prompt=prompt,
                    response=response,
                    model_name=model_name,
                )
                return selected, self.get_formatted_analysis_result(selected)
        except Exception as exc:
            response = f"selection failed: {exc}"

        limit = len(candidates) if max_k is None or max_k < 0 else max_k
        selected = candidates[:limit]
        self._record_selection_trace(
            operation="select_analysis_result_by_llm",
            query=query,
            candidate_records=candidate_records,
            selected_records=candidate_records[:limit],
            selected_by="fallback_no_selection",
            prompt=prompt,
            response=response,
            model_name=model_name,
        )
        return selected, self.get_formatted_analysis_result(selected)

    def _parse_selected_names(self, response: Any, key: str) -> List[str]:
        if response is None:
            return []
        text = str(response).strip()
        if not text:
            return []
        fenced_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fenced_matches:
            text = fenced_matches[-1].strip()

        parsed: Any = None
        try:
            import json_repair

            parsed = json_repair.loads(text)
        except Exception:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

        values = []
        if isinstance(parsed, dict):
            raw_values = parsed.get(key) or parsed.get("selected_items") or parsed.get("items")
            if isinstance(raw_values, list):
                values = raw_values
        elif isinstance(parsed, list):
            values = parsed

        return [str(item).strip() for item in values if str(item).strip()]

    def _parse_selected_indices(self, response: Any, upper_bound: int) -> List[int]:
        if response is None:
            return []
        text = str(response).strip()
        if not text:
            return []

        try:
            import json_repair

            parsed = json_repair.loads(text)
        except Exception:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

        indices: List[int] = []
        if isinstance(parsed, dict):
            for key in ("selected_indices", "indices", "ids"):
                if isinstance(parsed.get(key), list):
                    indices = [int(item) for item in parsed[key] if str(item).isdigit()]
                    break
        elif isinstance(parsed, list):
            indices = [int(item) for item in parsed if str(item).isdigit()]

        if not indices:
            indices = [int(match) for match in re.findall(r"\b\d+\b", text)]

        seen = set()
        valid_indices: List[int] = []
        for idx in indices:
            if 0 <= idx < upper_bound and idx not in seen:
                seen.add(idx)
                valid_indices.append(idx)
        return valid_indices
