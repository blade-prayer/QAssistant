from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import dill  # type: ignore
except Exception:  # pragma: no cover - fallback for lean environments
    dill = None


class Memory:
    """Persistent shared state for the report-generation pipeline."""

    def __init__(self, config):
        self.config = config
        self.working_dir = getattr(config, "working_dir", os.path.abspath("./outputs"))
        self.memory_dir = os.path.join(self.working_dir, "memory")
        self.memory_file = os.path.join(self.memory_dir, "memory.pkl")

        self.data: List[Any] = []
        self.data_signatures: set[str] = set()
        self.task_mapping: List[Dict[str, Any]] = []
        self.dependency: Dict[str, set[str]] = defaultdict(set)
        self.log: List[Dict[str, Any]] = []
        self.generated_collect_tasks: List[str] = []
        self.generated_analysis_tasks: List[str] = []
        self.metadata: Dict[str, Any] = {
            "schema_version": 1,
            "created_at": self._now(),
            "updated_at": self._now(),
        }
        self._prompt_loader = None

        os.makedirs(self.memory_dir, exist_ok=True)

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
        class_name = item.__class__.__name__.lower()
        type_name = str(getattr(item, "type", "")).lower()
        source = str(getattr(item, "source", "")).lower()

        if "analysisresult" in class_name:
            return "analysis"
        if "searchresult" in class_name or "deepsearchresult" in class_name:
            return "search"
        if "search" in type_name:
            return "search"
        if "deepsearch agent" in source:
            return "search"
        return "collect"

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
        return {
            "schema_version": self.metadata.get("schema_version", 1),
            "created_at": self.metadata.get("created_at", self._now()),
            "updated_at": self._now(),
            "data": self.data,
            "data_signatures": sorted(self.data_signatures),
            "task_mapping": self.task_mapping,
            "dependency": self._serialize_dependency(),
            "log": self.log,
            "generated_collect_tasks": self.generated_collect_tasks,
            "generated_analysis_tasks": self.generated_analysis_tasks,
        }

    def _load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.data = list(snapshot.get("data", []))
        self.data_signatures = set(snapshot.get("data_signatures", []))
        if not self.data_signatures:
            self.data_signatures = {self._item_signature(item) for item in self.data}
        self.task_mapping = list(snapshot.get("task_mapping", []))
        self.dependency = self._deserialize_dependency(snapshot.get("dependency", {}))
        self.log = list(snapshot.get("log", []))
        self.generated_collect_tasks = self._normalize_text_list(snapshot.get("generated_collect_tasks", []))
        self.generated_analysis_tasks = self._normalize_text_list(snapshot.get("generated_analysis_tasks", []))
        self.metadata = {
            "schema_version": snapshot.get("schema_version", 1),
            "created_at": snapshot.get("created_at", self._now()),
            "updated_at": snapshot.get("updated_at", self._now()),
        }

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

    def add_data(self, item: Any) -> Any:
        if item is None:
            return None

        if isinstance(item, (list, tuple, set)):
            for sub_item in item:
                self.add_data(sub_item)
            return item

        signature = self._item_signature(item)
        if signature in self.data_signatures:
            return item

        self.data.append(item)
        self.data_signatures.add(signature)
        self.metadata["updated_at"] = self._now()
        return item

    def add_log(self, *args, **kwargs) -> Dict[str, Any]:
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

        entry = {
            "timestamp": self._now(),
            "agent_id": agent_id,
            "type": agent_type,
            "input_data": input_data,
            "output_data": output_data,
            "error": bool(kwargs.get("error", False)),
            "note": kwargs.get("note", ""),
        }
        self.log.append(entry)
        self.metadata["updated_at"] = entry["timestamp"]
        return entry

    def add_dependency(self, child_id: str, parent_id: str) -> None:
        if not parent_id or not child_id:
            return
        self.dependency[str(parent_id)].add(str(child_id))
        self.metadata["updated_at"] = self._now()

    def get_collect_data(self, exclude_type: Optional[str] = None) -> List[Any]:
        items = [item for item in self.data if self._classify_item(item) != "analysis"]
        if exclude_type:
            exclude_label = str(exclude_type).strip().lower()
            if exclude_label:
                items = [item for item in items if self._classify_item(item) != exclude_label]
        return list(items)

    def get_analysis_result(self) -> List[Any]:
        return [item for item in self.data if self._classify_item(item) == "analysis"]

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
            prompt_key="generate_collect_tasks",
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
            prompt_key="generate_analyze_tasks",
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
        prompt_key: str,
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
                prompt_template = loader.get_prompt(prompt_key)
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
    ) -> tuple[List[Any], str]:
        candidates = self.data
        if not candidates:
            return [], "No data available"

        loader = self._get_prompt_loader()
        llm = self._resolve_llm(use_llm_name)
        if loader is None or llm is None:
            selected = await self.retrieve_relevant_data(query, top_k=top_k)
            return selected, "Prompt loader or LLM unavailable; used similarity fallback."

        try:
            prompt_template = loader.get_prompt("select_data_by_llm")
            candidate_lines = []
            for candidate in self._load_task_candidates(candidates[: min(len(candidates), 20)]):
                candidate_lines.append(f"{candidate['index']}. {candidate['class']}: {candidate['text']}")
            prompt = prompt_template.format(
                query=query,
                candidate_items="\n\n".join(candidate_lines),
                current_time=self._now(),
            )
            response = await llm.generate(messages=[{"role": "user", "content": prompt}])
            indices = self._parse_selected_indices(response, len(candidates))
            if indices:
                selected = [candidates[idx] for idx in indices[:top_k]]
                return selected, str(response)
        except Exception as exc:
            selected = await self.retrieve_relevant_data(query, top_k=top_k)
            return selected, f"LLM selection failed and fallback was used: {exc}"

        selected = await self.retrieve_relevant_data(query, top_k=top_k)
        return selected, str(response)

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
