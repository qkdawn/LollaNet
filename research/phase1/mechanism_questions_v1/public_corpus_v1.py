"""Build public, structured-source fact packs for mechanism-question-set V1."""
from __future__ import annotations
import hashlib, json, re, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import requests
from .question_set_v1 import DOMAINS, PROFILE_BY_QUESTION_KIND, load_semantic_assets, validate_fact_pack, write_jsonl

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_ROOT = Path(r"D:\\LollaNetData\\mechanism-question-set-v1")
AS_OF = "2026-07-20T00:00:00Z"
BUILD_DATE = "2026-07-20"
USER_AGENT = "LollaNetResearch/1.0 public-corpus@example.com"
CISA_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
CPSC_URL = "https://www.saferproducts.gov/RestWebServices/Recall?format=json"
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
SEC_FRAME = "https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json"
CONCEPT_IDS = tuple(sorted(load_semantic_assets()["concept_ids"]))

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _iso_ms(value: int | float | None, fallback: str = AS_OF) -> str:
    if not value: return fallback
    return datetime.fromtimestamp(float(value)/1000, timezone.utc).isoformat().replace("+00:00", "Z")

def _iso_date(value: Any, fallback: str = AS_OF) -> str:
    text = str(value or "")[:10]
    try: datetime.fromisoformat(text)
    except ValueError: return fallback
    return f"{text}T00:00:00Z"

def _compact(value: Any, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit-1].rstrip() + "…"

def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "record"

def _request_bytes(url: str, cache_path: Path, params: dict[str, Any] | None = None, timeout: int = 180) -> bytes:
    if cache_path.exists() and cache_path.stat().st_size: return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate", "Accept": "application/json,text/plain,*/*"}
    error = None
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            if not response.content: raise RuntimeError("empty response")
            temp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temp.write_bytes(response.content); temp.replace(cache_path)
            return response.content
        except Exception as exc:
            error = exc; time.sleep(2 ** attempt)
    raise RuntimeError(f"official source download failed: {url}: {error}")

def _json_source(url: str, cache_path: Path, params: dict[str, Any] | None = None) -> Any:
    data = _request_bytes(url, cache_path, params=params)
    try: return json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise RuntimeError(f"invalid official JSON: {url}") from exc

def _source(source_id: str, title: str, url: str, published_at: str, **extra: Any) -> dict[str, Any]:
    row = {"source_id": source_id, "title": title, "url": url, "published_at": published_at}; row.update(extra); return row

def _fact(fact_id: str, statement: str, refs: Iterable[str], available_at: str, **extra: Any) -> dict[str, Any]:
    row = {"fact_id": fact_id, "statement": _compact(statement), "evidence_refs": list(refs), "available_at": available_at}; row.update(extra); return row

def _states(active: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for concept_id in CONCEPT_IDS:
        item = active.get(concept_id)
        if not item:
            rows.append({"microconcept_id": concept_id, "status": "unknown", "direction": "unknown", "intensity": 0, "temporal_persistence": 0, "task_relevance": 0, "evidence_refs": []})
        else:
            rows.append({"microconcept_id": concept_id, "status": item.get("status", "supported"), "direction": item.get("direction", "present"), "intensity": int(item.get("intensity", 2)), "temporal_persistence": int(item.get("temporal_persistence", 2)), "task_relevance": int(item.get("task_relevance", 2)), "evidence_refs": list(item.get("evidence_refs", []))})
    return rows

MICROCONCEPT_PHRASES = {
    "capacity_headroom": "剩余承载空间",
    "constraint_pressure": "关键限制压力",
    "conversion_gap": "投入到可兑现结果之间的缺口",
    "countervailing_pressure": "抵消性力量",
    "evidence_quality": "证据的独立性与完整度",
    "feedback_polarity": "变化被放大或抵消的方向",
    "outcome_asymmetry": "上下行结果的不对称性",
    "regime_proximity": "距离失效边界的接近程度",
    "relative_gradient": "对象、时间或位置之间的差异斜率",
    "resource_balance": "资源流入、存量与流出的平衡",
    "temporal_persistence": "状态延续的时间尺度",
}

DOMAIN_TASK_CONTEXT = {
    "business_finance": "现金、负担、兑现能力与资源配置",
    "software_engineering": "暴露、整改窗口、运行连续性与工程资源",
    "product_organization": "危害暴露、处置落实、协作能力与使用者保护",
    "science_public_systems": "测量质量、影响边界、系统暴露与公共响应",
}

QUESTION_STEMS = {
    "mechanism_diagnosis": (
        "在当前资料截点，怎样的因果结构最能同时解释{state_summary}？",
        "若不借助机构身份，哪一种作用关系最能把{state_summary}连成可检验解释？",
        "面对{state_summary}，应优先解释哪个限制、传导或边界变化？",
        "哪些相互作用最可能造成{state_summary}，并会产生什么后续观察？",
        "把现象拆成状态变化后，什么机制最能解释{state_summary}而不过度外推？",
        "当前记录同时出现{state_summary}；哪个因果链条最值得作为首要假设？",
        "若要求解释能够被新数据推翻，如何说明{state_summary}的共同来源？",
        "对于{domain_context}任务，{state_summary}最支持哪一类首要机制假设？",
    ),
    "decision_tradeoff": (
        "在{domain_context}目标下，面对{state_summary}，下一步应先改变哪个可控条件？",
        "资源只能支持一个短周期动作时，怎样针对{state_summary}排序行动与暂缓项？",
        "若错误代价不对称，围绕{state_summary}应先购买信息还是直接干预？",
        "在预算、时间和可逆性受限时，什么动作最能缓解{state_summary}中的关键问题？",
        "如何把{state_summary}转化为一个有停止条件的优先行动，而不同时押注所有方案？",
        "当前只能改变一个条件；针对{state_summary}，应选哪个条件并保留什么安全检查？",
        "若短期改善不能牺牲长期韧性，怎样处理{state_summary}所揭示的取舍？",
        "对{domain_context}而言，哪项最小可逆行动最能验证并改善{state_summary}？",
    ),
    "evidence_discrimination": (
        "针对{state_summary}，哪一条新增观察最能让两个竞争解释给出不同预测？",
        "现有状态是否足以区分{state_summary}的不同成因；若不足，应核验什么？",
        "什么独立、带时间戳的证据最可能推翻当前对{state_summary}的首要解释？",
        "若只能补充一个测量，怎样选择才能区分{state_summary}中的竞争机制？",
        "对{domain_context}任务，什么证据缺口使{state_summary}仍可能需要暂缓判断？",
        "哪些观察结果会分别增强、削弱或推翻对{state_summary}的解释？",
        "要避免把相关性当成机制，应如何核验{state_summary}的方向与持续性？",
        "何种反例最能挑战对{state_summary}的当前判断，并触发 abstain？",
    ),
}

COUNTERFACTUAL_FIELD_PREFERENCES = {
    "business_finance": ("liabilities_usd", "cash_usd", "operating_cash_flow_usd", "revenue_usd", "assets_usd"),
    "software_engineering": ("deadline_days", "ransomware_known"),
    "product_organization": ("injury_entries", "incident_entries", "affected_units"),
    "science_public_systems": ("magnitude", "mmi", "station_count", "azimuthal_gap", "depth_km"),
}


# Each preferred scalar has an explicit semantic bridge to one factual
# microconcept state. ``polarity`` maps the fact to state intensity; an
# optional ``persistence_polarity`` records a different time-axis relation.
COUNTERFACTUAL_STATE_SPECS = {
    "liabilities_usd": {
        "microconcept_id": "constraint_pressure", "polarity": 1,
        "up_direction": "counterfactual_pressure_tightening", "down_direction": "counterfactual_pressure_easing",
    },
    "cash_usd": {
        "microconcept_id": "capacity_headroom", "polarity": 1,
        "up_direction": "counterfactual_headroom_expanding", "down_direction": "counterfactual_headroom_contracting",
    },
    "operating_cash_flow_usd": {
        "microconcept_id": "conversion_gap", "polarity": -1,
        "up_direction": "counterfactual_conversion_gap_narrowing", "down_direction": "counterfactual_conversion_gap_widening",
    },
    "revenue_usd": {
        "microconcept_id": "conversion_gap", "polarity": 1,
        "up_direction": "counterfactual_conversion_demand_increasing", "down_direction": "counterfactual_conversion_demand_decreasing",
    },
    "assets_usd": {
        "microconcept_id": "capacity_headroom", "polarity": 1,
        "up_direction": "counterfactual_asset_headroom_expanding", "down_direction": "counterfactual_asset_headroom_contracting",
    },
    "deadline_days": {
        "microconcept_id": "constraint_pressure", "polarity": -1, "persistence_polarity": 1,
        "up_direction": "counterfactual_window_relaxing", "down_direction": "counterfactual_window_tightening",
    },
    "ransomware_known": {
        "microconcept_id": "outcome_asymmetry", "polarity": 1,
        "up_direction": "counterfactual_downside_signal_increasing", "down_direction": "counterfactual_downside_signal_decreasing",
    },
    "injury_entries": {
        "microconcept_id": "outcome_asymmetry", "polarity": 1,
        "up_direction": "counterfactual_harm_signal_increasing", "down_direction": "counterfactual_harm_signal_decreasing",
    },
    "incident_entries": {
        "microconcept_id": "outcome_asymmetry", "polarity": 1,
        "up_direction": "counterfactual_incident_signal_increasing", "down_direction": "counterfactual_incident_signal_decreasing",
    },
    "affected_units": {
        "microconcept_id": "relative_gradient", "polarity": 1,
        "up_direction": "counterfactual_exposure_gradient_increasing", "down_direction": "counterfactual_exposure_gradient_decreasing",
    },
    "magnitude": {
        "microconcept_id": "resource_balance", "polarity": 1,
        "up_direction": "counterfactual_released_energy_increasing", "down_direction": "counterfactual_released_energy_decreasing",
    },
    "mmi": {
        "microconcept_id": "regime_proximity", "polarity": 1,
        "up_direction": "counterfactual_harm_boundary_nearer", "down_direction": "counterfactual_harm_boundary_farther",
    },
    "station_count": {
        "microconcept_id": "evidence_quality", "polarity": 1,
        "up_direction": "counterfactual_measurement_support_increasing", "down_direction": "counterfactual_measurement_support_decreasing",
    },
    "azimuthal_gap": {
        "microconcept_id": "evidence_quality", "polarity": -1,
        "up_direction": "counterfactual_coverage_weakening", "down_direction": "counterfactual_coverage_improving",
    },
    "depth_km": {
        "microconcept_id": "conversion_gap", "polarity": 1,
        "up_direction": "counterfactual_source_to_impact_gap_increasing", "down_direction": "counterfactual_source_to_impact_gap_decreasing",
    },
}


def _state_summary(states: list[dict[str, Any]]) -> tuple[str, str]:
    active = sorted(
        (row for row in states if row["status"] in {"supported", "candidate", "mixed"}),
        key=lambda row: (-row["task_relevance"], -row["intensity"], row["microconcept_id"]),
    )
    if not active:
        return "evidence_quality", "证据充分度仍未知"
    focus = active[0]["microconcept_id"]
    selected = active[:2]
    status_phrases = {"supported": "已有事实支持", "candidate": "仍属候选", "mixed": "证据相互冲突"}
    intensity_phrases = {0: "未测量", 1: "较弱", 2: "中等", 3: "较强"}
    parts = [
        f"{MICROCONCEPT_PHRASES[row['microconcept_id']]}{status_phrases[row['status']]}、强度{intensity_phrases[row['intensity']]}"
        for row in selected
    ]
    return focus, "；".join(parts)


def _changed_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1 if value <= 1 else max(0, round(value * 0.75))
    if isinstance(value, float):
        return round(value + 0.1, 6) if value <= 0.1 else round(value * 0.75, 6)
    raise TypeError(f"unsupported counterfactual scalar: {value!r}")


def _counterfactual_change(domain: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
    for field in COUNTERFACTUAL_FIELD_PREFERENCES[domain]:
        for fact in facts:
            if field in fact and isinstance(fact[field], (bool, int, float)):
                return {"fact_id": fact["fact_id"], "field": field, "before": fact[field], "after": _changed_value(fact[field])}
    raise RuntimeError(f"no auditable scalar counterfactual field for {domain}")


def _counterfactual_state_changes(change: dict[str, Any], states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate one declared scalar fact change into explicit non-task state deltas."""
    spec = COUNTERFACTUAL_STATE_SPECS[change["field"]]
    by_id = {state["microconcept_id"]: state for state in states}
    state = by_id[spec["microconcept_id"]]
    if state["status"] == "unknown":
        raise RuntimeError(f"counterfactual target state is unknown: {spec['microconcept_id']}")
    raw_sign = 1 if change["after"] > change["before"] else -1
    effect_sign = raw_sign * spec["polarity"]
    direction = spec["up_direction"] if raw_sign > 0 else spec["down_direction"]
    changes: list[dict[str, Any]] = []
    if state["status"] == "supported":
        # A state implied by a hypothetical fact is a candidate, even when the
        # factual baseline was directly supported.
        changes.append({
            "microconcept_id": state["microconcept_id"], "axis": "status",
            "before": state["status"], "after": "candidate",
        })
    intensity_after = max(0, min(3, state["intensity"] + effect_sign))
    if intensity_after != state["intensity"]:
        changes.append({
            "microconcept_id": state["microconcept_id"], "axis": "intensity",
            "before": state["intensity"], "after": intensity_after,
        })
    changes.append({
        "microconcept_id": state["microconcept_id"], "axis": "direction",
        "before": state["direction"], "after": direction,
    })
    persistence_sign = raw_sign * spec.get("persistence_polarity", spec["polarity"])
    persistence_after = max(0, min(3, state["temporal_persistence"] + persistence_sign))
    if persistence_after == state["temporal_persistence"]:
        persistence_after = state["temporal_persistence"] - 1 if state["temporal_persistence"] > 0 else 1
    changes.append({
        "microconcept_id": state["microconcept_id"], "axis": "temporal_persistence",
        "before": state["temporal_persistence"], "after": persistence_after,
    })
    return changes


def _variants(case_index: int, domain: str, facts: list[dict[str, Any]], states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fact_ids = [fact["fact_id"] for fact in facts]
    focus, state_summary = _state_summary(states)

    def task_fields(question_kind: str) -> dict[str, str]:
        # These fields are part of the fact-pack audit trail.  The exact vector is
        # materialized later per Episode, so Router-visible state is question conditioned.
        return {
            "task_relevance_profile": PROFILE_BY_QUESTION_KIND[question_kind],
            "task_focus_microconcept_id": focus,
        }
    transfer = case_index < 8
    source_domain = DOMAINS[(DOMAINS.index(domain)+1) % len(DOMAINS)]
    domain_context = DOMAIN_TASK_CONTEXT[domain]
    constraints = [
        "仅使用题面给出的任务要求和机制状态，不读取机构身份或来源元数据",
        "区分已支持、候选、混合与未知状态",
        "结论必须给出可被后续观察推翻的条件",
    ]
    prompts = [
        (
            "mechanism_diagnosis",
            "选择一个主解释，说明可观察预测，并与一个相邻解释区分。",
        ),
        (
            "decision_tradeoff",
            "给出有边界的优先行动、暂不投入的替代项和停止条件。",
        ),
        (
            "evidence_discrimination",
            "判断应继续选择主解释还是暂缓，并提出信息价值最高的核验。",
        ),
    ]
    result = []
    for offset, (kind, goal) in enumerate(prompts):
        stem = QUESTION_STEMS[kind][(case_index + offset + DOMAINS.index(domain)) % len(QUESTION_STEMS[kind])]
        question = stem.format(state_summary=state_summary, domain_context=domain_context)
        is_transfer = transfer and kind in {"mechanism_diagnosis", "evidence_discrimination"}
        transfer_rationale = None
        if is_transfer:
            transfer_rationale = {
                "source_domain": source_domain,
                "structural_mapping": (
                    f"只迁移“{MICROCONCEPT_PHRASES[focus]}”与目标、边界和证据之间的结构关系；"
                    "领域对象名称和历史结果不作为迁移依据。"
                ),
                "evidence_refs": fact_ids[:2],
            }
        result.append(
            {
                "question_kind": kind,
                **task_fields(kind),
                "question": question,
                "goal": goal,
                "constraints": list(constraints),
                "cross_domain_transfer": is_transfer,
                "transfer_rationale": transfer_rationale,
                "counterfactual_delta": None,
            }
        )
    if case_index % 2 == 0:
        cf_task_fields = task_fields("conditional_or_counterfactual")
        change = _counterfactual_change(domain, facts)
        state_changes = _counterfactual_state_changes(change, states)
        question = (
            f"若只把 {change['fact_id']} 的 {change['field']} 从 {change['before']} 改为 {change['after']}，"
            f"并保持其余事实不变，相关微状态、预测和行动边界应如何变化？"
        )
        result.append(
            {
                "question_kind": "conditional_or_counterfactual",
                **cf_task_fields,
                "question": question,
                "goal": "追踪一个已声明事实字段如何改变关键机制状态，并给出新的可检验预测。",
                "constraints": list(constraints) + ["只允许使用已声明的反事实变化，不得同时改写其他事实"],
                "cross_domain_transfer": False,
                "transfer_rationale": None,
                "counterfactual_delta": {
                    "changes": [change],
                    "microconcept_state_changes": state_changes,
                },
            }
        )
    else:
        conditional_stems = (
            "若下一观察期{focus_phrase}继续增强而证据完整度不变，当前预测和行动阈值应如何调整？",
            "若后续记录显示{focus_phrase}持续两个观察期，而其他关键状态保持不变，应怎样更新解释与行动？",
            "如果{focus_phrase}在下一时点减弱但目标结果没有改善，哪一项解释应被降级？",
            "若独立复测确认{focus_phrase}方向不变，什么预测应继续成立，什么情况会推翻它？",
            "如果{domain_context}中的{focus_phrase}越过当前边界，最先需要更新的是预测、行动还是 abstain 条件？",
            "若{focus_phrase}只短暂变化后恢复，当前判断的时间尺度和验证窗口应怎样改变？",
            "如果新增证据提高了可复现性但没有改变{focus_phrase}，路由置信度与下一步核验应如何更新？",
            "若下一期{focus_phrase}与当前方向相反，哪个假设必须被重新审查？",
        )
        question = conditional_stems[(case_index + DOMAINS.index(domain)) % len(conditional_stems)].format(
            focus_phrase=MICROCONCEPT_PHRASES[focus], domain_context=domain_context
        )
        result.append(
            {
                "question_kind": "conditional_or_counterfactual",
                **task_fields("conditional_or_counterfactual"),
                "question": question,
                "goal": "给出条件预测，明确什么观察会维持、削弱或推翻当前机制解释。",
                "constraints": list(constraints) + ["不得把条件预测表述为已经发生的结果"],
                "cross_domain_transfer": False,
                "transfer_rationale": None,
                "counterfactual_delta": None,
            }
        )
    return result

def _pack(case_id: str, domain: str, sources: list[dict[str, Any]], facts: list[dict[str, Any]], uncertainties: list[dict[str, Any]], evidence: list[dict[str, Any]], states: list[dict[str, Any]], index: int, date_observed: str) -> dict[str, Any]:
    pack = {"case_id": case_id, "domain": domain, "as_of": AS_OF, "public_available_at": max((row["published_at"] for row in sources), default=AS_OF), "source_refs": sources, "facts": facts, "factual_uncertainties": uncertainties, "structured_evidence": evidence, "microconcept_states": states, "question_variants": _variants(index, domain, facts, states), "date_observed": date_observed, "provenance": "public-sourced", "verification_status": "sourced"}
    validate_fact_pack(pack); return pack

def build_cisa_packs(cache_dir: Path) -> list[dict[str, Any]]:
    payload = _json_source(CISA_URL, cache_dir / "cisa_kev.json")
    rows = [row for row in payload.get("vulnerabilities", []) if row.get("dateAdded", "") <= "2024-12-31" and row.get("cveID") and row.get("cwes")]
    rows.sort(key=lambda row: (row.get("dateAdded", ""), row["cveID"]), reverse=True)
    if len(rows) < 40: raise RuntimeError("CISA KEV has fewer than 40 eligible records")
    result = []
    for index, row in enumerate(rows[:40]):
        added = _iso_date(row.get("dateAdded")); due = _iso_date(row.get("dueDate"))
        days = max(0, (datetime.fromisoformat(due[:10])-datetime.fromisoformat(added[:10])).days)
        ransomware = row.get("knownRansomwareCampaignUse") == "Known"
        sources = [_source("src-cisa-kev", f"CISA KEV catalog entry {row['cveID']}", CISA_URL, _iso_date(payload.get("dateReleased", BUILD_DATE)), locator=row["cveID"])]
        facts = [
            _fact("fact-1", f"CISA 将记录 {row['cveID']} 纳入已知遭利用漏洞目录，加入日期为 {row.get('dateAdded')}。", ["src-cisa-kev"], added, value=row["cveID"]),
            _fact("fact-2", f"记录涉及一个公开产品漏洞，分类包含 {', '.join(row.get('cwes', []))}；目录要求在期限内采取缓解措施或停止使用。", ["src-cisa-kev"], added, cwes=row.get("cwes", [])),
            _fact("fact-3", f"目录给出的整改截止日为 {row.get('dueDate')}，从加入目录到截止日共 {days} 天。", ["src-cisa-kev"], added, deadline_days=days),
            _fact("fact-4", f"目录中的已知勒索软件活动字段为 {'Known' if ransomware else 'Unknown'}。", ["src-cisa-kev"], added, ransomware_known=ransomware),
        ]
        urgency = 3 if days <= 14 else 2 if days <= 30 else 1
        active = {
            "constraint_pressure": {"status":"supported","direction":"increasing","intensity":3,"temporal_persistence":2,"task_relevance":urgency,"evidence_refs":["fact-1","fact-3"]},
            "conversion_gap": {"status":"candidate","direction":"open","intensity":2,"temporal_persistence":2,"task_relevance":2 if days <= 30 else 1,"evidence_refs":["fact-2","fact-3"]},
            "countervailing_pressure": {"status":"candidate","direction":"mitigating","intensity":2,"temporal_persistence":1,"task_relevance":2,"evidence_refs":["fact-2"]},
            "evidence_quality": {"status":"supported","direction":"reviewed","intensity":2,"temporal_persistence":2,"task_relevance":2,"evidence_refs":["fact-1","fact-2"]},
            "outcome_asymmetry": {"status":"supported","direction":"downside_skewed","intensity":3 if ransomware else 2,"temporal_persistence":2,"task_relevance":3 if ransomware else 1,"evidence_refs":["fact-1","fact-4"]},
            "regime_proximity": {"status":"supported","direction":"nearer_boundary","intensity":3,"temporal_persistence":2,"task_relevance":max(2, urgency),"evidence_refs":["fact-1","fact-3"]},
            "temporal_persistence": {"status":"supported","direction":"bounded_window","intensity":2,"temporal_persistence":2,"task_relevance":3 if days <= 14 else 2 if days <= 30 else 1,"evidence_refs":["fact-3"]},
        }
        if ransomware: active["feedback_polarity"] = {"status":"candidate","direction":"amplifying","intensity":2,"temporal_persistence":2,"task_relevance":3,"evidence_refs":["fact-4"]}
        result.append(_pack(f"mqv1-se-{index+1:03d}-{row['cveID'].lower()}", "software_engineering", sources, facts, [{"uncertainty_id":"unc-1","statement":"目录确认存在已知利用，但不提供该匿名系统的暴露范围、补丁状态或实际损失。","evidence_refs":["fact-1","fact-2"]}], [{"evidence_id":"metric-1","kind":"remediation_window_days","value":days,"fact_refs":["fact-3"]}], _states(active), index, row.get("dateAdded", "unknown")))
    return result

def _number(value: Any) -> int | None:
    match = re.search(r"([\d,]+)", str(value or ""))
    return int(match.group(1).replace(",", "")) if match else None

def build_cpsc_packs(cache_dir: Path) -> list[dict[str, Any]]:
    payload = _json_source(CPSC_URL, cache_dir / "cpsc_recalls.json")
    eligible = []
    for row in payload:
        recall_date = str(row.get("RecallDate", ""))[:10]; published = str(row.get("LastPublishDate") or row.get("RecallDate") or "")[:10]
        if recall_date and recall_date <= "2024-12-31" and published <= "2026-07-20" and row.get("URL") and row.get("Hazards") and row.get("Remedies") and row.get("Products"): eligible.append(row)
    eligible.sort(key=lambda row: (str(row.get("RecallDate", "")), int(row.get("RecallID") or 0)), reverse=True)
    if len(eligible) < 40: raise RuntimeError("CPSC has fewer than 40 eligible complete records")
    result = []
    for index, row in enumerate(eligible[:40]):
        rid = str(row["RecallID"]); recall_date = str(row.get("RecallDate"))[:10]; published = _iso_date(str(row.get("LastPublishDate") or row.get("RecallDate"))[:10])
        units = sum(n for n in (_number(p.get("NumberOfUnits")) for p in row.get("Products", [])) if n is not None)
        hazards = sorted({_compact(x.get("Name"), 80) for x in row.get("Hazards", []) if x.get("Name")})
        remedies = sorted({_compact(x.get("Name"), 80) for x in row.get("Remedies", []) if x.get("Name")})
        options = sorted({_compact(x, 60) for x in row.get("RemedyOptions", []) if x})
        incidents = len(row.get("Incidents") or []); injuries = len(row.get("Injuries") or [])
        sources = [_source("src-cpsc-record", f"CPSC recall {row.get('RecallNumber') or rid}", row["URL"], published, locator=rid), _source("src-cpsc-api", "CPSC public recall REST record", CPSC_URL, published, locator=rid)]
        facts = [
            _fact("fact-1", f"CPSC 公开记录在 {recall_date} 发布一项产品召回，记录编号为 {row.get('RecallNumber') or rid}。", ["src-cpsc-record","src-cpsc-api"], published, recall_id=rid),
            _fact("fact-2", f"记录列出的主要危险类别为：{'; '.join(hazards[:4]) or '未细分'}。", ["src-cpsc-record","src-cpsc-api"], published, hazards=hazards),
            _fact("fact-3", f"记录列出的处置方式为：{'; '.join((remedies+options)[:5]) or '未细分'}。", ["src-cpsc-record","src-cpsc-api"], published, remedies=remedies+options),
            _fact("fact-4", f"公开记录覆盖约 {units if units else '未量化'} 个产品单位，并包含 {incidents} 条事故条目和 {injuries} 条伤害条目。", ["src-cpsc-api"], published, affected_units=units, incident_entries=incidents, injury_entries=injuries),
        ]
        severity = 3 if injuries else 2
        report_signal = 3 if injuries else 2 if incidents else 1
        exposure_signal = 3 if units >= 1_000_000 else 2 if units >= 10_000 else 1
        active = {
            "constraint_pressure":{"status":"supported","direction":"binding","intensity":severity,"temporal_persistence":2,"task_relevance":max(report_signal, 2),"evidence_refs":["fact-1","fact-2"]},
            "conversion_gap":{"status":"candidate","direction":"closure_unknown","intensity":2,"temporal_persistence":2,"task_relevance":2 if units else 1,"evidence_refs":["fact-3","fact-4"]},
            "countervailing_pressure":{"status":"supported","direction":"mitigating","intensity":2,"temporal_persistence":2,"task_relevance":2,"evidence_refs":["fact-2","fact-3"]},
            "evidence_quality":{"status":"mixed","direction":"partial","intensity":2,"temporal_persistence":2,"task_relevance":report_signal,"evidence_refs":["fact-1","fact-4"]},
            "outcome_asymmetry":{"status":"supported","direction":"downside_skewed","intensity":severity,"temporal_persistence":2,"task_relevance":report_signal,"evidence_refs":["fact-2","fact-4"]},
            "relative_gradient":{"status":"candidate","direction":"exposure_exceeds_observed_reports","intensity":2,"temporal_persistence":1,"task_relevance":exposure_signal,"evidence_refs":["fact-4"]},
            "resource_balance":{"status":"candidate","direction":"remedy_capacity_unmeasured","intensity":2,"temporal_persistence":2,"task_relevance":exposure_signal,"evidence_refs":["fact-3","fact-4"]},
            "temporal_persistence":{"status":"candidate","direction":"persists_until_remedy","intensity":2,"temporal_persistence":2,"task_relevance":2 if units else 1,"evidence_refs":["fact-1","fact-3"]},
        }
        if injuries: active["regime_proximity"] = {"status":"candidate","direction":"nearer_harm_boundary","intensity":2,"temporal_persistence":2,"task_relevance":2,"evidence_refs":["fact-2","fact-4"]}
        result.append(_pack(f"mqv1-po-{index+1:03d}-cpsc-{rid}", "product_organization", sources, facts, [{"uncertainty_id":"unc-1","statement":"公开召回记录不能证明全部受影响单位已被找到，也不能把报告条目数当作真实发生率。","evidence_refs":["fact-3","fact-4"]}], [{"evidence_id":"metric-1","kind":"affected_units","value":units,"fact_refs":["fact-4"]},{"evidence_id":"metric-2","kind":"reported_entry_count","value":incidents+injuries,"fact_refs":["fact-4"]}], _states(active), index, recall_date))
    return result

def _frame_map(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    result = {}
    for row in payload.get("data", []):
        key = (int(row["cik"]), str(row.get("end", "")))
        if key not in result or str(row.get("accn", "")) > str(result[key].get("accn", "")): result[key] = row
    return result

def build_sec_packs(cache_dir: Path) -> list[dict[str, Any]]:
    specs = {"assets":("Assets","CY2023Q4I"),"liabilities":("Liabilities","CY2023Q4I"),"cash":("CashAndCashEquivalentsAtCarryingValue","CY2023Q4I"),"revenue":("RevenueFromContractWithCustomerExcludingAssessedTax","CY2023"),"ocf":("NetCashProvidedByUsedInOperatingActivities","CY2023")}
    payloads = {key:_json_source(SEC_FRAME.format(tag=tag,period=period), cache_dir/f"sec_{tag}_{period}.json") for key,(tag,period) in specs.items()}
    maps = {key:_frame_map(payload) for key,payload in payloads.items()}; common = set.intersection(*(set(value) for value in maps.values()))
    candidates = []
    for key in common:
        values = {name:int(mapping[key]["val"]) for name,mapping in maps.items()}
        accessions = {mapping[key].get("accn") for mapping in maps.values()}
        if values["assets"] <= 0 or values["revenue"] < 10_000_000 or values["liabilities"] < 0 or values["cash"] < 0 or len(accessions) != 1: continue
        liab = values["liabilities"]/values["assets"]; cash = values["cash"]/max(values["liabilities"],1); conv = values["ocf"]/values["revenue"]
        if liab > 2 or cash > 10 or abs(conv) > 2: continue
        candidates.append((liab,cash,conv,key,values))
    candidates.sort(key=lambda item:(item[0],item[1],item[3][0]))
    if len(candidates) < 40: raise RuntimeError("SEC frames did not provide 40 joined records")
    positions = [round(i*(len(candidates)-1)/39) for i in range(40)]
    result = []
    for index, (liab,cash,conv,key,values) in enumerate(candidates[position] for position in positions):
        cik, period_end = key; sources=[]
        for source_index,(metric,(tag,period)) in enumerate(specs.items(),1):
            row=maps[metric][key]
            sources.append(_source(f"src-sec-{source_index}", f"SEC XBRL frame {tag} for CIK {cik}", SEC_FRAME.format(tag=tag,period=period), AS_OF, locator={"cik":cik,"end":period_end,"accession":row.get("accn")}))
        facts=[
            _fact("fact-1",f"SEC XBRL 记录显示，该匿名申报实体在 {period_end} 的资产为 {values['assets']} 美元，负债为 {values['liabilities']} 美元。",["src-sec-1","src-sec-2"],AS_OF,assets_usd=values["assets"],liabilities_usd=values["liabilities"]),
            _fact("fact-2",f"同一期间截点的现金及现金等价物为 {values['cash']} 美元，现金与负债比约为 {cash:.4f}。",["src-sec-2","src-sec-3"],AS_OF,cash_usd=values["cash"],cash_to_liabilities=round(cash,6)),
            _fact("fact-3",f"2023 年度收入为 {values['revenue']} 美元，经营活动现金流为 {values['ocf']} 美元，现金转换比约为 {conv:.4f}。",["src-sec-4","src-sec-5"],AS_OF,revenue_usd=values["revenue"],operating_cash_flow_usd=values["ocf"],ocf_to_revenue=round(conv,6)),
            _fact("fact-4",f"负债与资产比约为 {liab:.4f}；该比率与现金转换比来自公开申报数值的确定性计算。",["src-sec-1","src-sec-2","src-sec-4","src-sec-5"],AS_OF,liabilities_to_assets=round(liab,6)),
        ]
        intensity=3 if liab>=.85 or cash<.05 else 2
        liquidity_signal = 3 if cash < .05 else 2 if cash < .20 else 1
        leverage_signal = 3 if liab >= .90 else 2 if liab >= .70 else 1
        conversion_signal = 3 if conv < 0 else 2 if conv < .10 else 1
        active={
            "capacity_headroom":{"status":"candidate","direction":"limited" if cash<.15 else "available","intensity":intensity,"temporal_persistence":2,"task_relevance":liquidity_signal,"evidence_refs":["fact-1","fact-2"]},
            "constraint_pressure":{"status":"candidate","direction":"increasing" if liab>.7 else "moderate","intensity":intensity,"temporal_persistence":2,"task_relevance":leverage_signal,"evidence_refs":["fact-1","fact-4"]},
            "conversion_gap":{"status":"supported" if conv<.1 else "candidate","direction":"wide" if conv<.1 else "narrower","intensity":3 if conv<0 else 2,"temporal_persistence":1,"task_relevance":conversion_signal,"evidence_refs":["fact-3"]},
            "countervailing_pressure":{"status":"supported","direction":"assets_offset_liabilities","intensity":2,"temporal_persistence":2,"task_relevance":3 if liab < .50 else 2,"evidence_refs":["fact-1","fact-4"]},
            "evidence_quality":{"status":"supported","direction":"structured_public_filing","intensity":2,"temporal_persistence":2,"task_relevance":2,"evidence_refs":["fact-1","fact-2","fact-3"]},
            "outcome_asymmetry":{"status":"candidate","direction":"downside_skewed","intensity":intensity,"temporal_persistence":2,"task_relevance":max(liquidity_signal, leverage_signal),"evidence_refs":["fact-2","fact-4"]},
            "regime_proximity":{"status":"candidate","direction":"nearer_boundary" if liab>.9 else "not_established","intensity":intensity,"temporal_persistence":2,"task_relevance":leverage_signal,"evidence_refs":["fact-2","fact-4"]},
            "relative_gradient":{"status":"supported","direction":"ratio_differentiated","intensity":2,"temporal_persistence":1,"task_relevance":2 if abs(conv) >= .10 else 1,"evidence_refs":["fact-2","fact-3","fact-4"]},
            "resource_balance":{"status":"supported","direction":"accounted","intensity":3,"temporal_persistence":2,"task_relevance":max(liquidity_signal, conversion_signal),"evidence_refs":["fact-1","fact-2","fact-3"]},
        }
        result.append(_pack(f"mqv1-bf-{index+1:03d}-sec-{cik}","business_finance",sources,facts,[{"uncertainty_id":"unc-1","statement":"单期公开申报不能证明这些比率会持续，也不提供该实体未来融资条件或现金需求。","evidence_refs":["fact-1","fact-2","fact-3"]}],[{"evidence_id":"metric-1","kind":"liabilities_to_assets","value":round(liab,6),"fact_refs":["fact-4"]},{"evidence_id":"metric-2","kind":"cash_to_liabilities","value":round(cash,6),"fact_refs":["fact-2"]},{"evidence_id":"metric-3","kind":"operating_cash_conversion","value":round(conv,6),"fact_refs":["fact-3"]}],_states(active),index,period_end))
    return result

def build_usgs_packs(cache_dir: Path) -> list[dict[str, Any]]:
    params={"format":"geojson","starttime":"2023-01-01","endtime":"2023-12-31","minmagnitude":5.5,"orderby":"magnitude","limit":200}
    payload=_json_source(USGS_URL,cache_dir/"usgs_earthquakes_2023_m55.json",params=params)
    rows=[row for row in payload.get("features",[]) if row.get("id") and row.get("properties",{}).get("status")=="reviewed" and row.get("properties",{}).get("mag") is not None]
    rows.sort(key=lambda row:(-float(row["properties"]["mag"]),row["id"]))
    if len(rows)<40: raise RuntimeError("USGS has fewer than 40 reviewed events")
    result=[]
    for index,row in enumerate(rows[:40]):
        prop=row["properties"]; lon,lat,depth=row["geometry"]["coordinates"]; observed=datetime.fromtimestamp(prop["time"]/1000,timezone.utc).date().isoformat(); updated=min(_iso_ms(prop.get("updated")),AS_OF); mag=float(prop["mag"]); mmi=prop.get("mmi"); felt=int(prop.get("felt") or 0); nst=int(prop.get("nst") or 0); alert=prop.get("alert") or "not_assigned"; high=mag>=7 or (mmi or 0)>=7 or alert in {"red","orange"}
        sources=[_source("src-usgs-event",f"USGS reviewed event {row['id']}",prop["url"],updated,locator=row["id"]),_source("src-usgs-query","USGS FDSN event query",USGS_URL,AS_OF,locator=params)]
        facts=[
            _fact("fact-1",f"USGS 将事件 {row['id']} 标记为 reviewed；发生时间为 {observed}，震级为 {mag:.1f}，深度约 {depth:.1f} 千米。",["src-usgs-event","src-usgs-query"],updated,magnitude=mag,depth_km=round(depth,3)),
            _fact("fact-2",f"事件的仪器烈度为 {mmi if mmi is not None else '未给出'}，影响警报等级为 {alert}，显著性指标为 {prop.get('sig')}。",["src-usgs-event"],updated,mmi=mmi,alert=alert,significance=prop.get("sig")),
            _fact("fact-3",f"公众感知报告数为 {felt}，台站数为 {nst}，方位覆盖缺口为 {prop.get('gap')} 度，残差指标为 {prop.get('rms')}。",["src-usgs-event"],updated,felt_reports=felt,station_count=nst,azimuthal_gap=prop.get("gap"),rms=prop.get("rms")),
            _fact("fact-4",f"USGS 记录的海啸标志为 {int(bool(prop.get('tsunami')))}；事件坐标为纬度 {lat:.4f}、经度 {lon:.4f}。",["src-usgs-event"],updated,tsunami=int(bool(prop.get("tsunami"))),latitude=lat,longitude=lon),
        ]
        measurement_signal = 3 if nst >= 100 and (prop.get("gap") or 360) <= 90 else 2 if nst >= 30 else 1
        hazard_signal = 3 if high else 2 if mag >= 6.5 else 1
        active={
            "conversion_gap":{"status":"candidate","direction":"source_to_impact_varies","intensity":2,"temporal_persistence":1,"task_relevance":2 if mmi is not None else 1,"evidence_refs":["fact-1","fact-2"]},
            "evidence_quality":{"status":"supported","direction":"reviewed_measurement","intensity":3 if nst>=100 else 2,"temporal_persistence":2,"task_relevance":measurement_signal,"evidence_refs":["fact-1","fact-3"]},
            "outcome_asymmetry":{"status":"supported","direction":"tail_risk","intensity":3 if high else 2,"temporal_persistence":1,"task_relevance":hazard_signal,"evidence_refs":["fact-2","fact-4"]},
            "regime_proximity":{"status":"candidate","direction":"nearer_harm_boundary" if high else "uncertain","intensity":3 if high else 2,"temporal_persistence":1,"task_relevance":hazard_signal,"evidence_refs":["fact-1","fact-2"]},
            "relative_gradient":{"status":"candidate","direction":"spatial_response_varies","intensity":2,"temporal_persistence":1,"task_relevance":3 if felt >= 1000 else 2 if felt else 1,"evidence_refs":["fact-1","fact-2","fact-3"]},
            "resource_balance":{"status":"candidate","direction":"released_energy_proxy","intensity":2,"temporal_persistence":1,"task_relevance":3 if mag >= 7 else 2 if mag >= 6.5 else 1,"evidence_refs":["fact-1"]},
            "temporal_persistence":{"status":"candidate","direction":"after_effects_unmeasured","intensity":1,"temporal_persistence":1,"task_relevance":2 if high else 1,"evidence_refs":["fact-1","fact-2"]},
        }
        result.append(_pack(f"mqv1-sp-{index+1:03d}-usgs-{row['id']}","science_public_systems",sources,facts,[{"uncertainty_id":"unc-1","statement":"单个事件记录不包含完整暴露人口、基础设施脆弱性、次生灾害和恢复过程，不能由震级直接推出社会损失。","evidence_refs":["fact-1","fact-2","fact-3"]}],[{"evidence_id":"metric-1","kind":"magnitude","value":mag,"fact_refs":["fact-1"]},{"evidence_id":"metric-2","kind":"instrumental_intensity","value":mmi,"fact_refs":["fact-2"]},{"evidence_id":"metric-3","kind":"station_count","value":nst,"fact_refs":["fact-3"]}],_states(active),index,observed))
    return result

def build_all_fact_packs(runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> list[dict[str, Any]]:
    cache = runtime_root / "source-cache"
    packs = build_sec_packs(cache) + build_cisa_packs(cache) + build_cpsc_packs(cache) + build_usgs_packs(cache)
    if len(packs) != 160: raise RuntimeError(f"expected 160 packs, built {len(packs)}")
    counts=defaultdict(int)
    for pack in packs: counts[pack["domain"]]+=1; validate_fact_pack(pack)
    if any(counts[domain]!=40 for domain in DOMAINS): raise RuntimeError(f"domain quota mismatch: {dict(counts)}")
    return packs

def _case_markdown(pack: dict[str, Any]) -> str:
    domain_title={"business_finance":"Business","software_engineering":"Engineering","product_organization":"Product-Organization","science_public_systems":"Public-Systems"}[pack["domain"]]
    refs="\n".join(f"  - {json.dumps(source['url'],ensure_ascii=False)}" for source in pack["source_refs"])
    facts="\n".join(f"{i}. {fact['statement']}" for i,fact in enumerate(pack["facts"],1))
    uncertainties="\n".join(f"- {item['statement']}" for item in pack["factual_uncertainties"]) or "- 无额外记录。"
    sources="\n".join(f"- {source['title']}：{source['url']}（定位：{json.dumps(source.get('locator'),ensure_ascii=False)}）" for source in pack["source_refs"])
    return f'''---
type: case
case_name: {json.dumps(pack['case_id'],ensure_ascii=False)}
domain: {json.dumps(domain_title,ensure_ascii=False)}
created: {BUILD_DATE}
updated: {BUILD_DATE}
date_observed: {json.dumps(pack.get('date_observed','unknown'),ensure_ascii=False)}
as_of: {BUILD_DATE}
privacy: public
provenance: public-sourced
verification_status: sourced
public_case: true
source_refs:
{refs}
related_models: []
tags:
  - case
  - mechanism-question-v1
  - {pack['domain']}
---

# {pack['case_id']}

## 案例摘要

本卡是机制问题集 V1 的公开事实摘要。正文使用中性案例编号；真实机构或产品身份只保留在来源定位中，不作为路由捷径。

## 背景

该记录来自公开、结构化的官方数据源，用于训练“事实—微概念状态—机制选择”接口。案例只保存回答问题所需的最小事实，不复制原始页面全文。

## 时代与制度条件

- 本卡按 {AS_OF} 的公开数据结构与口径整理。数据源后续修订、报告覆盖率、制度要求和测量方法变化都可能改变可比性。
- 跨领域迁移只在压力、剩余空间、证据质量、转换差距或失效代价等结构可回引时成立；对象名称相似不构成证据。

## 关键事实

{facts}

## 事实、推断与待验证项

| 类型 | 内容 |
| --- | --- |
| 事实 | 上述关键事实均回引到本卡来源，并受资料截点约束。 |
| 整理者推断 | 这些事实可以支持或挑战若干可迁移机制状态，但不能单独证明任何正式思维模型为真。 |
| 待验证 | {uncertainties.replace(chr(10), '<br>')} |

## 涉及模型

- 不在公共案例卡中写入教师路由。模型选择标签隔离保存在外部 Weak 标签目录。

## 模型解释

本卡只提供事实与机制状态的可审计基础。具体主模型、辅助模型和近邻排除由路由阶段根据固定关系索引生成，不反向修改本卡事实。

## 失败机制或关键转折

当前公开记录只允许识别压力、边界、转换、证据或资源关系的候选结构；若关键状态为 unknown，必须保留未知，不能补写一个确定性失败故事。

## 耐久机制

可迁移部分是可观察变量之间的结构关系及其失效条件，而不是机构名称、产品类别或历史结果本身。

## 竞争性解释与反证

- 同一组事实可能同时支持多个相邻机制，需用产生不同预测的新证据区分。
- 比率、期限和严重度字段不是对机构质量、责任或未来结果的总评。
- 若来源修订、口径不可比或关键变量缺失，当前状态应降级或 abstain。

## 来源

{sources}

本卡只概括公开记录中的必要字段，不复制原始网页或报告全文；教师路由标签不写入公共案例库。

## 模型修改提案

暂不提出。该记录用于弱监督路由研究，不足以单独修改正式模型语义。
'''

def write_public_cases(packs: Iterable[dict[str, Any]], cases_root: Path = ROOT / "09-Cases") -> list[Path]:
    dirs={"business_finance":cases_root/"Business"/"Mechanism-Questions-V1","software_engineering":cases_root/"Engineering"/"Mechanism-Questions-V1","product_organization":cases_root/"Product-Organization"/"Mechanism-Questions-V1","science_public_systems":cases_root/"Public-Systems"/"Mechanism-Questions-V1"}
    written=[]
    for pack in packs:
        directory=dirs[pack["domain"]]; directory.mkdir(parents=True,exist_ok=True); path=directory/f"{_slug(pack['case_id'])}.md"; path.write_text(_case_markdown(pack),encoding="utf-8",newline="\n"); written.append(path)
    return written

def write_source_ledger(runtime_root: Path, packs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger=[{"case_id":pack["case_id"],"domain":pack["domain"],**source} for pack in packs for source in pack["source_refs"]]
    write_jsonl(runtime_root/"source_ledger.jsonl",ledger); return ledger

def source_cache_manifest(runtime_root: Path) -> list[dict[str, Any]]:
    rows=[]
    for path in sorted((runtime_root/"source-cache").glob("*")):
        if path.is_file():
            data=path.read_bytes(); rows.append({"file":path.name,"bytes":len(data),"sha256":_sha256(data)})
    return rows

