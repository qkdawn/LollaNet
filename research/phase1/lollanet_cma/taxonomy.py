from __future__ import annotations


MODEL_FAMILIES = (
    "cm-associational",
    "cm-interventional",
    "cm-counterfactual",
    "cm-mediation",
)

QUERY_TO_FAMILY = {
    "marginal": "cm-associational",
    "correlation": "cm-associational",
    "exp_away": "cm-associational",
    "ate": "cm-interventional",
    "backadj": "cm-interventional",
    "collider_bias": "cm-interventional",
    "ett": "cm-counterfactual",
    "det-counterfactual": "cm-counterfactual",
    "nde": "cm-mediation",
    "nie": "cm-mediation",
}


def model_family(query_type: str) -> str:
    try:
        return QUERY_TO_FAMILY[query_type]
    except KeyError as exc:
        raise ValueError(f"unsupported CLADDER query_type: {query_type}") from exc

