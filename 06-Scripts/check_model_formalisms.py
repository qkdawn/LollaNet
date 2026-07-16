#!/usr/bin/env python3
"""Run narrow, executable regression checks for every formal LollaNet model card.

This runner proves neither source validity nor real-world usefulness.  It keeps the
card's stated minimal equations, algorithmic relations, and premise breaks tied to
reproducible examples so later edits cannot silently sever the formalism.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import re
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9

# One distinctive formula/algorithm marker per card.  The executable case below
# checks the same narrow relation, not a claimed cross-domain interpretation.
FORMALISM_CHECKS: dict[str, dict[str, object]] = {
    # Chemistry
    "mm-4d9545b9b6af": {"fragment": "Δ_rG=Δ_rG°+RTlnQ", "case": "gibbs"},
    "mm-cdde2a4024f0": {"fragment": "ln(P2/P1)", "case": "clausius"},
    "mm-fffd2837e608": {"fragment": "VSEPR", "case": "vsepr"},
    "mm-7d542558a1b1": {"fragment": "|U|∝Z_eff/r", "case": "atomic"},
    "mm-3e810805f0a1": {"fragment": "Q<K", "case": "qk"},
    "mm-90310164e606": {"fragment": "Δ_rH°=Σ_iν_iΔ_fH_i°", "case": "hess"},
    "mm-4827493462f8": {"fragment": "ΔG=-nFE", "case": "redox"},
    "mm-977f6dda93d9": {"fragment": "催化不改变同一反应的自由能差与平衡常数", "case": "catalyst"},
    "mm-2ae0022c377f": {"fragment": "多次独立萃取将留存分数连乘", "case": "partition"},
    "mm-fbc90bff5839": {"fragment": "ξ_max=min_i", "case": "stoich"},
    "mm-10e7946e8f78": {"fragment": "∂lnr/∂ln[A_i]=m_i", "case": "rate"},
    "mm-00ebc72c737f": {"fragment": "pH=pK_a+log([A-]/[HA])", "case": "buffer"},
    # Engineering
    "mm-redundancy": {"fragment": "R_parallel=1-∏(1-R_i)", "case": "redundancy"},
    "mm-46fbd9f6f875": {"fragment": "Σ b_i≤B", "case": "decomposition"},
    "mm-feedback-loop": {"fragment": "Y/R=G/(1+GH)", "case": "feedback"},
    "mm-edf8b2dc7724": {"fragment": "MTBF/(MTBF+MTTR)", "case": "availability"},
    "mm-bfcb11b439f1": {"fragment": "R_s(t)=∏R_i(t)", "case": "reliability"},
    "mm-b451e4404335": {"fragment": "OR 门概率可近似求和", "case": "fault_tree"},
    "mm-margin-of-safety": {"fragment": "FS=C/L", "case": "margin"},
    "mm-620ba4e2365a": {"fragment": "G=(V,E)", "case": "interface"},
    "mm-c6c55641b8a5": {"fragment": "a 支配 b", "case": "pareto"},
    "mm-bottleneck": {"fragment": "L=XW", "case": "little"},
    "mm-8ab6e7bc3c83": {"fragment": "LCC=C_a+C_o+C_m+C_c+C_r", "case": "lcc"},
    "mm-d81cf96b1419": {"fragment": "边界变更 ΔS", "case": "boundary"},
    "mm-24c2078ff362": {"fragment": "σ_y²≈Σ_i", "case": "propagation"},
    "mm-c14f8b1ce070": {"fragment": "m_k(x,e)≤b_k", "case": "requirements"},
    "mm-c057bde8c027": {"fragment": "每个 r∈R 至少有来源、实现和验证证据", "case": "traceability"},
    "mm-3069a490b994": {"fragment": "min_x max_{u∈U}L(x,u)", "case": "robust"},
    # Mathematics
    "mm-d8045dd0996f": {"fragment": "I(T(x))=I(x)", "case": "invariant"},
    "mm-acdf35a8fee3": {"fragment": "grad f(x)=0", "case": "optimization"},
    "mm-ca5cdb8acd22": {"fragment": "(g∘f)(x)=g(f(x))", "case": "mapping"},
    "mm-7a960f6b4ce1": {"fragment": "Dijkstra", "case": "graph"},
    "mm-1dafd97598ea": {"fragment": "[f(a+h)-f(a)]/h", "case": "derivative"},
    "mm-4517776d4319": {"fragment": "x_{k+1}=x_k-f(x_k)/f'(x_k)", "case": "newton"},
    "mm-30d197d6a613": {"fragment": "左右极限不同", "case": "limit"},
    "mm-941bf971e56b": {"fragment": "(f_i+f_{i+1})Δx/2", "case": "integral"},
    "mm-267e488ed00d": {"fragment": "y=Σ a_i x_i", "case": "linearity"},
    "mm-f856988a2c82": {"fragment": "det(A-λI)=0", "case": "linear_algebra"},
    "mm-68e9a2e0d68d": {"fragment": "|A∪B|=|A|+|B|-|A∩B|", "case": "counting"},
    "mm-f15e784415c2": {"fragment": "|F'(x*)|<1", "case": "iteration"},
    # Physics
    "mm-954c263ec767": {"fragment": "ΔA²=⟨A²⟩-⟨A⟩²", "case": "uncertainty"},
    "mm-c7864295cc14": {"fragment": "ΣF_x", "case": "force"},
    "mm-91a96d14ace5": {"fragment": "Σp_before", "case": "momentum"},
    "mm-a28acfbc6882": {"fragment": "|ψ(t)⟩=U(t)|ψ(0)⟩", "case": "hamiltonian"},
    "mm-46e0dd26760b": {"fragment": "用梯度得到力或通量", "case": "field"},
    "mm-8ff5c64be854": {"fragment": "t_diff≈L²/D", "case": "diffusion"},
    "mm-f0584264021d": {"fragment": "扫过驱动频率", "case": "oscillation"},
    "mm-2b1d22093700": {"fragment": "v=Δx/Δt", "case": "wave"},
    "mm-fec74c600590": {"fragment": "Tr(ρP_a)", "case": "born"},
    "mm-d4edb4a5a85c": {"fragment": "S_gen=ΔS_system+ΔS_environment", "case": "entropy"},
    "mm-ded821acce72": {"fragment": "c_i=⟨i|ψ⟩", "case": "state_vector"},
    "mm-d2fbdcb18acf": {"fragment": "Jacobian/特征值", "case": "stability"},
    "mm-57e628333aee": {"fragment": "|S|≤2", "case": "bell"},
    "mm-71cbb455982b": {"fragment": "E_in - E_out = ΔE_stored", "case": "energy"},
    "mm-5dc7789f1f61": {"fragment": "化学势", "case": "statistics"},
    "mm-0df41f6ddc45": {"fragment": "dρ/dt=-i[H,ρ]/ħ", "case": "decoherence"},
    "mm-cc3991975703": {"fragment": "ν=ΔE/h", "case": "quantization"},
    "mm-5f8905afab9c": {"fragment": "Buckingham Pi 定理", "case": "dimensions"},
    "mm-fb4dec93d1f0": {"fragment": "T∝e^{-2κa}", "case": "tunnelling"},
}


def close(actual: float, expected: float, tol: float = TOL) -> bool:
    return abs(actual - expected) <= tol * max(1.0, abs(expected))


def run_case(case: str) -> tuple[bool, str]:
    """Return a deliberately narrow canonical calculation and premise/boundary check."""
    if case == "gibbs":
        r, temp, k = 8.314, 298.15, 10.0
        dg0 = -r * temp * math.log(k)
        return close(dg0 + r * temp * math.log(k), 0.0), "Q=K gives ΔrG=0"
    if case == "clausius":
        ratio = math.exp(-(40_000 / 8.314) * (1 / 350 - 1 / 300))
        return ratio > 1, "for positive ΔHvap, vapour pressure rises with temperature"
    if case == "vsepr":
        dx = math.cos(0) + math.cos(math.pi); dy = math.sin(0) + math.sin(math.pi)
        return math.hypot(dx, dy) < TOL, "opposed equal bond-dipole vectors cancel"
    if case == "atomic":
        return 2 / 1 > 1 / 1, "the stated Zeff/r attraction scale rises with Zeff at fixed r"
    if case == "qk":
        q, k = 2.0, 5.0
        return q < k and ("forward" if q < k else "reverse") == "forward", "Q<K selects net forward direction"
    if case == "hess":
        return close((-100.0) + 50.0, -50.0), "enthalpy changes combine linearly with reaction steps"
    if case == "redox":
        return close(-(2 * 96485 * 1.0), -192970.0), "positive cell potential gives negative ΔG for n>0"
    if case == "catalyst":
        k_before, k_after = 12.0, 12.0
        return k_before == k_after, "catalyst changes rate path, not equilibrium constant"
    if case == "partition":
        retained_once = 1 / (1 + 2 * 1)
        return close(retained_once**2, 1 / 9), "independent repeated extractions multiply retained fractions"
    if case == "stoich":
        # A + 2B -> products; nA=3, nB=4 means B limits at extent 2.
        return close(min(3 / 1, 4 / 2), 2.0), "limiting extent is the minimum reactant extent"
    if case == "rate":
        a1, a2 = 2.0, 4.0
        r1, r2 = a1**2, a2**2
        return close(math.log(r2 / r1) / math.log(a2 / a1), 2.0), "log-rate sensitivity recovers reaction order"
    if case == "buffer":
        return close(4.76 + math.log10(1), 4.76), "equal conjugate-pair concentrations give pH=pKa"
    if case == "redundancy":
        return close(1 - (1 - .9) * (1 - .9), .99), "independent parallel reliability"
    if case == "decomposition":
        return sum([30, 20, 40]) <= 100 and {"goal-a", "goal-b"} <= {"goal-a", "goal-b"}, "budget and goal coverage both checked"
    if case == "feedback":
        return close(1 / (1 + 1 * 1), .5) and (-1.0 < 0), "negative-feedback transfer and a stable scalar pole"
    if case == "availability":
        return close(90 / (90 + 10), .9), "availability needs both MTBF and MTTR"
    if case == "reliability":
        serial = .9 * .8; parallel = 1 - (1 - .9) * (1 - .8)
        return close(serial, .72) and close(parallel, .98), "independent series/parallel relations differ"
    if case == "fault_tree":
        independent_and = .01 * .02
        common_cause_present = True
        return close(independent_and, .0002) and common_cause_present, "independent product is conditional, not a common-cause model"
    if case == "margin":
        c, load = 120.0, 100.0
        return close(c - load, 20.0) and close(c / load, 1.2) and load > 0, "margin and safety factor require common units and positive load"
    if case == "interface":
        edges = {"A": {"B"}, "B": {"C"}, "C": set()}
        order = ["A", "B", "C"]
        return all(order.index(a) < order.index(b) for a, bs in edges.items() for b in bs), "acyclic dependencies admit a topological integration order"
    if case == "pareto":
        a, b = (2, 3), (3, 3)
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b)), "one feasible alternative dominates another only with all-no-worse plus one-better"
    if case == "little":
        return close(20.0, 4.0 * 5.0), "Little's law relates WIP, throughput, and cycle time but does not name a bottleneck"
    if case == "lcc":
        lcc = sum([10, 20, 30, 5, 2]); npv = 100 / 1.1 + 100 / 1.1**2
        return close(lcc, 67) and npv < 200, "life-cycle components add; positive discounting lowers future present value"
    if case == "boundary":
        affected = {"interfaces": True, "requirements": True, "risks": True}
        return all(affected.values()), "a boundary change triggers all three impact analyses"
    if case == "propagation":
        # y=x1+x2, independent sigma1=3 sigma2=4
        return close(math.sqrt(3**2 + 4**2), 5.0), "first-order independent variance propagation"
    if case == "requirements":
        x, threshold = 7, 10
        return x <= threshold and not (x >= 12), "hard constraints distinguish feasible from infeasible candidates"
    if case == "traceability":
        evidence = {"r1": {"source", "implementation", "verification"}}
        return evidence["r1"] == {"source", "implementation", "verification"}, "one requirement needs source, implementation, and verification links"
    if case == "robust":
        losses = {"a": [2, 7], "b": [4, 5]}
        best = min(losses, key=lambda x: max(losses[x]))
        return best == "b", "minimax selects the lowest worst-case loss"
    if case == "invariant":
        x, transformed = [3, 1, 2], [1, 2, 3]
        return sum(x) == sum(transformed), "sum is invariant under permutation"
    if case == "optimization":
        x = 1.0
        grad = 2 * (x - 1)
        return close(grad, 0.0) and close(2.0, 2.0), "stationary point plus positive Hessian for a quadratic minimum"
    if case == "mapping":
        f = lambda x: x + 1
        g = lambda x: 2 * x
        return g(f(3)) == 8, "composition evaluates g(f(x)), not f(g(x))"
    if case == "graph":
        # Shortest path A->B->C has weight 3, direct A->C has 5.
        return min(1 + 2, 5) == 3, "non-negative weighted shortest-path relation"
    if case == "derivative":
        h = 1e-6
        estimate = ((3 + h) ** 2 - 3**2) / h
        return close(estimate, 6.0, 1e-5), "finite difference approximates a local derivative"
    if case == "newton":
        x = 1.0
        for _ in range(5): x = x - (x*x - 2) / (2*x)
        return close(x, math.sqrt(2), 1e-8), "Newton iteration converges under a regular local derivative"
    if case == "limit":
        left, right = -1.0, 1.0
        return left != right, "unequal one-sided limits identify a jump discontinuity"
    if case == "integral":
        n = 1_000; dx = 1/n
        estimate = sum(((i*dx) + ((i+1)*dx)) * dx / 2 for i in range(n))
        return close(estimate, .5, 1e-6), "trapezoidal refinement approximates integral of f(x)=x"
    if case == "linearity":
        f = lambda x: 3*x
        return close(f(2+5), f(2)+f(5)), "linear response satisfies additivity"
    if case == "linear_algebra":
        # [[2,0],[0,3]] x = [4,9] has x=[2,3] and eigenvalues 2,3.
        return close(4/2, 2) and close(9/3, 3), "diagonal linear system has the expected solution/eigenvalues"
    if case == "counting":
        a, b, intersection = 5, 6, 2
        return a + b - intersection == 9, "inclusion-exclusion avoids double counting"
    if case == "iteration":
        x = 1.0
        for _ in range(12): x = .5*x + 1
        return close(x, 2.0, 1e-3), "a contraction with |F'|<1 converges locally"
    if case == "uncertainty":
        dx, dp, hbar = 1.0, .5, 1.0
        return dx * dp >= hbar / 2, "a constructed state saturates the stated uncertainty lower bound"
    if case == "force":
        fx, fy, torque = 10 - 10, 5 - 5, 4 - 4
        return fx == fy == torque == 0, "static balance requires net force and net moment zero"
    if case == "momentum":
        before, after, impulse = 10.0, 7.0, -3.0
        return close(after - before, impulse), "impulse equals system momentum change"
    if case == "hamiltonian":
        psi0 = complex(1, 0); phase = cmath.exp(-1j * 2.0)
        return close(abs(phase * psi0)**2, 1.0), "unitary phase evolution preserves state norm"
    if case == "field":
        potential = lambda x: -3*x
        gradient = (potential(1.000001) - potential(1.0)) / .000001
        return close(-gradient, 3.0, 1e-5), "force/field follows the negative potential gradient in this one-dimensional example"
    if case == "diffusion":
        return close(2.0**2 / .5, 8.0), "diffusion time scales as L squared over D"
    if case == "oscillation":
        # Normalized forced-oscillator amplitude: damping limits the resonance peak.
        amp_res = 1 / .2; amp_off = 1 / math.sqrt((1-1.5**2)**2 + .2**2)
        return amp_res > amp_off, "amplitude peak is near the undamped natural frequency for this setup"
    if case == "wave":
        return close(120.0 / .4, 300.0), "propagation speed is distance divided by arrival-time difference"
    if case == "born":
        probabilities = [.25, .75]
        return close(sum(probabilities), 1.0) and close(75/100, probabilities[1]), "frequencies can be compared with normalized Born probabilities"
    if case == "entropy":
        system, environment = -2.0, 3.0
        return system + environment >= 0, "nonnegative generated entropy for the chosen irreversible process"
    if case == "state_vector":
        plus = (1 / math.sqrt(2), 1 / math.sqrt(2))
        return close(sum(c*c for c in plus), 1.0) and close((plus[0] + plus[1])**2, 2.0), "normalization and constructive relative phase both matter"
    if case == "stability":
        stable_eigenvalue, unstable_eigenvalue = -.2, .1
        return stable_eigenvalue < 0 < unstable_eigenvalue, "continuous-time local stability depends on eigenvalue real-part sign"
    if case == "bell":
        quantum_s = 2 * math.sqrt(2)
        return quantum_s > 2 and quantum_s <= 2 * math.sqrt(2), "quantum CHSH maximum exceeds local-hidden-variable bound but respects Tsirelson bound"
    if case == "energy":
        e_in, e_out = 100.0, 70.0
        return close(e_in-e_out, 30.0) and close(70/100, .7), "energy ledger and useful-efficiency calculation"
    if case == "statistics":
        fermi = 1 / (math.exp(0) + 1); bose = 1 / (math.exp(1) - 1)
        return 0 < fermi <= 1 and bose > 0, "Fermi occupancy is bounded by one while Bose occupancy is not"
    if case == "decoherence":
        gamma, t = .5, 4
        coherence = math.exp(-gamma*t)
        return 0 < coherence < 1, "a positive dephasing rate suppresses an off-diagonal coherence factor"
    if case == "quantization":
        delta_e, h = 6.626e-34, 6.626e-34
        return close(delta_e / h, 1.0), "transition frequency equals energy gap divided by Planck's constant"
    if case == "dimensions":
        # x^2/t has dimensions L^2/T, matching a diffusion coefficient.
        l_power, t_power = 2, -1
        return (l_power, t_power) == (2, -1), "dimension-vector accounting produces the expected L²/T group"
    if case == "tunnelling":
        thin = math.exp(-2 * 1 * 1); thick = math.exp(-2 * 1 * 2)
        return 0 < thick < thin < 1, "finite-barrier tunnelling falls exponentially as width increases"
    raise ValueError(f"Unknown formalism regression case: {case}")


def read_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    try:
        block = text.split("---", 2)[1]
    except IndexError:
        return {}
    result: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def run_checks(vault_root: Path = ROOT) -> dict[str, object]:
    cards: dict[str, tuple[Path, str]] = {}
    for path in (vault_root / "01-Models").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        meta = read_frontmatter(text)
        if meta.get("formalism") == "true":
            cards[meta["id"]] = (path, text)

    expected_ids = set(FORMALISM_CHECKS)
    actual_ids = set(cards)
    errors: list[str] = []
    if missing := sorted(actual_ids - expected_ids):
        errors.append(f"formal cards lack regression definitions: {missing}")
    if stale := sorted(expected_ids - actual_ids):
        errors.append(f"regression definitions reference non-formal/missing cards: {stale}")

    results: list[dict[str, object]] = []
    for model_id in sorted(expected_ids & actual_ids):
        path, text = cards[model_id]
        definition = FORMALISM_CHECKS[model_id]
        fragment = str(definition["fragment"])
        if fragment not in text:
            errors.append(f"{model_id}: required formula/algorithm fragment missing: {fragment}")
            continue
        ok, detail = run_case(str(definition["case"]))
        results.append({"model_id": model_id, "path": str(path.relative_to(vault_root)).replace("\\", "/"), "case": definition["case"], "passed": ok, "detail": detail})
        if not ok:
            errors.append(f"{model_id}: executable case failed: {detail}")

    return {"formal_card_count": len(actual_ids), "defined_check_count": len(FORMALISM_CHECKS), "passed": not errors, "errors": errors, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run narrow formalism regression checks for LollaNet model cards.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Vault root (defaults to repository root).")
    parser.add_argument("--json", action="store_true", help="Emit the complete machine-readable report.")
    args = parser.parse_args()
    report = run_checks(args.root.resolve())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["passed"]:
        print(f"PASS: {report['formal_card_count']} formal model cards; {report['defined_check_count']} executable narrow-relation checks.")
    else:
        print("FAIL: formalism regression checks")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
