#### PATCHARAKORN ####

"""
Indexed re-implementation of py_arg's `ABAF.generate_af()` and grounded semantics, for the
frameworks too large for py_arg's own loops.

Why: py_arg builds defeats by comparing every argument with every other argument
(py_arg/aba_classes/aba_framework.py, generate_af), and computes grounded semantics as a naive
least fixed point that re-checks every argument on every pass. Both are fine for the voted
topics (<= ~2,000 arguments) and hopeless for the LLM-only ones: a synthetic framework the size
of `food` (~12,800 arguments) already took 32 s + 119 s, and `room` is several times larger.

What is kept identical, deliberately:
  * the same arguments — one `{a} |- a` per assumption, plus one argument per premise set of
    every assumption's contrary, built by the same backward chaining and the same
    cross-product merge (a non-assumption body atom with no derivation kills the rule);
  * the same argument names (py_arg's InstantiatedArgument format), so the AA view and the
    extensions read exactly as py_arg's would;
  * the same defeats — arg2 defeats arg1 when arg2 concludes the contrary of one of arg1's
    premises;
  * the grounded extension, via the standard labelling algorithm (IN when every attacker is
    OUT, OUT when some attacker is IN), which computes the same least fixed point.
The only change is indexing (rules by head, arguments by conclusion, memoised premise sets),
which is exact for acyclic rule sets — the builder only produces those, and `generate` refuses
anything else so a cyclic framework falls back to py_arg. `tests/test_fast_af.py` checks the
result against py_arg on every human-voted topic.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from py_arg.aba_classes.aba_framework import ABAF


def argument_name(premise, conclusion: str) -> str:
    """py_arg's InstantiatedArgument name: sorted premise atoms, then the conclusion."""
    return ", ".join(str(atom) for atom in sorted(premise)) + " ⊢ " + str(conclusion)


@dataclass(frozen=True)
class FastAF:
    arguments: dict[str, tuple[frozenset[str], str]]   # name -> (premise, conclusion)
    defeats: set[tuple[str, str]]                       # (attacker name, attacked name)


class CyclicRules(ValueError):
    """The memoised derivation is only exact for acyclic rules."""


def _check_acyclic(rules_by_head: dict[str, list]) -> None:
    state: dict[str, int] = {}          # 1 = on the current path, 2 = done
    for start in rules_by_head:
        if state.get(start):
            continue
        stack = [(start, iter(atom for r in rules_by_head[start] for atom in r.body))]
        state[start] = 1
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                state[node] = 2
                stack.pop()
            elif child in rules_by_head:
                if state.get(child) == 1:
                    raise CyclicRules(f"rule cycle through '{child}'")
                if not state.get(child):
                    state[child] = 1
                    stack.append((child, iter(a for r in rules_by_head[child] for a in r.body)))


def generate(framework: ABAF) -> FastAF:
    assumptions = framework.assumptions
    rules_by_head: dict[str, list] = defaultdict(list)
    for rule in framework.rules:
        rules_by_head[rule.head].append(rule)
    _check_acyclic(rules_by_head)

    memo: dict[str, set[frozenset[str]]] = {}

    def premise_sets(target: str) -> set[frozenset[str]]:
        # Iterative post-order so deep rule chains cannot hit the recursion limit.
        if target in memo:
            return memo[target]
        stack = [target]
        while stack:
            atom = stack[-1]
            pending = [b for r in rules_by_head.get(atom, ()) for b in r.body
                       if b not in assumptions and b not in memo]
            if pending:
                stack.extend(pending)
                continue
            stack.pop()
            if atom in memo:
                continue
            out: set[frozenset[str]] = set()
            for rule in rules_by_head.get(atom, ()):
                sets = {frozenset(b for b in rule.body if b in assumptions)}
                for b in rule.body:
                    if b not in assumptions:
                        sets = {s1 | s2 for s1 in sets for s2 in memo[b]}  # empty -> rule dies
                out |= sets
            memo[atom] = out
        return memo[target]

    arguments: dict[str, tuple[frozenset[str], str]] = {}
    for assumption in assumptions:
        premise = frozenset({assumption})
        arguments[argument_name(premise, assumption)] = (premise, assumption)
    for assumption in assumptions:
        contrary = framework.contraries[assumption]
        for premise in premise_sets(contrary):
            arguments.setdefault(argument_name(premise, contrary), (premise, contrary))

    by_conclusion: dict[str, list[str]] = defaultdict(list)
    for name, (_, conclusion) in arguments.items():
        by_conclusion[conclusion].append(name)

    defeats: set[tuple[str, str]] = set()
    for name, (premise, _) in arguments.items():
        for atom in premise:
            for attacker in by_conclusion.get(framework.contraries[atom], ()):
                defeats.add((attacker, name))
    return FastAF(arguments=arguments, defeats=defeats)


def grounded_arguments(af: FastAF) -> set[str]:
    attackers: dict[str, set[str]] = defaultdict(set)
    attacks: dict[str, set[str]] = defaultdict(set)
    for src, dst in af.defeats:
        attackers[dst].add(src)
        attacks[src].add(dst)

    live = {name: len(attackers[name]) for name in af.arguments}   # attackers not yet OUT
    queue = deque(name for name, n in live.items() if n == 0)
    accepted: set[str] = set()
    rejected: set[str] = set()
    while queue:
        arg = queue.popleft()
        if arg in accepted or arg in rejected:
            continue
        accepted.add(arg)
        for victim in attacks[arg]:
            if victim in rejected:
                continue
            rejected.add(victim)
            for target in attacks[victim]:
                live[target] -= 1
                if live[target] == 0 and target not in rejected:
                    queue.append(target)
    return accepted


def grounded_assumptions(framework: ABAF, af: FastAF) -> frozenset[str]:
    """ABA grounded extension the way py_arg reports it: the accepted arguments' conclusions
    that are assumptions."""
    return frozenset(af.arguments[name][1] for name in grounded_arguments(af)
                     if af.arguments[name][1] in framework.assumptions)
