#### PATCHARAKORN ####

"""The indexed AF builder must agree with py_arg exactly.

    python -m unittest tests.test_fast_af -v      (from internship/ABA_app)
"""

from __future__ import annotations

import unittest

from py_arg.aba_classes.aba_framework import ABAF
from py_arg.aba_classes.rule import Rule
from py_arg.aba_classes.semantics import get_grounded_extensions as py_grounded

from app import fast_af
from app.aba_builder import build_aba_framework
from app.datasets import DATASETS, LA_VERANDA


def pyarg_view(framework: ABAF):
    af = framework.generate_af()
    arguments = {a.name: (frozenset(a.premise), a.conclusion) for a in af.arguments}
    defeats = {(d.from_argument.name, d.to_argument.name) for d in af.defeats}
    return arguments, defeats


class MatchesPyArg(unittest.TestCase):
    def assert_same(self, framework: ABAF) -> None:
        fast = fast_af.generate(framework)
        arguments, defeats = pyarg_view(framework)
        self.assertEqual(fast.arguments, arguments)
        self.assertEqual(fast.defeats, defeats)
        expected = next(iter(py_grounded.get_preferred_extensions(framework)))  # py_arg's name
        self.assertEqual(fast_af.grounded_assumptions(framework, fast), expected)

    def test_voted_topics(self):
        # The gold-workbook hotel; the per-topic-vote hotel is covered by test_voted_topics_all_hotels.
        for aspect in sorted(LA_VERANDA.aspect_gold_files):
            with self.subTest(aspect=aspect):
                self.assert_same(build_aba_framework(aspect).framework)

    def test_voted_topics_all_hotels(self):
        """Every hotel's voted topics agree between the fast path and py_arg's own generate_af."""
        for key, hotel in DATASETS.items():
            from app.contrary_sources import vote_topics
            for aspect in sorted(set(hotel.aspect_gold_files) | set(vote_topics(key))):
                with self.subTest(hotel=key, aspect=aspect):
                    self.assert_same(build_aba_framework(aspect, dataset=key).framework)

    def test_chains_underivable_atoms_and_reinstatement(self):
        # a is attacked by c (derivable through b and assumption d); d is attacked by e, which
        # has no derivation; f attacks itself through g; h is attacked by a derivable fact.
        framework = ABAF(
            assumptions={"a", "d", "f", "h"},
            rules={
                Rule("r1", {"b", "d"}, "c"), Rule("r2", set(), "b"),
                Rule("r3", {"x"}, "e"),                       # x never derivable
                Rule("r4", {"f"}, "g"), Rule("r5", set(), "k"),
            },
            language={"a", "b", "c", "d", "e", "f", "g", "h", "k", "x"},
            contraries={"a": "c", "d": "e", "f": "g", "h": "k"},
        )
        self.assert_same(framework)
        grounded = fast_af.grounded_assumptions(framework, fast_af.generate(framework))
        self.assertEqual(grounded, frozenset({"d"}))   # a and h attacked; f self-defeating

    def test_refuses_cyclic_rules(self):
        framework = ABAF(
            assumptions={"a"},
            rules={Rule("r1", {"p"}, "q"), Rule("r2", {"q"}, "p")},
            language={"a", "p", "q"},
            contraries={"a": "p"},
        )
        with self.assertRaises(fast_af.CyclicRules):
            fast_af.generate(framework)


if __name__ == "__main__":
    unittest.main()
