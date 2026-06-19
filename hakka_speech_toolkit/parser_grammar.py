"""X-bar-inspired phrase and dependency rules for Hakka parser output.

The grammar layer is deliberately small and inspectable. It does not claim to
replace hand annotation; instead, it provides a rule baseline that can be
evaluated against a 100-300 sentence gold sample and refined rule by rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import ParseToken


GRAMMAR_REFERENCES = [
    {
        "key": "chomsky_1970_xbar",
        "citation": (
            "Chomsky, N. (1970). Remarks on nominalization. In R. Jacobs & "
            "P. Rosenbaum (Eds.), Readings in English Transformational Grammar."
        ),
        "supports": "X-bar assumption that phrases are projections of lexical or functional heads.",
    },
    {
        "key": "jackendoff_1977_xbar",
        "citation": "Jackendoff, R. (1977). X-bar Syntax: A Study of Phrase Structure. MIT Press.",
        "supports": "Head-complement-adjunct organization used by the phrase builder.",
    },
    {
        "key": "ud_2021",
        "citation": (
            "de Marneffe, M.-C., Manning, C. D., Nivre, J., & Zeman, D. "
            "(2021). Universal Dependencies. Computational Linguistics, 47(2), 255-308."
        ),
        "supports": "Dependency labels such as nsubj, obj, advmod, case, clf, and punct.",
        "doi": "10.1162/coli_a_00402",
    },
]


@dataclass
class XBarNode:
    """A lightweight X-bar projection node."""

    label: str
    head: Optional[str] = None
    role: str = "projection"
    token: Optional[Dict[str, Any]] = None
    children: List["XBarNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this phrase node as a JSON-ready tree dictionary."""

        result = {
            "label": self.label,
            "head": self.head,
            "role": self.role,
        }
        if self.token is not None:
            result["token"] = self.token
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result


@dataclass(frozen=True)
class DependencyArc:
    head: int
    dependent: int
    relation: str
    rule: str

    def to_dict(self, tokens: Sequence["ParseToken"]) -> Dict[str, Any]:
        """Return this dependency arc with readable token text."""

        return {
            "head": self.head,
            "head_text": "ROOT" if self.head == -1 else tokens[self.head].text,
            "dependent": self.dependent,
            "dependent_text": tokens[self.dependent].text,
            "relation": self.relation,
            "rule": self.rule,
        }


class HakkaXBarGrammar:
    """Rule-based grammar over POS-tagged Hakka tokens."""

    NOUN_POS = {"ENTITY_noun", "ENTITY_person", "ENTITY_pronoun", "ENTITY_oov", "LOCATION"}
    VERB_POS = {"ACTION_verb", "ACTION_lightVerb", "ACTION_quantifiedVerb"}
    MODIFIER_POS = {"MODIFIER", "MODIFIER_color", "FUNC_degreeHead"}

    def analyze(self, tokens: Iterable["ParseToken"]) -> Dict[str, Any]:
        """Build X-bar and dependency analyses from POS-tagged tokens."""

        content = [token for token in tokens if token.pos != "PUNCTUATION"]
        if not content:
            return {
                "xbar_tree": None,
                "dependencies": [],
                "grammar_rules_applied": [],
                "annotation_framework": self.annotation_framework(),
                "grammar_references": GRAMMAR_REFERENCES,
            }

        rules = []
        dependencies: List[DependencyArc] = []
        predicate_index = self._find_main_predicate(content)
        if predicate_index is None:
            root_index = self._find_nominal_head(content)
            dependencies.append(DependencyArc(-1, root_index, "root", "nominal_fragment_as_root"))
            rules.append("nominal_fragment_as_root")
            root = self._nominal_projection(content, root_index, rules, dependencies)
        else:
            root = self._build_clause_projection(content, predicate_index, rules, dependencies)

        return {
            "xbar_tree": root.to_dict(),
            "dependencies": [arc.to_dict(content) for arc in dependencies],
            "grammar_rules_applied": rules,
            "annotation_framework": self.annotation_framework(),
            "grammar_references": GRAMMAR_REFERENCES,
        }

    @classmethod
    def annotation_framework(cls) -> Dict[str, Any]:
        """Describe the annotation conventions used by this grammar layer."""

        return {
            "recommended_gold_sample": "Hand-annotate 100-300 sentences before claiming syntactic coverage.",
            "framework": "UD-compatible dependencies plus X-bar phrase projections.",
            "core_labels": {
                "root": "Main predicate of the clause.",
                "nsubj": "Nominal/pronominal subject before the predicate.",
                "obj": "Nominal object or complement after the predicate.",
                "advmod": "Degree/modifier material modifying predicate or nominal head.",
                "neg": "Negator attached to predicate.",
                "aux": "Modal or auxiliary attached to predicate.",
                "case": "Possessive particle attached to possessor/nominal phrase.",
                "clf": "Classifier attached to following nominal.",
                "mark": "Clause-final question/aspect particle.",
            },
            "xbar_schema": "XP -> (Specifier) X' ; X' -> X (Complement) (Adjunct*)",
        }

    def _build_clause_projection(
        self,
        tokens: Sequence["ParseToken"],
        predicate_index: int,
        rules: List[str],
        dependencies: List[DependencyArc],
    ) -> XBarNode:
        """Build a TP projection around the main verbal predicate."""

        predicate = tokens[predicate_index]
        dependencies.append(DependencyArc(-1, predicate_index, "root", "main_predicate_as_root"))
        rules.append("main_predicate_as_root")

        specifier = self._nearest_left_index(tokens, predicate_index, self.NOUN_POS)
        complement = self._nearest_right_index(tokens, predicate_index, self.NOUN_POS)
        left_functional = self._left_functional_dependents(tokens, predicate_index)
        right_particles = self._right_particles(tokens, predicate_index)

        children: List[XBarNode] = []
        if specifier is not None:
            dependencies.append(DependencyArc(predicate_index, specifier, "nsubj", "preverbal_np_as_subject"))
            rules.append("preverbal_np_as_subject")
            children.append(self._nominal_projection(tokens, specifier, rules, dependencies))

        vbar_children = [self._terminal_node("V", predicate)]
        for function_index, relation in left_functional:
            dependencies.append(DependencyArc(predicate_index, function_index, relation, "functional_left_dependent"))
            rules.append("functional_left_dependent")
            vbar_children.insert(0, self._terminal_node(tokens[function_index].pos, tokens[function_index], "functional"))

        if complement is not None:
            dependencies.append(DependencyArc(predicate_index, complement, "obj", "postverbal_np_as_object"))
            rules.append("postverbal_np_as_object")
            vbar_children.append(self._nominal_projection(tokens, complement, rules, dependencies))

        for particle_index in right_particles:
            dependencies.append(DependencyArc(predicate_index, particle_index, "mark", "right_clause_particle"))
            rules.append("right_clause_particle")
            vbar_children.append(self._terminal_node(tokens[particle_index].pos, tokens[particle_index], "particle"))

        children.append(XBarNode("V'", head=predicate.text, role="intermediate", children=vbar_children))
        return XBarNode("TP", head=predicate.text, role="clause", children=children)

    def _nominal_projection(
        self,
        tokens: Sequence["ParseToken"],
        head_index: int,
        rules: List[str],
        dependencies: List[DependencyArc],
    ) -> XBarNode:
        """Build an NP projection with possessors, classifiers, and modifiers."""

        head = tokens[head_index]
        children = []
        possessor = self._nearest_left_index(tokens, head_index, {"ENTITY_pronoun", "ENTITY_person"}, max_distance=2)
        classifier = self._nearest_left_index(tokens, head_index, {"ENTITY_classifier"}, max_distance=2)
        modifier = self._nearest_left_index(tokens, head_index, self.MODIFIER_POS, max_distance=2)
        possessive = self._nearest_left_index(tokens, head_index, {"ENTITY_possessive"}, max_distance=1)

        if possessor is not None and possessive is not None:
            dependencies.append(DependencyArc(head_index, possessor, "nmod:poss", "possessor_before_possessive_particle"))
            dependencies.append(DependencyArc(possessor, possessive, "case", "possessive_particle_as_case"))
            rules.append("possessor_before_possessive_particle")
            children.append(self._terminal_node(tokens[possessor].pos, tokens[possessor], "specifier"))
            children.append(self._terminal_node(tokens[possessive].pos, tokens[possessive], "case"))

        if classifier is not None:
            dependencies.append(DependencyArc(head_index, classifier, "clf", "classifier_before_nominal_head"))
            rules.append("classifier_before_nominal_head")
            children.append(self._terminal_node(tokens[classifier].pos, tokens[classifier], "classifier"))

        if modifier is not None:
            dependencies.append(DependencyArc(head_index, modifier, "amod", "modifier_before_nominal_head"))
            rules.append("modifier_before_nominal_head")
            children.append(self._terminal_node(tokens[modifier].pos, tokens[modifier], "adjunct"))

        children.append(self._terminal_node("N", head))
        return XBarNode("NP", head=head.text, role="argument", children=children)

    def _find_main_predicate(self, tokens: Sequence["ParseToken"]) -> Optional[int]:
        """Return the first verbal predicate index, if one exists."""

        for index, token in enumerate(tokens):
            if token.pos in self.VERB_POS:
                return index
        return None

    def _find_nominal_head(self, tokens: Sequence["ParseToken"]) -> int:
        """Return the rightmost nominal head or the final token as fallback."""

        for index in range(len(tokens) - 1, -1, -1):
            if tokens[index].pos in self.NOUN_POS:
                return index
        return len(tokens) - 1

    def _left_functional_dependents(
        self,
        tokens: Sequence["ParseToken"],
        predicate_index: int,
    ) -> List[tuple[int, str]]:
        """Collect negation, auxiliary, and modifier dependents before a predicate."""

        dependents = []
        for index in range(max(0, predicate_index - 3), predicate_index):
            pos = tokens[index].pos
            if pos == "FUNC_negation":
                dependents.append((index, "neg"))
            elif pos in {"MODAL", "AUX"}:
                dependents.append((index, "aux"))
            elif pos in self.MODIFIER_POS:
                dependents.append((index, "advmod"))
        return dependents

    def _right_particles(self, tokens: Sequence["ParseToken"], predicate_index: int) -> List[int]:
        """Collect clause-final particles and aspects after a predicate."""

        particles = []
        for index in range(predicate_index + 1, min(len(tokens), predicate_index + 4)):
            if tokens[index].pos in {"CLAUSE_Q", "CLAUSE_particle", "ASPECT"}:
                particles.append(index)
        return particles

    @staticmethod
    def _terminal_node(label: str, token: "ParseToken", role: str = "head") -> XBarNode:
        """Create a terminal X-bar node for a single token."""

        return XBarNode(label, head=token.text, role=role, token=token.to_obj())

    @staticmethod
    def _nearest_left_index(
        tokens: Sequence["ParseToken"],
        index: int,
        pos_set: set,
        max_distance: int = 6,
    ) -> Optional[int]:
        """Find the nearest matching token index to the left."""

        for distance, cursor in enumerate(range(index - 1, -1, -1), start=1):
            if distance > max_distance:
                break
            if tokens[cursor].pos in pos_set:
                return cursor
        return None

    @staticmethod
    def _nearest_right_index(
        tokens: Sequence["ParseToken"],
        index: int,
        pos_set: set,
        max_distance: int = 6,
    ) -> Optional[int]:
        """Find the nearest matching token index to the right."""

        for distance, cursor in enumerate(range(index + 1, len(tokens)), start=1):
            if distance > max_distance:
                break
            if tokens[cursor].pos in pos_set:
                return cursor
        return None
