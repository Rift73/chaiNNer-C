from __future__ import annotations

from nodes.impl.native_graph import graph


class ReplacementInterpolation:
    def __init__(self, name: str):
        self.name = name


class ReplacementString:
    """
    A parser and interpolator for chainner's string replacement patterns.

    The syntax is as follows (ANTLR 4):

        Pattern: ( LiteralChar | EscapedChar | Interpolation )* ;
        LiteralChar: ~[{] ; // any character except "{"
        EscapedChar: '{{' ;
        Interpolation: '{' InterpolationContent '}' ;
        InterpolationContent: [A-Za-z0-9]+ ;
    """

    def __init__(self, pattern: str):
        self.tokens: list[str | ReplacementInterpolation]
        self.names: set[str]
        graph().utility_replacement_init(self, pattern, ReplacementInterpolation)

    def replace(self, replacements: dict[str, str]) -> str:
        return graph().utility_replacement_replace(self, replacements)
