from typing import Any

class ASTSource:
    def __init__(
        self,
        fn: Any,
        signature: dict[str, str],
        constexprs: dict[str, Any],
        attrs: dict[tuple[int, ...], list[list[Any]]] | None = None,
    ) -> None: ...
