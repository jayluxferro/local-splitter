"""Optional third-party tactic hooks (extension point).

Load custom logic by importing this package from your own code and
calling registered hooks — the core pipeline does not auto-import
arbitrary modules (SSRF-safe default).

Example::

    from local_splitter.plugins import TacticHook

    class MyHook:
        name = "example"

        async def after_compress(self, messages, context):
            return messages

See docs/EXTENSIONS.md for the intended contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from local_splitter.models import Message


@runtime_checkable
class TacticHook(Protocol):
    """Optional hook implementations live outside the core package."""

    name: str

    async def after_compress(
        self,
        messages: Sequence[Message],
        context: Mapping[str, Any],
    ) -> Sequence[Message]:
        """Return possibly modified messages (default: passthrough)."""
        ...


__all__ = ["TacticHook"]
