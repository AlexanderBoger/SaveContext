"""SaveContext — a versioned, loss-aware context compression layer for LLMs.

"Git LFS for LLM context": keep large inputs/outputs out of the model context
window while exposing compact briefs, stable handles, semantic maps, exact
quotes, diffs, and lazy expansion through MCP.
"""

from .service import VaultService

__all__ = ["VaultService"]
__version__ = "0.1.0"
