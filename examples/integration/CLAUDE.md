# SaveContext usage policy

This project has the **SaveContext** MCP server available (tools appear as
`mcp__savecontext__ingest`, `mcp__savecontext__brief`, etc.). It keeps
large content out of the context window.

**Rules:**

1. Any document, log, transcript, code dump, or tool/API output larger than
   ~2,000 tokens MUST be stored, not pasted inline:
   - Incoming source → `ingest` (returns a `ctx://…@vN` handle).
   - Large generated output → `zip_output`.
2. Work from the **handle + brief**. With several questions about one
   context, call `lookup(queries=[…])` — one call answers them all.
   For single items, call:
   - `brief(context_id, task=…)` to find relevant blocks,
   - `expand(context_id, selector=…, fidelity=…)` for content,
   - `quote(context_id, …)` for exact wording.
3. Never reproduce a stored document verbatim in the conversation unless the
   user asks; cite via `quote` / `expand(fidelity="quotes")` instead.
4. Before relying on exact numbers, dates, money, or obligations, verify them
   with `quote` — the brief is lossy by design.
5. When a document changes, use `diff` rather than re-ingesting
   blindly; report the meaning-impact it returns.

If unsure whether something is "large", ingest it — handles are cheap and the
brief tells you what you have.
