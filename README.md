<div align="center">

# 🗜️ SaveContext

**Git LFS for your LLM context window.**

Keep big documents *outside* the model — hand it a small searchable handle instead.
The model reads a cheap summary to find what it needs, then pulls back the **exact
original words** to quote, using a fraction of the tokens.

An **MCP server** for Claude Code, Claude Desktop, and any Model Context Protocol
client — with a standalone CLI too.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/AlexanderBoger/SaveContext/blob/master/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![MCP server](https://img.shields.io/badge/MCP-server-6E56CF.svg)](https://modelcontextprotocol.io)
[![tests](https://img.shields.io/badge/tests-80%20passing-brightgreen.svg)](#reference)
[![status](https://img.shields.io/badge/status-beta-orange.svg)](#reference)

</div>

---

```
  ingest 150k-token contract ─▶ ctx://acme@v1
                                 ├─ ~700-token semantic brief
                                 ├─ block map  (b0000 … b0042)
                                 ├─ protected atoms  (money · dates · obligations …)
                                 └─ raw source stored compressed, never auto-sent

  ask 6 questions ─▶ lookup(...)  ─▶  one call · top blocks · exact atoms
                                       · verbatim quotes · confidence flag
```

## Install

With [uv](https://docs.astral.sh/uv/), add SaveContext to Claude Code in one line — no
clone, no `pip`, no virtualenv:

```bash
claude mcp add savecontext -- uvx savecontext
```

Then open Claude Code, run `/mcp` to confirm the tools loaded, and **point the model at
a big document and just ask** — it ingests, looks up, and quotes on its own. To pin
where the database lives, add `-e SAVECONTEXT_DB=/abs/path.db` before the `--`.

Using a different MCP client (Claude Desktop, etc.)? Point its server config at the same
command: `uvx savecontext`. Prefer no MCP at all? The [CLI](#reference) does everything
standalone.

## Why it's different

Most context tools are lossy summarizers — helpful, but they blur the exact numbers and
can't tell you when they're guessing. SaveContext does the two things they can't:

- **🎯 Exact, with receipts.** Facts come back as the *original characters* with their
  source location, never a paraphrase — every span reconstructs byte-for-byte. A summary
  is for *finding*; the quote is for *trusting*.
- **🙅 Says "not found" instead of inventing.** When the answer isn't in the document,
  retrieval is flagged weak and the agent abstains — rather than hallucinating a
  plausible one.
- **📉 Flat cost.** The payload to answer questions stays ~7,300 tokens whether the
  document is 15k or 150k. Pasting scales linearly; this doesn't.
- **🔁 One call, many questions.** `lookup` answers a batch of queries in a single round
  trip, not one retrieve-cycle per question.

## By the numbers

<div align="center">

<table border="0">
<tr>
<td align="center" width="180"><h2>20.5×</h2>less context<br>at 150k tokens</td>
<td align="center" width="180"><h2>~7.3k</h2>flat token cost,<br>any document size</td>
<td align="center" width="180"><h2>54/54</h2>live answers correct,<br>incl. <em>not&nbsp;found</em></td>
<td align="center" width="180"><h2>100%</h2>byte-exact quotes,<br>0 hallucinations</td>
</tr>
</table>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AlexanderBoger/SaveContext/master/assets/benchmark-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/AlexanderBoger/SaveContext/master/assets/benchmark-light.svg">
  <img alt="Bar chart: context tokens to answer six questions. Pasting the whole document scales with size (15,135 to 150,135 tokens) while SaveContext stays flat near 7,300 — 20.5x less at 150k." src="https://raw.githubusercontent.com/AlexanderBoger/SaveContext/master/assets/benchmark-light.svg" width="720">
</picture>

</div>

Measured with real agents on a contract set plus **held-out** documents (an RFC and a
synthetic incident log the retriever was never tuned on), answering fact questions that
have distractor values planted next to the true ones, graded against ground truth.
Crossover is ≈ 5k tokens — below that, just paste; above it, the gap widens with size.

> Honest note: live runs are single samples — treat small differences as noise. The
> durable findings are the flat-vs-scaling cost curve, byte-exact retrieval, and zero
> hallucination on absent facts. Dense technical prose is the current retrieval frontier.
> Reproduce it yourself: [`examples/benchmark_suite/`](https://github.com/AlexanderBoger/SaveContext/blob/master/examples/benchmark_suite).

---

## Reference

<details>
<summary><b>The 11 tools</b></summary>

<br>

All tools return clean JSON. Handles are stable and readable: `ctx://label@vN` for
contexts, `out://label@vN` for outputs.

| Tool | Purpose |
|------|---------|
| `ingest` | Store long text; return handle, brief, block map, protected-atom summary, safety notes. **Does not return the raw source.** Optional `agent_brief` lets the model supply its own summary. |
| `lookup` | **Batched retrieval** — answer several queries in one call. Per query: the top block(s), the exact atoms inside them, a verbatim best-matching sentence, and a `confidence` flag (`strong`/`weak`) so the agent can abstain instead of guessing. |
| `brief` | Task-specific brief: relevant blocks + atoms for a given task, with uncertainty notes. |
| `expand` | Lazily expand a block / atom / section / search match at `fidelity = summary \| facts \| quotes \| full`. |
| `quote` | Exact verbatim quote (by atom id or search query) with surrounding context and source ref. |
| `map` | Full (or paginated) block map + atom summary, fetched lazily. |
| `set_brief` | Replace a context's brief with an agent-authored one. Raw source and atoms unchanged. |
| `diff` | Diff new source against a stored context; report added/removed/changed **atoms** and meaning impact. Creates a new version. |
| `zip_output` | Store a large generated **output**; return section map + preview. |
| `expand_output` | Expand part of a stored output at `fidelity = preview \| section \| full`. |
| `audit` | Compression ratio, atom counts, per-category preservation estimate, `safe_for` / `unsafe_for`, warnings. |

**`expand` fidelity levels:** `summary` (extractive lead sentences, cheapest) · `facts`
(summary **plus** exact atoms, `[money] $500,000`) · `quotes` (verbatim spans you can
cite) · `full` (raw block text — the only path that returns source prose).

</details>

<details>
<summary><b>What a turn looks like</b></summary>

<br>

```text
1. User pastes a 150,000-token contract.
2. Claude → ingest(source_text=…, label="acme")
        ← ctx://acme@v1, ~700-token brief, block map, atoms.
3. User: "Liability cap? Notice period? Governing law?"
4. Claude → lookup(ctx://acme@v1, ["liability cap", "termination notice", "governing law"])
        ← per query: the governing block, its atoms, a verbatim sentence, confidence.
5. Claude → quote(ctx://acme@v1, search_query="shall not exceed")   # verify exact wording
        ← "…total liability shall not exceed $750,000…" + source offsets.
6. Claude answers using only the relevant, verbatim, cited context.
```

A runnable version is in [`examples/demo.py`](https://github.com/AlexanderBoger/SaveContext/blob/master/examples/demo.py) (`python examples/demo.py`).

</details>

<details>
<summary><b>Command line (<code>savectx</code>)</b></summary>

<br>

The package ships a CLI that shares the same store as the MCP server — ingest on one,
query from the other:

```bash
savectx ingest report.md --label q4 --type auto      # ingest a file
savectx ingest - --label pasted                       # or read stdin
savectx list                                          # what's in the store
savectx lookup ctx://q4@v1 --query "liability cap" --query "fees"   # batched
savectx brief  ctx://q4@v1 --task "liability risks"
savectx expand ctx://q4@v1 --selector liability --fidelity facts
savectx quote  ctx://q4@v1 --search "shall not exceed"
savectx audit  ctx://q4@v1
savectx diff   ctx://q4@v1 report_v2.md               # version + change report
savectx serve                                         # run the MCP server
```

Bare `savecontext` (and `python -m savecontext`) runs the MCP server, so MCP configs
keep working unchanged.

</details>

<details>
<summary><b>Configuration</b></summary>

<br>

Set via environment (or the MCP `env` block):

| Variable | Default | Purpose |
|---|---|---|
| `SAVECONTEXT_DB` | `~/.savecontext/savecontext.db` | Store location. `:memory:` for ephemeral. Shared by CLI and server. |
| `SAVECONTEXT_TRANSPORT` | `stdio` | `stdio` \| `http` \| `sse` |
| `SAVECONTEXT_HOST` / `SAVECONTEXT_PORT` | FastMCP defaults | For `http`/`sse` transport |
| `SAVECONTEXT_EMBEDDINGS` | off | `1` enables a local sentence-transformers retrieval backend (default is dependency-free BM25) |
| `SAVECONTEXT_LLM_SUMMARY` | off | `1` has a local Ollama model write an abstractive brief at ingest (fails closed to extractive) |

The `ctx://` handle scheme is stable. `SAVECONTEXT_*` vars take precedence; the
pre-rename `CONTEXTSAVER_*` / `CONTEXTVAULT_*` names are still honored as fallbacks.

</details>

<details>
<summary><b>How it works</b> (architecture + the loss-aware guarantee)</summary>

<br>

| Layer | Module | Behaviour |
|-------|--------|-----------|
| Tokenizer | `tokenizer.py` | tiktoken `cl100k_base`, else `chars/4` |
| Chunking | `chunking.py` | deterministic: headings + paragraph packing to a token budget, exact offsets |
| Extraction | `extraction.py` | regex atoms: dates, money, %, numbers, URLs, emails, entities, negations, obligations, code ids, file paths |
| Retrieval | `retrieval.py` | **BM25** + concept-group expansion + answer-type intents + saturation scoring; optional embedding backend behind the same `rank` interface |
| Summary | `summarize.py` | extractive: ranked blocks + lead sentences + high-value atom lines; optional local-LLM |
| Diff | `diffing.py` | atom-set diff + coarse block text diff + risk impact |
| Storage | `storage.py` | SQLite metadata + **zstd-compressed** raw source; blocks/atoms as offsets |
| Service / Server | `service.py`, `server.py` | the eleven tools; FastMCP over stdio / HTTP / SSE |

**Loss-aware guarantee.** Blocks and atoms are stored as *offsets into the raw source*,
so any span reconstructs byte-for-byte. The brief is explicitly flagged lossy; exact
wording always comes from `quote()` or `expand(fidelity="quotes")`. The full raw source
is only ever returned via `expand(fidelity="full")`.

</details>

<details>
<summary><b>Who writes the brief?</b></summary>

<br>

Three summarizer options, best first — all keep the protected atoms exact; only the
prose brief changes:

1. **The calling agent (recommended).** Claude already has the document in context at
   `ingest` time, so it writes a faithful, dense brief for free — inline via
   `ingest(agent_brief=…)` or afterwards via `set_brief`. (`brief_mode='agent'`)
2. **Local LLM (optional).** `SAVECONTEXT_LLM_SUMMARY=1` → a local Ollama model writes an
   abstractive brief; fails closed to extractive. (`brief_mode='abstractive(local-llm)'`)
3. **Extractive (default).** Rule-based real sentences from the source — free, offline,
   deterministic, always available. (`brief_mode='extractive'`)

</details>

<details>
<summary><b>Development & tests</b></summary>

<br>

```bash
git clone https://github.com/AlexanderBoger/SaveContext && cd SaveContext
python -m pip install -e ".[tokenizer,dev]"
python -m pytest            # 80 tests
```

Covers tokenizer / handles / chunking / extraction / BM25 retrieval units, an end-to-end
pass over all eleven tools (offset exactness, verbatim quotes, versioning, money-change
diff, audit preservation buckets, error handling), the batched-`lookup` retrieval +
confidence logic, and the CLI flow.

</details>

<details>
<summary><b>Roadmap</b></summary>

<br>

- ✅ BM25 + hybrid semantic ranking (concept groups, answer-type intents) — no model required
- ✅ Batched `lookup` with per-query confidence / abstention
- ✅ Verbatim-integrity + extraction-recall reporting in `audit`
- ✅ PDF / DOCX / HTML loaders and recursive folder ingest
- ✅ HTTP / SSE transport in addition to stdio
- ⬜ Embedding backend benchmarked against the held-out suite (dense-prose retrieval)
- ⬜ Vector-store persistence so embeddings aren't recomputed per query
- ⬜ LLM-assisted atom extraction for fuzzy / multi-span facts

</details>

## License

[Apache License 2.0](https://github.com/AlexanderBoger/SaveContext/blob/master/LICENSE) — see also [NOTICE](https://github.com/AlexanderBoger/SaveContext/blob/master/NOTICE). If you redistribute
SaveContext or build on it, retain the copyright and the `NOTICE` file, per the license
(that's how attribution is required).
