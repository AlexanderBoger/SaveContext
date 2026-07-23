# Integrating SaveContext with Claude Code

Three layers, weakest → strongest enforcement. Use as many as you want.

## 1. Register the server (`.mcp.json`)

Copy `.mcp.json` to your project root (set the absolute `SAVECONTEXT_DB`
path), or run:

```bash
# zero-install via uvx (recommended — no clone, auto-updates):
claude mcp add savecontext -- uvx savecontext
# or, if installed from source:
claude mcp add savecontext -- python -m savecontext
```

Verify with `/mcp` — you should see the 11 savecontext tools
(`ingest`, `set_brief`, `brief`, `lookup`, `map`, `expand`, `quote`, `diff`,
`zip_output`, `expand_output`, `audit`).

## 2. Tell Claude to use it (`CLAUDE.md`)

Append the contents of `CLAUDE.md` to your project's `CLAUDE.md`. This makes
"vault large content" the default behaviour. Advisory — Claude follows it, but
nothing forces it.

## 3. Enforce it on large input (hook)

The harness runs hooks deterministically, so this actually fires every time:

1. Copy `savecontext_hint.py` somewhere in your project.
2. Merge `settings.hook.json` into `.claude/settings.json` (fix the path).
3. Optionally set `SAVECONTEXT_HINT_TOKENS` (default 2000).

Now whenever you paste something large, Claude is reminded to `ingest` it
first. Test the hook directly:

```bash
echo '{"prompt":"'"$(python -c 'print("word "*3000)')"'"}' \
  | python examples/integration/savecontext_hint.py
```

## Strongest: pre-ingest in your app

If you control the app feeding Claude, call `ingest` (or the CLI)
*before* the model sees the document and pass only the handle. Then there's
nothing to enforce — the raw text physically never enters the context window.
