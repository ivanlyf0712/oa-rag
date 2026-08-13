Status: ready-for-agent
Type: task

# 01 — Search package boundary and CLI split

**What to build:** a clean modular search boundary so the contract search stack has an unambiguous package namespace and a separate CLI entrypoint.

**Blocked by:** None — can start immediately.

## Acceptance criteria
- The search package namespace is unambiguous and stable.
- The CLI entrypoint is separate from the package namespace.
- Existing public search entrypoints continue to work through the new structure.
- The structural move preserves current behavior.
- The app layer can keep importing the public search API without ambiguity.
- The package boundary is ready for the later agentic port without introducing a compatibility shim trap.
