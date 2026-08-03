# 02 — LLM expansion on the verified base

**What to build:** The LLM query-expansion layer (semantic rephrase + keyword expansion, weighted RRF fusion) works on top of the fixed Chinese-capable base. A user searching with expansion enabled gets results that are at least as relevant as the base alone, and ideally better — expansion must not degrade the base's precision. The base regression suite stays green.

**Blocked by:** 01 — Chinese-capable hybrid base

**Status:** ready-for-agent

- [ ] `QueryExpander` re-enabled on the verified base
- [ ] Weighted RRF fusion (0.5 / 1.3 / 1.0) verified against the base-only results on the regression queries
- [ ] Expansion demonstrably improves (or at worst matches) base-only retrieval on the regression suite
- [ ] Regression suite stays green with expansion on