# Spec: Retrieval Base Consolidation for CorpChat RAG

## Problem Statement

The CorpChat RAG system's retrieval base cannot reliably retrieve relevant messages for Chinese queries, which is the core language of the corpus. When searching for a precise multi-term phrase such as `物流報價 方案`, the system returns messages that merely contain the word `方案` but are unrelated to the intended context (logistics quotation). When searching for an unspaced Chinese phrase such as `投資美國債券跟藍籌股`, the system sometimes returns messages that contain none of the keywords at all.

The root causes are structural, not bolt-on: the keyword (BM25) side cannot tokenize Chinese; the embedding model is English-only on a Traditional-Chinese corpus; metadata is glued into the searchable text so matches fire on labels and titles rather than content; and the hybrid fusion path re-fetches each result with a bogus text search. As a result, the RAG answer generator receives imprecise top-k context, compromising answer accuracy.

## Solution

Make the retrieval base natively Chinese-capable and structurally clean, so that precise top-k context is retrieved for RAG QA. Then verify each enhancement layer — LLM expansion, graph, reranker — on top of the verified base, one layer at a time, keeping the base's regression suite green throughout.

The base is fixed in place (txtai stack): Chinese word segmentation for BM25, a multilingual embedding model (bge-m3), a clean schema where only content plus a curated customer title is matched and metadata is kept out of the match surface, and a correct document-fetch path. The stale machine-specific index and committed secrets are removed, and the dependency manifest is corrected.

## User Stories

1. As a user asking a precise multi-term Chinese query (e.g. `物流報價 方案`), I want the top retrieved messages to be those actually about the queried context, so that the RAG answer is grounded in the right messages.
2. As a user searching an unspaced Chinese phrase (e.g. `投資美國債券跟藍籌股`), I want the messages that actually contain those keywords to be retrieved, so that keyword-exact search works.
3. As a user searching a bare category label (e.g. `product_inquiry`), I want the system to not treat the label as content and rank all documents of that category, so that matches reflect real content relevance.
4. As a user filtering by label, I want label filtering to still scope results correctly, so that category is a filter, not a match signal.
5. As a user of the search UI, I want to see clean message content (not raw enriched text with title/metadata glued in), so that results are readable.
6. As a developer, I want the keyword and semantic sides both to work on Chinese, so that hybrid retrieval is meaningful rather than vector-only.
7. As a developer, I want a regression test suite that goes red on the current retrieval bugs and green after the fix, so that the base is verified and later layers cannot silently regress it.
8. As a developer, I want the LLM expansion layer verified against the fixed base, so that expansion improves rather than degrades retrieval.
9. As a developer, I want the graph layer rebuilt on real conversation relationships (sender, receiver, chatroom, company), not vector similarity, so that graph expansion surfaces genuine connections.
10. As a developer, I want a reranker that can read Chinese (or is disabled until one is available), so that it does not demote correct Chinese results.
11. As an operator, I want the repo to have no committed secrets and a correct dependency manifest, so that a fresh clone installs and runs the search correctly.
12. As an operator, I want no stale machine-specific prebuilt index in the repo, so that a fresh clone cannot silently load a wrong-format index.

## Implementation Decisions

1. **Fix the base in place on the txtai stack.** No stack replacement. The four base defects (A–D) are addressed together as one unit.
2. **A — Chinese word segmentation for BM25.** Use jieba `cut_for_search` at both index and query time so unspaced Chinese phrases are segmented into matchable tokens. The BM25 scorer must apply the same segmenter on both sides.
3. **B — Multilingual embedding model.** Replace `all-MiniLM-L6-v2` with `BAAI/bge-m3` as the production default (1024-dim, Chinese-capable). The build must support a local model path consistent with the existing deployment pattern.
4. **C — Clean schema.** The match surface is **message content plus a curated title** (`customer_name (company)`). All other fields — label, send time, sender, receiver, message id, origin, chunk index — are **structured metadata only**: used for filtering, display, and LLM context, never matched. The current `\n---\nMetadata:` string-glue and its reverse-parsing are removed. The label is a filter, never a match signal.
5. **D — Correct document fetch.** The hybrid fusion path must fetch each fused document by its actual document id (txtai document lookup), not by re-searching `id:xxx` as text. This removes the per-doc N+1 bogus-search bug.
6. **Reranker disabled by default in this pass.** The current `cross-encoder/ms-marco-MiniLM-L-6-v2` is English-only and can demote correct Chinese results. It is disabled unless a multilingual reranker is available (see Out of Scope / Phase 2).
7. **Single test seam.** All layers are tested through `Searcher.search()` — the public interface `app.py` already calls. The index is a deterministic fixture built from the conversation templates, not a separate seam.
8. **Hygiene.** Remove the stale machine-specific `search_index/` from the repo and git-ignore it. Remove hardcoded API keys and `key.txt`; require keys from environment. Correct `requirements.txt` to include the real search-core dependencies (`txtai`, `chonkie`, `tabulate`, `sentence-transformers`, `jieba`, `pytest`) and git-ignore `.env`, `models/`, `search_index/`.
9. **Layered verification.** Step 1 (base) is built and verified first. Steps 2–4 (LLM expansion, graph rewrite, reranker) each build on the verified base and must keep Step 1's regression suite green.

## Testing Decisions

- A good test asserts **external behavior** through `Searcher.search()` — what a user sees — not implementation details of the index or scorer.
- The regression suite is a **pytest** suite using a **deterministic in-memory index** built from the conversation templates, using the production embedding model (bge-m3) so tests exercise exactly what runs in production.
- The suite must assert:
  1. `物流報價 方案` returns the message actually about logistics quotation/報價, not merely any message containing 方案.
  2. `投資美國債券跟藍籌股` returns the message containing those keywords, not keywordless noise.
  3. Searching a bare label (e.g. `product_inquiry`) does not rank all documents of that label.
  4. Label filtering still scopes results correctly.
- This suite is the **permanent gate**: every later layer (LLM expansion, graph, reranker) must keep it green.
- Prior art: the existing `synthetic-benchmark` command in the search module checks label-based recall; this suite extends that concept to content-relevance assertions and is automated under pytest.

## Out of Scope

- **Graph rewrite (Phase 2 / Step 3):** rebuilding the graph on real conversation relationships (sender, receiver, chatroom, company) rather than vector similarity. This is a separate design task.
- **LLM expansion tuning (Step 2):** re-enabling and tuning `QueryExpander` + weighted RRF is a separate layer after the base.
- **Multilingual reranker (Step 4):** selecting and enabling a Chinese-capable reranker is a separate layer after the base.
- **Secret rotation:** the LiteLLM key already committed can be revoked/rotated by the operator; the repo remediation (removing from code, env-only) is in scope, but rotation of the existing key is an operator action.
- **Full stack replacement:** explicitly rejected; the base is fixed in place on txtai.
- **Domain doc / skill configuration** (`setup-matt-pocock-skills`): a separate future pass.

## Further Notes

- The corpus is ~100% Traditional Chinese; queries are Traditional Chinese; the production index is built on a Linux box with local model files.
- The production index path embeds a machine-specific absolute path; the build must not rely on it.
- The exposed LiteLLM key in git history requires rotation by the operator — this is flagged, not silently done.