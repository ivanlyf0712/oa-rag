# ADR-0001: Structural conversation graph instead of vector-similarity graph

We replaced txtai's auto-inferred, vector-similarity graph with a structural conversation graph built from real conversation relationships: sender, receiver, chatroom (`open_kfid`), company, and label. Graph expansion traverses these structural edges to surface genuine connections rather than a similarity echo.

The graph is deterministic: every chunk node gets at least one structural edge at index time via txtai's `relationships` field, and since txtai graph inference defaults to `approximate: true`, nodes that already have edges are skipped by vector inference — the graph is purely structural. The previous `graph_mode=llm` path (`_extract_relations_with_llm`) was removed: it called a non-existent API (`add_edge`), was non-deterministic, and was the text-guessing anti-pattern this ticket removes.

Five structural edge types are recorded — `same_conversation`, `sender_receiver`, `same_sender`, `same_company`, `same_label` — but only the first four are traversal-eligible. `same_label` is recorded but never traversed, so a label never acts as a match signal.

Trade-offs: we chose one-node-per-chunk (homogeneous) over entity nodes for people/companies/labels because the test seam is `Searcher.search()`, which returns documents — chunk-to-chunk expansion yields documents directly, while entity nodes need 2-hop traversal and node-type filtering. Expansion is append-only below the base ranking (score = parent x 0.8, never re-sorted), so it can only add relevance, never reorder or displace base results.