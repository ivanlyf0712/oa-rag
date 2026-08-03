# CorpChat RAG Context

The CorpChat RAG system retrieves WeChat Work customer-service messages for QA answering. This context covers the retrieval stack (search base, expansion, graph) built and verified incrementally in `.scratch/retrieval-base-consolidation/`.

## Language

**Base retrieval**:
A search over message content plus the curated customer title, using the hybrid (keyword + semantic) index, without any enhancement layer. The permanent regression gate measures the base.
_Avoid_: bare search, vanilla search

**Match surface**:
The part of a document that is matched during retrieval: message content plus a curated title (`customer_name (label)`). All other fields are structured metadata only — used for filtering, display, and LLM context, never for matching.
_Avoid_: enriched text, searchable text

**Graph expansion**:
The enhancement that appends structurally-connected messages to the base results when `graph_expand` is enabled. Graph expansion is append-only: it adds documents below the base ranking and never reorders or displaces base results.
_Avoid_: graph boost, neighbor re-ranking

**Conversation**:
A chatroom (one `open_kfid`) holding a sequence of customer-service turns between an initiator and a responder.
_Avoid_: thread, room, chat session

**Message chunk**:
The unit of indexing and the node of the conversation graph. A message is split into sentence-level chunks; each chunk is one document and one graph node.
_Avoid_: segment, snippet

**Conversation graph**:
The graph built on real conversation structure — sender, receiver, chatroom (`open_kfid`), company, and label — with edges carrying the relationship type. Edges are deterministic and structural, never inferred from vector similarity.
_Avoid_: similarity graph, auto-inferred graph

**Structural edge**:
A graph edge whose existence is guaranteed by the conversation's structured metadata, not by content similarity. Encodes one of: same conversation, sender–receiver, same sender, same company, or same label.
_Avoid_: inferred edge, similarity edge

**Traversal-eligible edge**:
A structural edge type that graph expansion is allowed to walk. Currently: same conversation, sender–receiver, same sender, same company. The same-label edge is recorded but never traversed, so a label never acts as a match signal.
_Avoid_: walkable edge, expandable edge