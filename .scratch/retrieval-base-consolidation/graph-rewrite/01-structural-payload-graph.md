# 01 — Structural payload into the graph

**What to build:** The index builder computes the five structural relationships for each message chunk — same conversation, sender–receiver, same sender, same company, same label — and supplies them to the graph at index time as manually-provided relationships, so the graph encodes real conversation structure rather than vector similarity.

**Blocked by:** None — can start immediately (parent: ticket 03 — graph rewrite on real conversation relationships)

**Status:** ready-for-agent

- [ ] Chunk metadata carries `company` (the real company of the sender, not an empty string)
- [ ] For each chunk, the five structural edge descriptors are computed deterministically from metadata: `same_conversation`, `sender_receiver`, `same_sender`, `same_company`, `same_label`
- [ ] The descriptors are passed to txtai via the `relationships` field, and the graph config enables manual relationships (`columns.relationships`)
- [ ] Index builds with the graph enabled and the structural payload lands in the graph
- [ ] No base retrieval behaviour is changed by this ticket — the graph is an addition, not an alteration