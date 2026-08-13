# Epic: Unify contract search + risk search behind a LangChain agentic UI

## Background

oa-rag currently has:
- a normal contract retrieval search path
- a separate risk search path
- a Streamlit app that needs to make the agentic workflow explicit

The new product goal is to make the app behave like a single intelligent assistant:
- the user asks one question
- LangChain classifies the intent
- the system decides whether to use normal search or risk search
- the UI clearly shows what happened

The risk search should be available as a unique tool to the agent, and the default behavior should still be normal search for general questions.

---

## Ticket 1 — Add LangChain agent orchestration over contract search and risk search

### Title
Implement LangChain tool-calling agent for unified query routing

### Description
Create a LangChain-based orchestration layer that sits above the existing retrieval code and decides whether a query should use:
- normal contract search
- risk search
- clarification / fallback

The agent must use tool calling so the LLM can automatically invoke  when the query is risk-related. Normal search should remain the default path for general contract questions.

This ticket should not rewrite retrieval logic. It should wrap the existing search and risk pipelines in agent-friendly tools.

### Requirements
- Create a LangChain agent module or service layer.
- Expose at least two tools:
  - 
  - 
- Use tool calling to let the model select  when appropriate.
- Keep normal contract search as the default tool path.
- Return structured information from the agent:
  - detected intent
  - selected tool
  - fallback/clarification state
  - output text
  - optional raw tool observations
- Preserve existing retrieval behavior underneath the agent.

### Acceptance criteria
- A contract-related query routes to normal search.
- A risk-related query automatically invokes .
- A vague or ambiguous query triggers clarification or a safe fallback.
- The agent can return a final answer with tool provenance.
- Tool failure does not crash the app; it degrades safely.

### Implementation notes
- Prefer an adapter layer around the existing  and  functions.
- Keep the orchestration logic independent from Streamlit UI code.
- Make the agent easy to test without the UI.

### Test expectations
- Unit tests for tool selection.
- Unit tests for risk tool invocation.
- Unit tests for default contract search behavior.
- Unit tests for fallback / clarification behavior.

---

## Ticket 2 — Refactor  into a single unified agentic UI

### Title
Replace dual-tab app with one agentic Streamlit workflow

### Description
Refactor the Streamlit app so the user interacts with a single search/chat surface instead of separate tabs.

The UI should make the agentic workflow explicit:
- user submits a query
- the agent decides what to do
- the UI displays the detected intent
- the UI displays which tool was used
- the UI displays retrieved evidence and final answer

If helpful, the overall layout may follow the CorpChat-RAG style, but the implementation must live in  and work end-to-end in oa-rag.

### Requirements
- Remove the current two-tab layout.
- Use a single primary interface.
- Show agent decision metadata in the UI.
- Show contract search results and risk search results in clearly separated sections.
- Keep the UI understandable to non-technical users.
- Ensure the app still starts from .

### Acceptance criteria
- The app no longer exposes two separate tabs for search modes.
- The agentic workflow is visible to the user.
- Users can see whether the answer came from normal search or risk search.
- The app still works with existing search data and index setup.
- The page layout is clean and consistent.

### Implementation notes
- Use expanders or side panels if you need to surface agent steps without overwhelming the main page.
- The UI should not duplicate routing logic.
- Keep presentation code separate from search/tool logic where possible.

### Test expectations
- UI seam tests that confirm the app renders a unified workflow.
- Tests for agent decision metadata rendering.
- Tests that the app still calls the proper underlying search functions.

---

## Ticket 3 — Make agentic search the default behavior for user queries

### Title
Route all user queries through the agent by default

### Description
Update the app’s default behavior so the agent is what the user experiences first.

The agent should become the default entrypoint for normal search interactions. This does not mean normal search is removed. It means the agent internally chooses normal search unless the query clearly requires risk analysis or clarification.

### Requirements
- Make the agent the default path for app searches.
- Keep normal contract search as the default tool choice.
- Preserve the ability to run risk search automatically when the agent detects risk intent.
- Ensure no separate mode selection is required for basic use.
- If there is a CLI or API entrypoint, decide whether it should also default to the agent or remain explicit.

### Acceptance criteria
- A user can ask a general question and get normal contract search behavior through the agent.
- A risk question automatically routes to risk search.
- There is no manual mode switching required for normal use.
- The app’s default interaction feels like one assistant, not two disconnected features.

### Implementation notes
- This ticket is mostly product-flow and wiring.
- It may require updating the app startup path, default UI state, or command semantics.
- Keep the retrieval internals unchanged unless needed for routing.

### Test expectations
- Tests that confirm default search requests flow through the agent.
- Tests that confirm normal search remains the fallback/default tool.
- Tests that confirm risk intent still reaches the risk tool automatically.

---

## Ticket 4 — Add regression tests for agentic routing and unified UI behavior

### Title
Add regression gate for LangChain agent routing and app integration

### Description
Create a regression test suite that protects the new agentic behavior from future breakage.

These tests should cover both routing behavior and UI seams so we can safely refactor later.

### Requirements
- Add tests for routing:
  - contract question → contract search
  - risk question → risk_search
  - unclear question → clarify/fallback
- Add tests that validate agent output shape or metadata.
- Add tests that validate the app integrates with the agent layer.
- Add tests that ensure no accidental regression to separate-tab behavior.
- Keep tests deterministic and fast.

### Acceptance criteria
- Routing regressions are caught.
- Tool selection regressions are caught.
- App wiring regressions are caught.
- Tests run in the default suite and pass reliably.

### Implementation notes
- Mock LangChain model/tool calls where necessary.
- Avoid tests that depend on live LLM behavior.
- Focus on external behavior and stable seams.

---

## Ticket 5 — Add LangChain dependency and app wiring support if missing

### Title
Introduce LangChain dependency and supporting configuration

### Description
If LangChain is not already available in the environment, add the minimum required dependency set and configuration needed for the agent implementation.

This includes any model provider integration, tool-calling support, and environment variables needed for local development.

### Requirements
- Add LangChain packages to project dependencies.
- Configure the model provider used for tool calling.
- Ensure the app can run locally without manual dependency hacks.
- Document required environment variables in README or app startup notes.

### Acceptance criteria
- The project installs and runs with the new agent dependencies.
- The app can call the model provider successfully.
- Missing configuration fails clearly with a useful error message.

### Implementation notes
- Keep dependency additions minimal.
- Prefer provider-agnostic LangChain abstractions if practical.

---

## Suggested order of implementation
1. LangChain agent orchestration
2. Unified Streamlit UI
3. Default-to-agent behavior
4. Regression tests
5. Dependency/config support

---

## Suggested acceptance language for the epic

> The oa-rag app presents a single agentic search experience.  
> The LangChain agent automatically routes general queries to normal contract search and risk-related queries to  via tool calling.  
> The UI makes the routing and workflow visible, and regression tests protect the behavior.
