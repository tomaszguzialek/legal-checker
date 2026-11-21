# ADR 0001: LangGraph `StateGraph` over Deep Agents for the PoC agentic layer

**Status:** Accepted  
**Date:** 2026-11-21  
**Applies to:** `specs/SPEC.md` (version 1.1), sections 1.2, 3.4–3.5, Stage 4

---

## Context

The agentic layer of the AI Legal & Compliance Checker PoC is built on the LangChain ecosystem. Two levels of abstraction were considered:

1. **LangGraph `StateGraph`**: an explicitly defined state graph with our own nodes, conditional edges, `ToolNode` and limits.
2. **Deep Agents** (`deepagents`, `create_deep_agent`): a ready-made agent harness built on LangGraph. By default it adds planning middleware (a to-do list), a virtual filesystem for managing context, delegation to sub-agents and automatic summarization of the history. It returns a compiled LangGraph graph.

The shape of the problem in the PoC:
* One call checks one document fragment against one legal act named by the user.
* The flow is mostly deterministic and known in advance: ELI resolver → fetch act → split by legal structure → BM25 → LLM assessment → validation.
* The only agentic decision is whether to fetch additional provisions (e.g. an article cited inside a candidate). Two read-only tools handle it.
* The context is small: a few dozen legal units in RAM, top-K = 12.
* Key guardrail: the LLM judges **only** the provisions it actually received, and code verifies this deterministically (`consulted_units`, quote validation).
* Tests must deterministically check routing, tool-call limits and retries on fake models, with no network and no API key.

LangChain's documentation recommends Deep Agents for long-running, multi-step tasks that need planning and decomposition. It recommends `StateGraph` when you need low-level control over orchestration and workflows that mix deterministic and agentic steps.

## Decision

The PoC implements the agentic layer as a **custom LangGraph `StateGraph`**, using the topology in section 3.5 of the spec:
* deterministic steps as plain Python nodes,
* a single `research ⇄ tools` loop capped by `AGENT_MAX_TOOL_CALLS`,
* a separate evaluation call with Structured Outputs,
* a deterministic post-validator.

**The PoC does not use Deep Agents.** They are planned as the top layer of the production version (see "Evolution path").

## Rationale

* **No planning needed:** the sequence of steps is fixed. A to-do list and task decomposition add nothing when assessing one fragment against one act.
* **No context problem:** the virtual filesystem and history summarization solve context that outgrows the model's window. That doesn't happen in the PoC.
* **Stronger guardrails:** the explicit graph guarantees three things:
  * the LLM doesn't control the deterministic steps,
  * it sees only provisions from BM25 and the read-only tools,
  * code computes the set of provisions it saw.

  An agent with default filesystem tools and sub-agents can do more, which makes this property harder to prove.
* **Testability:** every graph path can be tested deterministically on `GenericFakeChatModel`. That covers the BM25 = 0 shortcut, the tool-call limit and the schema retry. An open-ended planning loop is much harder to cover with tests.
* **Cost and latency:** planning and sub-agent calls add tokens and response time without improving a single assessment.
* **Clarity for PoC reviewers:** the graph diagram (`--show-graph`) shows clearly which work is code and which is the LLM.

## Consequences

**Positive:**
* Full control over the flow, limits and error handling, and a predictable cost per call.
* Deterministic graph tests with no network and no API key.
* Fewer dependencies (no `deepagents`).

**Negative / accepted trade-offs:**
* We write our own orchestration code (nodes, router, retry). Deep Agents partly provide this out of the box.
* Extending to multiple acts or a whole document needs a new layer. The current graph can't handle that by itself.

## Evolution path

Deep Agents become the natural choice for the top layer once the scope goes beyond the PoC. That is the "ambitious variant": a whole document, no act hint and multiple acts.
* A top-level agent splits the document into clauses and plans which acts to check for each.
* It delegates each (clause, act) pair to a sub-agent. That sub-agent is the **current compiled graph**, wrapped in `CompiledSubAgent`.
* It collects the results into a report on the virtual filesystem.

All PoC work can be reused this way. Revisit this decision when work on the ambitious variant starts.

## Alternatives considered

* **Deep Agents (`create_deep_agent`) with the `search_provisions` / `get_provision` tools:** rejected for the reasons under "Rationale".
* **LangChain `create_agent` (prebuilt ReAct + `response_format`):** simpler than a custom graph, but it merges research and evaluation into one loop. That makes three things harder:
  * an explicit BM25 = 0 shortcut,
  * a separate schema retry,
  * deterministic nodes before and after the agent.

## Sources

* [Deep Agents — LangChain docs](https://docs.langchain.com/oss/python/deepagents)
* [deepagents — API reference](https://reference.langchain.com/python/deepagents/)
* [LangChain — products and when to use them](https://docs.langchain.com/oss/python/concepts/products)
* [Deep Agents vs LangChain vs LangGraph (LangChain blog)](https://langchain.com/blog/deep-agents-vs-langchain-vs-langgraph)
* [Doubling down on deepagents (LangChain blog)](https://www.langchain.com/blog/doubling-down-on-deepagents)
