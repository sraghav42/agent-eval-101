# agent-eval-101

A minimal starter repo for learning how to **evaluate LLM agents**. A tiny tool-calling weather agent (AWS Bedrock + Converse API) is tested with [DeepEval](https://github.com/confident-ai/deepeval) — showing how to verify both **what tools the agent called** and **how good its final answer is**.

## What's inside

- **`agent.py`** — A simple agent loop using Amazon Nova via the Bedrock Converse API, with two mock tools:
  - `get_weather(location)` — returns fake weather for a city
  - `convert_celsius_to_fahrenheit(celsius)` — unit conversion
- **`test_agent.py`** — Three evals demonstrating the core patterns of agent evaluation:
  | Test | What it checks |
  |------|----------------|
  | Weather → Fahrenheit | Tool trajectory (weather + conversion) & answer completeness |
  | Tokyo vs. Paris comparison | Multi-tool calling (order-independent) & comparative accuracy |
  | Non-weather question | Correctly calls **no** tools & still answers accurately |
- Two metric types are showcased:
  - **`ToolCorrectnessMetric`** — deterministic check that the right tools were called with the right arguments
  - **`GEval`** — LLM-as-a-judge rubric scoring the final answer (powered by a custom Bedrock judge)
- **`conftest.py` + `agent_report.py`** — Prints each agent run (input, tool calls, response) in the pytest summary for easy inspection.

## Prerequisites

1. **Python ≥ 3.11** and [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. **AWS credentials** configured (via `aws configure`, or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars)
3. **Amazon Bedrock model access** enabled in `us-east-1` for **Amazon Nova 2 Lite** (`global.amazon.nova-2-lite-v1:0`) — the same model powers both the agent and the eval judge. Check this in the AWS Console → Bedrock → Model access.

## Quickstart

```bash
git clone <this-repo-url> && cd agent-eval-101
uv sync                                    # install dependencies

uv run python agent.py                     # run the agent once
uv run deepeval test run test_agent.py     # run the eval suite
```

The pytest output ends with an **AGENT RESPONSES** section showing exactly what the agent did for each test — a handy way to debug failing evals.

## Next steps

Swap in your own agent and tools, add test cases to `test_agent.py`, and tune the G-Eval rubrics to your use case. Happy evaluating!
