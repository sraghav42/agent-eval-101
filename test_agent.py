import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall, SingleTurnParams
from deepeval.metrics import ToolCorrectnessMetric, GEval
from deepeval.models.base_model import DeepEvalBaseLLM
import boto3
import json
from agent import run_agent
from agent_report import AGENT_RUNS

# Configure Bedrock as the LLM-as-a-Judge for DeepEval metrics
class BedrockJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1")

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return "amazon.nova-2-lite-v1:0"

    def generate(self, prompt: str) -> str:
        res = self.client.converse(
            modelId="global.amazon.nova-2-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}]
        )
        return res["output"]["message"]["content"][0]["text"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

bedrock_judge = BedrockJudge()

def test_weather_and_conversion_trajectory():
    prompt = "What is the temperature in Tokyo right now in Fahrenheit?"
    
    # Run the agent
    final_output, tool_logs = run_agent(prompt)

    # Record the run so it shows up in the report summary
    AGENT_RUNS.append({"input": prompt, "output": final_output, "tool_calls": tool_logs})

    # Reconstruct actual tool calls for DeepEval
    actual_tool_calls = [
        ToolCall(name=call["name"], input_parameters=call["args"]) 
        for call in tool_logs
    ]

    # Define expected tool trajectory
    expected_tool_calls = [
        ToolCall(name="get_weather", input_parameters={"location": "Tokyo"}),
        ToolCall(name="convert_celsius_to_fahrenheit", input_parameters={"celsius": 20})
    ]

    # Create the DeepEval Test Case
    test_case = LLMTestCase(
        input=prompt,
        actual_output=final_output,
        tools_called=actual_tool_calls,
        expected_tools=expected_tool_calls
    )

    # Metric 1: Deterministic check on tool selection and arguments
    tool_metric = ToolCorrectnessMetric(
        threshold=0.8,
        model=bedrock_judge
    )

    # Metric 2: G-Eval Rubric to check output quality
    rubric_metric = GEval(
        name="Answer Completeness",
        criteria="Check if the final output clearly mentions both the original weather condition and the converted Fahrenheit temperature.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=bedrock_judge
    )

    # Execute tests
    assert_test(test_case, [tool_metric, rubric_metric])

def test_multi_city_comparision_trajectory():
    prompt= "Which city is the warmer right now? Tokyo or Paris?"

    # Run the agent
    final_output, tool_logs = run_agent(prompt)

    # Record the run so it shows up in the report summary
    AGENT_RUNS.append({"input": prompt, "output": final_output, "tool_calls": tool_logs})

    # Reconstruct actual tool calls for DeepEval
    actual_tool_calls = [
        ToolCall(name=call["name"], input_parameters=call["args"]) 
        for call in tool_logs
    ]

    # Define expected tool trajectory
    expected_tool_calls = [
        ToolCall(name="get_weather", input_parameters={"location": "Tokyo"}),
        ToolCall(name="get_weather", input_parameters={"location": "Paris"}),
    ]

    # Create the DeepEval Test Case
    test_case = LLMTestCase(
        input=prompt,
        actual_output=final_output,
        tools_called=actual_tool_calls,
        expected_tools=expected_tool_calls
    )

    # Metric 1: Deterministic check on tool selection and arguments
    tool_metric = ToolCorrectnessMetric(
        threshold=0.8,
        model=bedrock_judge,
        should_consider_ordering=False
    )

    # Metric 2: G-Eval Rubric to check output quality
    rubric_metric = GEval(
        name="Comparative Accuracy",
        criteria="Evaluate whether the final answer correctly identifies which city is warmer "
            "based on the retrieved temperatures"
            "and explicitly states both temperature values.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=bedrock_judge
    )

    # Execute tests
    assert_test(test_case, [tool_metric, rubric_metric])

def test_non_temperature_query():
    prompt="What is the capital of Japan?"

    # Run the agent
    final_output, tool_logs = run_agent(prompt)

    # Record the run so it shows up in the report summary
    AGENT_RUNS.append({"input": prompt, "output": final_output, "tool_calls": tool_logs})

    # Reconstruct actual tool calls for DeepEval
    actual_tool_calls = [
        ToolCall(name=call["name"], input_parameters=call["args"]) 
        for call in tool_logs
    ]

    # Define expected tool trajectory
    expected_tool_calls = []  # No tools are expected to be called for a non-temperature query

    # Create the DeepEval Test Case
    test_case = LLMTestCase(
        input=prompt,
        actual_output=final_output,
        tools_called=actual_tool_calls,
        expected_tools=expected_tool_calls
    )

    # Metric 1: Deterministic check on tool selection and arguments
    tool_metric = ToolCorrectnessMetric(
        threshold=0.8,
        model=bedrock_judge,
        should_consider_ordering=False
    )

    # Metric 2: G-Eval Rubric to check output quality
    rubric_metric = GEval(
        name="Answer Accuracy",
        criteria="Evaluate whether the final answer correctly provides the capital of Japan.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=bedrock_judge
    )

    # Execute tests
    assert_test(test_case, [tool_metric, rubric_metric])