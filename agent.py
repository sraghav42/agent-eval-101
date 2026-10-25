import json
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Mock tool implementations
def get_weather(location: str) -> str:
    weather_db = {
        "tokyo": {"temp_celsius": 18, "condition": "Cloudy"},
        "paris": {"temp_celsius": 12, "condition": "Rainy"},
    }
    loc = location.lower()
    if loc in weather_db:
        return json.dumps(weather_db[loc])
    return json.dumps({"temp_celsius": 20, "condition": "Sunny"})

def convert_celsius_to_fahrenheit(celsius: float) -> str:
    fahrenheit = (celsius * 9/5) + 32
    return json.dumps({"temp_fahrenheit": fahrenheit})

# Tool definitions for AWS Bedrock Converse API
tools = [
    {
        "toolSpec": {
            "name": "get_weather",
            "description": "Get current weather details for a given city.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "convert_celsius_to_fahrenheit",
            "description": "Convert temperature from Celsius to Fahrenheit.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"celsius": {"type": "number", "description": "Temperature in Celsius"}},
                    "required": ["celsius"]
                }
            }
        }
    }
]

def run_agent(user_prompt: str):
    """Executes the agent loop and returns the final answer along with logged tool calls."""
    messages = [{"role": "user", "content": [{"text": user_prompt}]}]
    tool_calls_logged = []

    # Model call loop
    while True:
        response = bedrock.converse(
            modelId="global.amazon.nova-2-lite-v1:0",  # or amazon.nova-micro-v1:0
            messages=messages,
            toolConfig={"tools": tools}
        )
        
        output_message = response["output"]["message"]
        messages.append(output_message)

        # Check if the model wants to call a tool
        content_blocks = output_message["content"]
        tool_requests = [b for b in content_blocks if "toolUse" in b]

        if not tool_requests:
            # Model finished, extract text
            text_response = "".join([b["text"] for b in content_blocks if "text" in b])
            return text_response, tool_calls_logged

        # Process tool calls
        tool_results = []
        for req in tool_requests:
            tool_use = req["toolUse"]
            name = tool_use["name"]
            args = tool_use["input"]
            tool_id = tool_use["toolUseId"]

            tool_calls_logged.append({"name": name, "args": args})

            if name == "get_weather":
                result = get_weather(args["location"])
            elif name == "convert_celsius_to_fahrenheit":
                result = convert_celsius_to_fahrenheit(args["celsius"])
            else:
                result = "Unknown tool"

            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_id,
                    "content": [{"json": json.loads(result)}]
                }
            })

        messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    answer, tool_calls = run_agent("What is the weather in Tokyo in Fahrenheit?")
    print("Answer:", answer)
    print("Tool calls:", tool_calls)