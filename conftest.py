from agent_report import AGENT_RUNS


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.write_sep("=", "AGENT RESPONSES")
    if not AGENT_RUNS:
        terminalreporter.write_line("No agent runs recorded.")
        return
    for i, run in enumerate(AGENT_RUNS, 1):
        terminalreporter.write_line(f"[{i}] Input: {run['input']}")
        terminalreporter.write_line(f"    Tool calls: {run['tool_calls']}")
        terminalreporter.write_line(f"    Agent response: {run['output']}")
        terminalreporter.write_line("")
