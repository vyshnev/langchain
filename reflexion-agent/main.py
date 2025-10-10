from dotenv import load_dotenv
from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph, StateGraph

from chains import revisor, first_responder
from tool_executor import execute_tools



load_dotenv()


MAX_ITERATIONS = 2
builder = MessageGraph()
builder.add_node(key: "draft", first_responder)
builder.add_node(key: "execute_tools", execute_tools)
builder.add_node("revise", revisor)
builder.add_edge(start_key: "draft", end_key: "execute_tools")
builder.add_edge(start_key: "execute_tools", end_key: "revise")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visit = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visit
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder.add_conditional_edges(start_key: "revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()

print(graph.get_graph().draw_ascii())
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    print("hello reflexion")
    res = graph.invoke("Write about AI-Powered SOC / autonomous")
    print(res[-1].tool_calls[0]["args"]["answer"])