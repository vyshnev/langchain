from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph import add_messages

from chains import generate_chain, reflect_chain

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflect_chain)
builder.set_entry_point(GENERATE)



def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("Hello")
    inputs = HumanMessage(content="""Make this tweet better:" 
                                    @LangChainAI
    
            - newly Tool Calling feature is seriously underrated.
            After a long wait, it's here- making the implementation of agents across different models with function calling - super easy.
            Make a video covering their newest blog post
                          
                          """)

response = graph.invoke(inputs)