from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from IPython.display import display, Image


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


agent_name = "my-agent"

@tool
def add(a:int, b:int):
    """Function adds two numbers. """
    return a + b
@tool
def subtract(a:int, b:int):
    """Function subtracts two numbers. """
    return a - b
@tool
def divide(a:int, b:int):
    """Function divide two numbers. """
    return a / b
@tool
def multiply(a:int, b:int):
    """Function multiply two numbers. """
    return a * b


tools = [add, divide, multiply, subtract]


model = init_chat_model("google_genai:gemini-2.0-flash").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are a helpful assistant please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node(
    agent_name,
    model_call,
)

tool_node = ToolNode(tools = tools)
graph.add_node("my_tools", tool_node)

graph.set_entry_point(agent_name)
graph.add_conditional_edges(
    agent_name,
    should_continue,
    {
        "continue": "my_tools",
        "end": END,
    },
)
graph.add_edge("my_tools", agent_name)

app = graph.compile()

graph_image = app.get_graph().draw_mermaid_png()
with open(f"agent_visual_graphs/{agent_name}.png", "wb") as f:
    f.write(graph_image)
print("Graph saved as graph.png. Open it to view.")

def print_stream(stream):
    for message in stream:
        message = message["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "(40+12)/2+2*(3-1)")]}
print_stream(app.stream(inputs, stream_mode="values"))