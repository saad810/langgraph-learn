from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from IPython.display import display, Image


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


agent_name = "drafting-agent"
content_path = "drafter_agent_content/"
document_content = ""


@tool
def update_doc_content(text: str):
    """Function updates the document content."""
    global document_content
    document_content = text
    return f"Document content updated. Current content:\n{document_content}"


@tool
def save_doc(filename: str):
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """
    global document_content
    if not filename.endswith(".txt"):
        filename += ".txt"

    if not document_content:
        return "Document is empty. Nothing to save."
    
    try:
        with open(f"{content_path}/{filename}", "w") as f:
            f.write(document_content)
        print(f"Document saved to {content_path}/{filename}")
        return "Document saved successfully."
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
tools = [update_doc_content, save_doc]

model = init_chat_model("google_genai:gemini-2.0-flash").bind_tools(tools)
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}

    """)

    if not state["messages"]:
        user_input = "What would you like to write today?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}
    

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]

    if not messages:
        return "continue"
    
    last_message = messages[-1]
    if (isinstance(last_message, ToolMessage) and 
            "saved" in last_message.content.lower() and
            "document" in last_message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint

    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node(
    agent_name,
    model_call,
)

tool_node = ToolNode(tools=tools)
graph.add_node("my_tools", tool_node)
graph.set_entry_point(agent_name)
graph.add_edge(agent_name, "my_tools")
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


def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()