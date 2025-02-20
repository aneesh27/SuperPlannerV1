from pydantic import BaseModel, Field
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from tools import dataTools, analysisTools, reportTools
from dotenv import load_dotenv
import os
from IPython.display import Image, display

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, api_key=OPENAI_API_KEY)


members = ["researcher", "reporter"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal["researcher", "reporter", "FINISH"]



class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal["researcher", "reporter", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})



research_agent = create_react_agent(
    llm, analysisTools, prompt="You are a research agent and you have dummy tools available to you. The first tool is called statTests runs a p-test on the data. The second tool genrates a matplot visualization. Pick the right tool based on the subtask."
)


def research_node(state: State) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
report_agent = create_react_agent(llm, reportTools, prompt="You are a reporting agent and you have dummy tools available to you. The first tool is called summarize and summarizes the findings. The second tool is called report findings and reports your findings in detail. Pick the right tool based on the subtask.")


def report_node(state: State) -> Command[Literal["supervisor"]]:
    result = report_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="reporter")
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("reporter", report_node)
graph = builder.compile()

finalOutput = ""
for s in graph.stream(
    {"messages": [("user", "Run tests and then give me a summary of potential findings.")]}, subgraphs=True
):
    print("\n=== Step Execution ===")
    
    # Extract agent name
    if isinstance(s, tuple) and len(s) > 1:
        agent_key = list(s[1].keys())[0]
        print(f"Active Agent: {agent_key}")

        # Extract messages
        if "messages" in s[1][agent_key]:
            messages = s[1][agent_key]["messages"]
            for msg in messages:
                if hasattr(msg, "content"):
                    print(f"Message: {msg.content}")
                else:
                    print(f"Tool Response: {msg}")

        # Extract tool usage
        if "tools" in s[1]:
            for tool_msg in s[1]["tools"]["messages"]:
                print(f"Tool Used: {tool_msg.name}, Output: {tool_msg.content}")

        # Extract errors
        if "errors" in s[1]:
            print("Errors: ", s[1]["errors"])
    
    print("----")
    # finalOutput = finalOutput + s + "\n----"
    # print(finalOutput)
    print(s)
    print("----")