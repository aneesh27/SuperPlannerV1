from langchain import hub
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing import Annotated, List, Tuple, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
import asyncio
from dotenv import load_dotenv
import os
from tools import dataTools, analysisTools, reportTools, allTools
from IPython.display import Image, display

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000, api_key=OPENAI_API_KEY)

# Define Planning and Execution States
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    steps: List[str] = Field(description="Steps to follow in sorted order.")

class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan] = Field(description="Decide on next action.")

# Planning Prompts
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """For the given objective, create a simple step-by-step plan. 
    Ensure each step contains all necessary information without skipping."""),
    ("placeholder", "{messages}")
])
planner = planner_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Plan)

replanner_prompt = ChatPromptTemplate.from_template(
    """Update the plan based on completed steps. Only add steps that still need to be done.
    If the task is complete, respond with the final answer.
    
    Your original objective: {input}
    
    Original plan: {plan}
    Completed steps: {past_steps}
    """
)
replanner = replanner_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0).with_structured_output(Act)

# Supervisor Workflow
members = ["researcher", "reporter"]
options = members + ["FINISH"]
system_prompt = f"You are a supervisor managing workers: {members}. Assign tasks and decide when the work is complete."

class Router(TypedDict):
    next: Literal["researcher", "reporter", "FINISH"]

class State(TypedDict, total=False):
    messages: List[HumanMessage]
    next: str

# def supervisor_node(state: State) -> Command[Literal["researcher", "reporter", "__end__"]]:
#     response = llm.with_structured_output(Router).invoke([
#         {"role": "system", "content": system_prompt},
#         *state["messages"]
#     ])
#     goto = response["next"]
#     return Command(goto=END if goto == "FINISH" else goto, update={"next": goto})
def supervisor_node(state: State) -> Command[Literal["researcher", "reporter", "__end__"]]:
    messages = state.get("messages", [])  # Ensure messages exists
    response = llm.with_structured_output(Router).invoke([
        {"role": "system", "content": system_prompt},
        *messages  # Now safe to unpack
    ])
    goto = response["next"]
    return Command(goto=END if goto == "FINISH" else goto, update={"next": goto})

research_agent = create_react_agent(llm, analysisTools, prompt="You are a research agent using appropriate tools.")
report_agent = create_react_agent(llm, reportTools, prompt="You are a reporting agent summarizing and reporting findings.")

def research_node(state: State) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    if "messages" not in result or not result["messages"]:
        raise ValueError("research_agent returned no messages")
    last_message = result["messages"][-1].content if result["messages"] else "No response from research agent."
    return Command(update={"messages": [HumanMessage(content=last_message, name="researcher")]}, goto="supervisor")

# def research_node(state: State) -> Command[Literal["supervisor"]]:
#     result = research_agent.invoke(state)
#     return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="researcher")]}, goto="supervisor")

def report_node(state: State) -> Command[Literal["supervisor"]]:
    result = report_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="reporter")]}, goto="supervisor")

# Workflow Definition
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", lambda state: {"plan": planner.invoke({"messages": [("user", state["input"])]}).steps})
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", research_node)
workflow.add_node("reporter", report_node)
workflow.add_node("replan", lambda state: {"response": replanner.invoke(state).action.response} if isinstance(replanner.invoke(state).action, Response) else {"plan": replanner.invoke(state).action.steps})
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "supervisor")
workflow.add_edge("supervisor", "researcher")
workflow.add_edge("supervisor", "reporter")
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("reporter", "supervisor")
workflow.add_edge("supervisor", "replan")
workflow.add_edge("replan", "supervisor")
workflow.add_edge("replan", END)

app = workflow.compile()

try:
    # Show the workflow
    img = Image(app.get_graph().draw_mermaid_png())
    with open("diagrams/superPlanV1.png", "wb") as png:
        png.write(img.data)
    # display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass


async def main():
    config = {"recursion_limit": 10}
    inputs = {"input": "How do I analyze sales data and create a report?", "messages": []}  # Add messages key
    
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)
# async def main():
#     config = {"recursion_limit": 10}
#     inputs = {"input": "How do I analyze sales data and create a report?"}
    
#     async for event in app.astream(inputs, config=config):
#         for k, v in event.items():
#             if k != "__end__":
#                 print(v)

asyncio.run(main())