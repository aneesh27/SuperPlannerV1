# LangChain and LangGraph
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

# Standard Library
import operator
import asyncio
import os

# Typing and Data Validation
from typing import Annotated, List, Tuple, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Tools and Environment
from tools import dataTools, analysisTools, reportTools, allTools
from dotenv import load_dotenv
from IPython.display import Image, display


tools = []

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000, api_key=OPENAI_API_KEY)

# Choose the LLM that will drive the agent
prompt = "You are a helpful assistant."
agent_executor = create_react_agent(llm, tools, prompt=prompt)

# agent_executor.invoke({"messages": [("user", "who is the winnner of the us open")]})

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a very simple step by step plan. \
This plan should involve individual tasks that uses the tools: run ptest on data, generate matplot visualization, summarize findings, report findings in detail. And that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).with_structured_output(Plan)

# planner.invoke(
#     {
#         "messages": [
#             ("user", "what is the hometown of the current Australia open winner?")
#         ]
#     }
# )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a a very simple step by step plan. \
ONLY come up with a plan if the OBJECTIVE is NOT achieved. BUT IF the FINDINGS have been reported well enough, then just respond back to the user with the result
This plan should involve individual tasks that uses the tools: run test on data, generate matplot visualization, summarize findings, report findings in detail. And that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).with_structured_output(Act)

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
    past_steps: Annotated[List[Tuple], operator.add] = []

# async def execute_step(state: PlanExecute) -> Command[Literal["researcher", "reporter", "replan"]]:
#     plan = state["plan"]
#     plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
#     task = plan[0]
#     task_formatted = f"""For the following plan:
# {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
#     agent_response = await agent_executor.ainvoke(
#         {"messages": [("user", task_formatted)]}
#     )
#     return {
#         "past_steps": [(task, agent_response["messages"][-1].content)],
#     }

# def supervisor_node(state: State) -> Command[Literal["researcher", "reporter", "replan"]]:
#     messages = [
#         {"role": "system", "content": system_prompt},
#     ] + state["messages"]
#     response = llm.with_structured_output(Router).invoke(messages)
#     goto = response["next"]
#     if goto == "FINISH":
#         goto = "replan"

#     return Command(goto=goto, update={"next": goto})

research_agent = create_react_agent(
    llm, analysisTools, prompt="You are a research agent and you have dummy tools available to you. The first tool is called statTests runs a p-test on the data. The second tool genrates a matplot visualization. Pick the right tool based on the subtask."
)
def supervisor_node(state: State) -> Command[Literal["researcher", "reporter", "replan"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    
    if goto == "FINISH":
        goto = "replan"

    return Command(
        goto=goto,
        update={
            "next": goto,
            "past_steps": state["past_steps"]  # Ensure past_steps persists
        }
    )


# def research_node(state: State) -> Command[Literal["supervisor"]]:
#     result = research_agent.invoke(state)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result["messages"][-1].content, name="researcher")
#             ]
#         },
#         goto="supervisor",
#     )
def research_node(state: State) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    step_result = result["messages"][-1].content
    
    return Command(
        update={
            "messages": [HumanMessage(content=step_result, name="researcher")],
            "past_steps": state["past_steps"] + [("research", step_result)]
        },
        goto="supervisor",
    )

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
report_agent = create_react_agent(llm, reportTools, prompt="You are a reporting agent and you have dummy tools available to you. The first tool is called summarize and summarizes the findings. The second tool is called report findings and reports your findings in detail. Pick the right tool based on the subtask.")


# def report_node(state: State) -> Command[Literal["supervisor"]]:
#     result = report_agent.invoke(state)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result["messages"][-1].content, name="reporter")
#             ]
#         },
#         goto="supervisor",
#     )

def report_node(state: State) -> Command[Literal["supervisor"]]:
    result = report_agent.invoke(state)
    step_result = result["messages"][-1].content

    return Command(
        update={
            "messages": [HumanMessage(content=step_result, name="reporter")],
            "past_steps": state["past_steps"] + [("report", step_result)]
        },
        goto="supervisor",
    )

async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


# async def replan_step(state: PlanExecute):
#     output = await replanner.ainvoke(state)
#     if isinstance(output.action, Response):
#         return {"response": output.action.response}
#     else:
#         return {"plan": output.action.steps}
async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke({
        "input": state["input"],
        "plan": state["plan"],
        "past_steps": state["past_steps"]  # Ensure replanner gets past_steps
    })
    
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {
            "plan": output.action.steps,
            "past_steps": state["past_steps"]  # Preserve past_steps
        }


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "supervisor"


from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)
# Add the execution step
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", research_node)
workflow.add_node("reporter", report_node)
# workflow.add_node("agent", execute_step)
# Add a replan node
workflow.add_node("replan", replan_step)
workflow.add_edge(START, "planner")
# From plan we go to agent
workflow.add_edge("planner", "supervisor")
# From agent, we replan
# workflow.add_edge("supervisor", "replan")
workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["supervisor", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

try:
    # Show the workflow
    img = Image(app.get_graph().draw_mermaid_png())
    with open("diagrams/superPlanV3.png", "wb") as png:
        png.write(img.data)
    # display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

async def main():
    config = {"recursion_limit": 20}
    inputs = {"input": "The hypothetical data given shows how water seeping into ground affects construction and laying of cement over the soil. Pretend there is data and give me a plausible response."}
    # inputs = {"input": "what is the hometown of the mens 1980 Australia open winner?"}
    # what is the hometown of the mens 1960 Australia open winner?
    
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

asyncio.run(main())