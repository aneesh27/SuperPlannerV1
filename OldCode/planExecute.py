# from langchain_community.tools.tavily_search import TavilySearchResults

from langchain import hub
from langgraph.graph import END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

import operator
from typing import Annotated, List, Tuple, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from tools import dataTools, analysisTools, reportTools, allTools
import asyncio
from IPython.display import Image, display
from dotenv import load_dotenv
import os
from IPython.display import Image, display

# tools = [TavilySearchResults(max_results=3)]
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
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
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
ONLY come up with a plan if the OBJECTIVE is NOT achieved. Or else, just respond back to the user with the result
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
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


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)
# Add the execution step
workflow.add_node("agent", execute_step)
# Add a replan node
workflow.add_node("replan", replan_step)
workflow.add_edge(START, "planner")
# From plan we go to agent
workflow.add_edge("planner", "agent")
# From agent, we replan
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)


# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

try:
    # Show the workflow
    img = Image(app.get_graph().draw_mermaid_png())
    with open("diagrams/planExecute.png", "wb") as png:
        png.write(img.data)
    # display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

async def main():
    config = {"recursion_limit": 20}
    # inputs = {"input": "Give me a very simple and short summary about dragons"}
    inputs = {"input": "what is the hometown of the mens 1980 Australia open winner?"}
    # what is the hometown of the mens 1960 Australia open winner?
    
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

asyncio.run(main())










# # from langchain_community.tools.tavily_search import TavilySearchResults

# from langchain import hub
# from langgraph.graph import END
# from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.tools.tavily_search import TavilySearchResults

# import operator
# from typing import Annotated, List, Tuple, Union, Literal
# from typing_extensions import TypedDict
# from pydantic import BaseModel, Field

# import asyncio
# from IPython.display import Image, display
# from dotenv import load_dotenv
# import os

# # tools = [TavilySearchResults(max_results=3)]
# tools = []

# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, api_key=OPENAI_API_KEY)

# # Choose the LLM that will drive the agent
# prompt = "You are a helpful assistant."
# agent_executor = create_react_agent(llm, tools, prompt=prompt)

# # agent_executor.invoke({"messages": [("user", "who is the winnner of the us open")]})

# class PlanExecute(TypedDict):
#     input: str
#     plan: List[str]
#     past_steps: Annotated[List[Tuple], operator.add]
#     response: str

# class Plan(BaseModel):
#     """Plan to follow in future"""

#     steps: List[str] = Field(
#         description="different steps to follow, should be in sorted order"
#     )

# planner_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """For the given objective, come up with a simple step by step plan. \
# This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
# The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# )
# planner = planner_prompt | ChatOpenAI(
#     model="gpt-3.5-turbo", temperature=0
# ).with_structured_output(Plan)

# # planner.invoke(
# #     {
# #         "messages": [
# #             ("user", "what is the hometown of the current Australia open winner?")
# #         ]
# #     }
# # )


# class Response(BaseModel):
#     """Response to user."""

#     response: str


# class Act(BaseModel):
#     """Action to perform."""

#     action: Union[Response, Plan] = Field(
#         description="Action to perform. If you want to respond to user, use Response. "
#         "If you need to further use tools to get the answer, use Plan."
#     )


# replanner_prompt = ChatPromptTemplate.from_template(
#     """For the given objective, come up with a simple step by step plan. \
# ONLY come up with a plan if the OBJECTIVE is NOT achieved. Or else, just respond back to the user with the result
# This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
# The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

# Your objective was this:
# {input}

# Your original plan was this:
# {plan}

# You have currently done the follow steps:
# {past_steps}

# Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
# )


# replanner = replanner_prompt | ChatOpenAI(
#     model="gpt-3.5-turbo", temperature=0
# ).with_structured_output(Act)


# async def execute_step(state: PlanExecute):
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


# async def plan_step(state: PlanExecute):
#     plan = await planner.ainvoke({"messages": [("user", state["input"])]})
#     return {"plan": plan.steps}


# async def replan_step(state: PlanExecute):
#     output = await replanner.ainvoke(state)
#     if isinstance(output.action, Response):
#         return {"response": output.action.response}
#     else:
#         return {"plan": output.action.steps}


# def should_end(state: PlanExecute):
#     if "response" in state and state["response"]:
#         return END
#     else:
#         return "agent"


# from langgraph.graph import StateGraph, START

# workflow = StateGraph(PlanExecute)

# # Add the plan node
# workflow.add_node("planner", plan_step)
# # Add the execution step
# workflow.add_node("agent", execute_step)
# # Add a replan node
# workflow.add_node("replan", replan_step)
# workflow.add_edge(START, "planner")
# # From plan we go to agent
# workflow.add_edge("planner", "agent")
# # From agent, we replan
# workflow.add_edge("agent", "replan")
# workflow.add_conditional_edges(
#     "replan",
#     # Next, we pass in the function that will determine which node is called next.
#     should_end,
#     ["agent", END],
# )

# # Finally, we compile it!
# # This compiles it into a LangChain Runnable,
# # meaning you can use it as you would any other runnable
# app = workflow.compile()

# async def main():
#     config = {"recursion_limit": 50}
#     inputs = {"input": "what is the hometown of the mens 1960 Australia open winner?"}
    
#     async for event in app.astream(inputs, config=config):
#         for k, v in event.items():
#             if k != "__end__":
#                 print(v)

# asyncio.run(main())