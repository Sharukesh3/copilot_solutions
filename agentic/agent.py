import json
from dataclasses import dataclass
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from agentic.remove_think import remove_think_tags

# Initialize the LLM using ChatGroq
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define state classes to hold the input and outputs.
@dataclass
class KqlQueryState:
    input_prompt: str = ""
    select_clause: str = ""
    where_clause: str = ""
    groupby_clause: str = ""
    aggregated_clauses: str = ""
    final_query: str = ""

@dataclass
class KqlQueryStateInput:
    input_prompt: str

@dataclass
class KqlQueryStateOutput:
    final_query: str

# Instructions for each clause writer
select_clause_instructions = (
    "Your goal is to generate the SELECT clause for a KQL query.\n"
    "Input: {input_prompt}\n"
    "Return your clause as a plain text string."
)
where_clause_instructions = (
    "Your goal is to generate the WHERE clause for a KQL query.\n"
    "Input: {input_prompt}\n"
    "Return your clause as a plain text string."
)
groupby_clause_instructions = (
    "Your goal is to generate the GROUP BY clause for a KQL query.\n"
    "Input: {input_prompt}\n"
    "Return your clause as a plain text string."
)

def generate_select_clause(state: KqlQueryState) -> dict:
    system_msg = SystemMessage(content=select_clause_instructions.format(input_prompt=state.input_prompt))
    human_msg = HumanMessage(content=f"Generate the SELECT clause for the following KQL query: {state.input_prompt}")
    result = llm.invoke([system_msg, human_msg])
    state.select_clause = result.content.strip()
    return {"select_clause": state.select_clause}

def generate_where_clause(state: KqlQueryState) -> dict:
    system_msg = SystemMessage(content=where_clause_instructions.format(input_prompt=state.input_prompt))
    human_msg = HumanMessage(content=f"Generate the WHERE clause for the following KQL query: {state.input_prompt}")
    result = llm.invoke([system_msg, human_msg])
    state.where_clause = result.content.strip()
    return {"where_clause": state.where_clause}

def generate_groupby_clause(state: KqlQueryState) -> dict:
    system_msg = SystemMessage(content=groupby_clause_instructions.format(input_prompt=state.input_prompt))
    human_msg = HumanMessage(content=f"Generate the GROUP BY clause for the following KQL query: {state.input_prompt}")
    result = llm.invoke([system_msg, human_msg])
    state.groupby_clause = result.content.strip()
    return {"groupby_clause": state.groupby_clause}

def aggregate_clauses(state: KqlQueryState) -> dict:
    # First, concatenate the three clauses into an aggregated string.
    aggregated = f"SELECT clause: {state.select_clause}\nWHERE clause: {state.where_clause}\nGROUP BY clause: {state.groupby_clause}"
    state.aggregated_clauses = aggregated
    return {"aggregated_clauses": state.aggregated_clauses}

def finalize_query(state: KqlQueryState) -> dict:
    # Use the aggregated clauses and ask the LLM to combine them into a final KQL query.
    final_system_msg = SystemMessage(
        content=(
            "You are an assistant whose goal is to finalize a KQL query. "
            "Given the following components, generate a complete and syntactically correct KQL query:\n"
            "{aggregated_clauses}\n"
            "Return your final query as plain text."
        ).format(aggregated_clauses=state.aggregated_clauses)
    )
    final_human_msg = HumanMessage(content="Finalize the KQL query.")
    result = llm.invoke([final_system_msg, final_human_msg])
    state.final_query = result.content.strip()
    return {"final_query": state.final_query}

# Build the StateGraph with nodes and edges.
builder = StateGraph(KqlQueryState, input=KqlQueryStateInput, output=KqlQueryStateOutput)
builder.add_node("generate_select", generate_select_clause)
builder.add_node("generate_where", generate_where_clause)
builder.add_node("generate_groupby", generate_groupby_clause)
builder.add_node("aggregate_clauses", aggregate_clauses)
builder.add_node("finalize_query", finalize_query)

# Define edges so that the nodes execute sequentially:
builder.add_edge(START, "generate_select")
builder.add_edge("generate_select", "generate_where")
builder.add_edge("generate_where", "generate_groupby")
builder.add_edge("generate_groupby", "aggregate_clauses")
builder.add_edge("aggregate_clauses", "finalize_query")
builder.add_edge("finalize_query", END)

# Compile the graph.
graph = builder.compile()

# Optionally display the graph visualization if running in an IPython environment.
from IPython.display import Image, display
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

def agent_pipeline(input_prompt: str) -> str:
    """
    Runs the agent pipeline to generate a KQL query based on the input prompt.

    Args:
        input_prompt (str): The input prompt.

    Returns:
        str: The final KQL query.
    """
    initial_state = KqlQueryState(input_prompt=input_prompt)
    final_state = graph.invoke(initial_state)
    return remove_think_tags(final_state["final_query"])

# Example usage:
if __name__ == "__main__":
    initial_state = KqlQueryState(input_prompt="Get all sales records where revenue is greater than 1000 grouped by region.")
    final_state = graph.invoke(initial_state)
    print("Final KQL Query:")
    print(final_state["final_query"])
    print("Removing think tags...")
    print(remove_think_tags(final_state["final_query"]))
