import json
from dataclasses import dataclass
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from remove_think import remove_think_tags

# Initialize the LLM using ChatGroq
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Extend the state to include flags for calling subordinate agents.
@dataclass
class KqlQueryState:
    input_prompt: str = ""
    select_clause: str = ""
    where_clause: str = ""
    groupby_clause: str = ""
    aggregated_clauses: str = ""
    final_query: str = ""
    call_where: bool = False
    call_groupby: bool = False

@dataclass
class KqlQueryStateInput:
    input_prompt: str

@dataclass
class KqlQueryStateOutput:
    final_query: str

# Supervisor agent instructions.
supervisor_instructions = (
    "Your goal is to decide which components of a KQL query are needed based on the input.\n"
    "Examine the input: {input_prompt}\n"
    "If the input implies a filtering condition (i.e. a WHERE clause is needed), set call_where to true; otherwise, false.\n"
    "If the input implies grouping (i.e. a GROUP BY clause is needed), set call_groupby to true; otherwise, false.\n"
    "Return your decision as a JSON object with keys 'call_where' and 'call_groupby'."
)

def supervisor_agent(state: KqlQueryState) -> dict:
    prompt = supervisor_instructions.format(input_prompt=state.input_prompt)
    system_msg = SystemMessage(content=prompt)
    human_msg = HumanMessage(content="Decide which KQL clauses are needed.")
    result = llm.invoke([system_msg, human_msg])
    try:
        decision = json.loads(result.content)
    except json.JSONDecodeError:
        decision = {"call_where": False, "call_groupby": False}
    state.call_where = decision.get("call_where", False)
    state.call_groupby = decision.get("call_groupby", False)
    return {"call_where": state.call_where, "call_groupby": state.call_groupby}

# Instructions for each clause writer.
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
    # Concatenate only the generated clauses. If a clause was skipped, its value remains empty.
    aggregated = (
        f"SELECT clause: {state.select_clause}\n"
        f"WHERE clause: {state.where_clause}\n"
        f"GROUP BY clause: {state.groupby_clause}"
    )
    state.aggregated_clauses = aggregated
    return {"aggregated_clauses": state.aggregated_clauses}

def finalize_query(state: KqlQueryState) -> dict:
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

# Routing functions that use the supervisor's decisions.
def route_after_select(state: KqlQueryState, config: RunnableConfig) -> str:
    if state.call_where:
        return "generate_where"
    else:
        # If no WHERE clause is needed, go directly to GROUP BY.
        return "generate_groupby"

def route_after_where(state: KqlQueryState, config: RunnableConfig) -> str:
    if state.call_groupby:
        return "generate_groupby"
    else:
        return "aggregate_clauses"

# Build the StateGraph with nodes and edges.
builder = StateGraph(KqlQueryState, input=KqlQueryStateInput, output=KqlQueryStateOutput)
builder.add_node("supervisor", supervisor_agent)
builder.add_node("generate_select", generate_select_clause)
builder.add_node("generate_where", generate_where_clause)
builder.add_node("generate_groupby", generate_groupby_clause)
builder.add_node("aggregate_clauses", aggregate_clauses)
builder.add_node("finalize_query", finalize_query)

# Define edges:
# Start with the supervisor agent.
builder.add_edge(START, "supervisor")
# Then always call the SELECT clause writer.
builder.add_edge("supervisor", "generate_select")
# Conditionally route after SELECT:
builder.add_conditional_edges("generate_select", route_after_select)
# Conditionally route after WHERE (if it was run):
builder.add_conditional_edges("generate_where", route_after_where)
# If GROUP BY is executed, then go to aggregation.
builder.add_edge("generate_groupby", "aggregate_clauses")
# Finally, finalize the query.
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
    # Example with all clauses needed:
    input_prompt1 = "Get all sales records where revenue is greater than 1000 grouped by region."
    final_state1 = graph.invoke(KqlQueryState(input_prompt=input_prompt1))
    print("Final KQL Query for input 1:")
    print(final_state1["final_query"])
    print("Removing think tags...")
    print(remove_think_tags(final_state1["final_query"]))
    
    # Example with only a SELECT clause (no filtering or grouping):
    input_prompt2 = "Get all sales records."
    final_state2 = graph.invoke(KqlQueryState(input_prompt=input_prompt2))
    print("\nFinal KQL Query for input 2:")
    print(final_state2["final_query"])
    print("Removing think tags...")
    print(remove_think_tags(final_state2["final_query"]))
