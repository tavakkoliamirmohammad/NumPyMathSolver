import os
import re
import numpy as np
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
import ollama
import io
import sys
import traceback

# Initialize Ollama
client = ollama.Client()

# ----------------------
# Model Configuration
# ----------------------
MODEL_NAME = "qwen2.5-coder:32b"
TEMPERTURE = 0.3

class GraphState(TypedDict):
    question: str
    math_related: bool
    generated_code: str
    verification_result: dict
    refinements: List[str]
    final_answer: str

# ----------------------
# Prompts
# ----------------------


MATH_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Determine if the following question requires mathematical computation. 
Consider problems involving arithmetic, algebra, calculus, statistics, trigonometry, or numerical analysis as math-related.
Question: {question}
Respond ONLY with 'YES' or 'NO'."""),
])

CODE_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Python programmer. Generate NumPy code to solve this math problem:

{question}

Guidelines:
1. Import only numpy as np
2. Use vectorized operations
3. Store final result in a variable called 'result'
4. Print the result but do not include the print statement within any generated function; it should be outside.
5. Do not use input() calls for exmaple float(input("Enter a number: ")).
6. Try to minimize the number of lines of code.
7. Don't include any comments unrelated to the problem.
8. The code should be as generalized as possible and the user prompt should be input to the generated function.
9. If the user prompt is generic, generate the code and provide an example of how to use it.
10. In the example provided don't get any input() calls.
11. If the user has any specific requirements, include them in the code."""),
])

VERIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Verify the correctness of this Python code for solving: {question}

Code:
{code}

Checklist:
1. Syntax validity
2. Logical correctness
3. Numerical stability
4. Alignment with problem requirements
5. Code is parameterized and can be reused for different inputs.
     
If the code is correct and aligns with the problem requirements, return the state 'APPROVED'. Otherwise, provide detailed feedback including errors such as the presence of input() calls."""),
])

REFINEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Refine this code based on feedback. Original problem: {question}

Original Code:
{code}

Feedback:
{feedback}

Generate improved code addressing all issues while ensuring no input() calls are present."""),
])

# ----------------------
# Nodes
# ----------------------


def check_math_question(state: GraphState):
    response = client.chat(
        model=MODEL_NAME,
        options={"temperature": TEMPERTURE},
        messages=[{
            "role": "user",
            "content": MATH_CHECK_PROMPT.format(question=state["question"])
        }]
    )
    is_math = 'YES' in response['message']['content'].strip().upper()
    return {"math_related": is_math}


def generate_initial_code(state: GraphState):
    response = client.chat(
        model=MODEL_NAME,
        options={"temperature": TEMPERTURE},
        messages=[{
            "role": "user",
            "content": CODE_GENERATION_PROMPT.format(question=state["question"])
        }]
    )
    code = extract_code(response['message']['content'])
    return {"generated_code": code}


def verify_code(state: GraphState):
    # Check for forbidden input() calls before verification
    code = state["generated_code"]
    response = client.chat(
        model=MODEL_NAME,
        options={"temperature": TEMPERTURE},
        messages=[{
            "role": "user",
            "content": VERIFICATION_PROMPT.format(
                question=state["question"],
                code=code
            )
        }]
    )
    feedback = response['message']['content']
    approved = 'APPROVED' in feedback.upper()
    return {"verification_result": {"approved": approved, "feedback": feedback}}


def refine_code(state: GraphState):
    response = client.chat(
        model=MODEL_NAME,
        options={"temperature": TEMPERTURE},
        messages=[{
            "role": "user",
            "content": REFINEMENT_PROMPT.format(
                question=state["question"],
                code=state["generated_code"],
                feedback=state["verification_result"]["feedback"]
            )
        }]
    )
    new_code = extract_code(response['message']['content'])
    return {"generated_code": new_code}


def execute_code(state: GraphState):
    try:
        loc = {"np": np}
        # Capture printed output
        output_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output_capture

        try:
            exec(state["generated_code"], loc)
        finally:
            sys.stdout = old_stdout

        printed_output = output_capture.getvalue().strip()
        final_text = (
            f"Final Code:\n\n```\n{state['generated_code']}\n```\n\n"
            f"Printed Output:\n{printed_output}"
        )
        return {"final_answer": final_text, "execution_failed": False}
    except Exception as e:
        error_message = str(e)
        # Capture the full stack trace as a string.
        stack_trace = traceback.format_exc()
        # Append both the error message and the stack trace to the feedback.
        updated_feedback = f"\nExecution Error: {error_message}\nStack Trace:\n{stack_trace}"
        final_text = (
            f"Final Code:\n{state['generated_code']}\n\n"
            f"Error in execution: {error_message}\n"
            f"Stack Trace:\n{stack_trace}"
        )
        return {
            "final_answer": final_text,
            "execution_failed": True,
            "verification_result": {
                "approved": False,
                "feedback": updated_feedback
            }
        }


def regular_response(state: GraphState):
    # If the question is not coding-related, refuse to answer.
    return {"final_answer": "I'm sorry, I only answer coding questions."}

# ----------------------
# Helpers
# ----------------------


def extract_code(text: str) -> str:
    code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
    return code_blocks[0] if code_blocks else text


def route_based_on_math(state: GraphState):
    if state["math_related"]:
        return "math_pipeline"
    return "regular_response"


def route_based_on_verification(state: GraphState):
    if state["verification_result"]["approved"]:
        return "execute_code"
    return "refine_code"


def route_based_on_execution(state: GraphState):
    if state.get("execution_failed", False):
        return "refine_code"
    return "END"

# ----------------------
# Graph Setup
# ----------------------


workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("check_math", check_math_question)
workflow.add_node("math_pipeline", generate_initial_code)
workflow.add_node("verify_code", verify_code)
workflow.add_node("refine_code", refine_code)
workflow.add_node("execute_code", execute_code)
workflow.add_node("regular_response", regular_response)

# Set entry point
workflow.set_entry_point("check_math")

# Add edges
workflow.add_conditional_edges(
    "check_math",
    route_based_on_math,
    {
        "math_pipeline": "math_pipeline",
        "regular_response": "regular_response"
    }
)

workflow.add_edge("math_pipeline", "verify_code")
workflow.add_conditional_edges(
    "verify_code",
    route_based_on_verification,
    {
        "execute_code": "execute_code",
        "refine_code": "refine_code"
    }
)

workflow.add_edge("refine_code", "verify_code")
# Instead of routing directly to END, check if execution succeeds
workflow.add_conditional_edges(
    "execute_code",
    route_based_on_execution,
    {
        "refine_code": "refine_code",
        "END": END
    }
)
workflow.add_edge("regular_response", END)

graph = workflow.compile()
