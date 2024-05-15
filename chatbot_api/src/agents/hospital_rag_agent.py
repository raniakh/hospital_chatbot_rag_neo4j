import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from chains.hospital_review_chain import reviews_vector_chain
from chains.hospital_cypher_chain import hospital_cypher_chain
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

load_dotenv()

HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")  # the LLM that will act as your agent’s brain, deciding
# which tools to call and what inputs to pass them.

hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")