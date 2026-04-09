from langchain_openai import ChatOpenAI


junior_analyst_llm = ChatOpenAI(
    model="Qwen/Qwen2-1.5B-Instruct",
    temperature=0.7,
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama",
)

cio_llm = ChatOpenAI(
    model="Qwen/Qwen2-1.5B-Instruct",
    temperature=0.0,
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama",
)

# Placeholder for our model-under-training
senior_strategist_llm = ChatOpenAI(
    model="senior_strategist_llm",
    temperature=0.1,
    openai_api_base="http://placeholder-will-be-replaced:8000/v1",
    openai_api_key="dummy_key",
)


review_board_llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.0,
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama",
)
