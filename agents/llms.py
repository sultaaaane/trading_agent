from langchain_openai import ChatOpenAI


junior_analyst_llm = ChatOpenAI(
    model="llama3",
    temperature=0.7,
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama",
)

cio_llm = ChatOpenAI(
    model="qwen2:1.5b",
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
    model="mixtral",
    temperature=0.0,
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama",
)
