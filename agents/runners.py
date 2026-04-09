from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_agent_runner(llm, system_prompt, tools):
    """A factory function to create a runnable agent executor."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # Bind the tools to the LLM, making them callable by the agent
    return prompt | llm.bind_tools(tools)
