import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_agent_runner(llm, system_prompt, tools):
    """A factory function to create a runnable agent executor."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    enable_tool_binding = os.environ.get("ENABLE_TOOL_BINDING", "0") == "1"
    if tools and enable_tool_binding:
        return prompt | llm.bind_tools(tools)
    return prompt | llm
