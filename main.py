from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# Structured output model
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Load local LLaMA 3
llm = ChatOllama(model="llama3", temperature=0)

def run_research(query: str):
    chat_history = []
    tools_used = []
    sources = []

    # First prompt — ask model how to research
    system_prompt = (
        "You are a research assistant. "
        "You can use the following tools: search, wiki, save. "
        "When you need information, say: TOOL: <tool_name> | <input>. "
        "Otherwise, respond with the final research summary."
    )

    while True:
        # Build conversation
        messages = [{"role": "system", "content": system_prompt}]
        for role, content in chat_history:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        # Ask the LLM
        response = llm.invoke(messages)
        content = response.content.strip()
        print("\nAssistant:", content)

        if content.lower().startswith("tool:"):
            try:
                _, rest = content.split(":", 1)
                tool_name, tool_input = rest.strip().split("|", 1)
                tool_name = tool_name.strip().lower()
                tool_input = tool_input.strip()

                if tool_name == "search":
                    result = search_tool.run(tool_input)
                elif tool_name == "wiki":
                    result = wiki_tool.run(tool_input)
                elif tool_name == "save":
                    result = save_tool.run(tool_input)
                else:
                    result = f"Unknown tool: {tool_name}"

                tools_used.append(tool_name)
                sources.append(tool_input)
                print(f"\n[Tool {tool_name} Output]:\n{result}")

                # Feed tool result back into chat
                chat_history.append(("assistant", content))
                chat_history.append(("user", f"Tool result:\n{result}"))

            except Exception as e:
                print(f"Error running tool: {e}")
                break
        else:
            # Final structured result
            try:
                structured = ResearchResponse(
                    topic=query,
                    summary=content,
                    sources=sources,
                    tools_used=tools_used
                )
                print("\nFinal Structured Output:", structured)
            except Exception as e:
                print("Error structuring output:", e)
            break

if __name__ == "__main__":
    query = input("What can I help you research? ")
    run_research(query)
