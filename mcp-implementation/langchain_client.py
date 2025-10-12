import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

llm = ChatOpenAI()


stdio_server_params = StdioServerParameters(
    command="python",
    args=[r"D:\Project\LangChain\mcp-implementation\server\math_server.py"],
)

async def main():
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("Session Initialized")
            tools = await load_mcp_tools(session)
            print(tools)
            agent = create_react_agent(llm, tools)

            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 72 + 2 * 3")]})
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
