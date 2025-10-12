import asyncio
from dotenv import load_dotenv
import os

load_dotenv()


async def main():
    print("Hello from mcp-implementation!")
    print(os.environ["OPENAI_API_KEY"])

if __name__ == "__main__":
    asyncio.run(main())