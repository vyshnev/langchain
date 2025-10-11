from typing import List, TypedDict

from langchain_core.outputs import generation


class GraphState(TypedDict):
     """Represents the state of the graph.
     Attributes:
         question: question
         generation: LLM generation
         web_search: whether to add search
         documents: List of documents
    """

     question: str
     generation: int
     web_search: bool
     documents: List[str]