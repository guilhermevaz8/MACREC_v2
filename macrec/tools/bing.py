from langchain_community.retrievers.bing import BingSearchRetriever
from macrec.tools.base import RetrievalTool

class Bing(RetrievalTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k: int = self.config.get('top_k', 5)
        api_key: str = self.config.get('api_key', 'your_api_key_here')
        self.retriever = BingSearchRetriever(api_key=api_key, top_k_results=self.top_k)

    def search(self, query: str) -> str:
        # Similar ao Wikipedia.search()
