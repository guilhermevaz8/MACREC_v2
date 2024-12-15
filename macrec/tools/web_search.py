from langchain_core.documents import Document
from tavily import TavilyClient
from macrec.tools.base import RetrievalTool

class WebSearch(RetrievalTool):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.api_key: str = self.config.get('api_key', 'KEY')
        self.top_k: int = self.config.get('top_k', 3)
        self.tavily_client = TavilyClient(api_key=self.api_key)
        self.cache = {}

    def reset(self) -> None:
        self.cache = {}

    def _format_documents(self, documents: list[dict]) -> str:
        titles = []
        summary = []
        for document in documents:
            title = document.get('title', 'Untitled')
            if title not in self.cache:
                self.cache[title] = {
                    'document': document,
                    'lookup_index': {},
                }
            titles.append(title)
            summary_content = document.get('snippet', '')  # Tavily can provide a snippet
            if len(summary_content.split()) > 20:
                summary_content = ' '.join(summary_content.split()[:20]) + '...'
            summary.append(summary_content)
        return ', '.join([f'{title} ({summary})' for title, summary in zip(titles, summary)])
    
    def search(self, query: str) -> str:
        try:
            results = self.tavily_client.search(query=query)
            return results   
        except Exception as e:
            return f'Error occurred during search: {e}'
    
    def lookup(self, title: str, term: str) -> str:
        if title not in self.cache:
            return 'No title found in search results.'
        document = self.cache[title]['document']
        if term not in self.cache[title]['lookup_index']:
            self.cache[title]['lookup_index'][term] = 0
        else:
            self.cache[title]['lookup_index'][term] += 1
        lookups = [p for p in document.get('content', '').split("\n\n") if term.lower() in p.lower()]
        if len(lookups) == 0:
            return f'No results for term {term} in document {title}.'
        elif self.cache[title]['lookup_index'][term] >= len(lookups):
            return f'No more results for term {term} in document {title}.'
        else:
            result_prefix = f'(Result {self.cache[title]["lookup_index"][term] + 1} / {len(lookups)})'
            return f'{result_prefix} {lookups[self.cache[title]["lookup_index"][term]]}'
