from loguru import logger
from macrec.agents import Manager, Searcher, Interpreter
from huggingface_hub import InferenceClient  # Instalar o pacote `huggingface_hub`
import tiktoken
from loguru import logger
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from macrec.agents.base import Agent
from macrec.llms import AnyOpenAILLM

class SearchManager(Agent):
    """
    Manager agent que implementa o Contract Net Protocol para negociação de resultados entre searchers.
    """
    def __init__(self, searcher,config: dict, *args, **kwargs):
        """Inicializa o SearchManager com configuração de LLMs, searchers e pesos.

        Args:
            thought_config_path (`str`): Caminho para o config do LLM de pensamento.
            action_config_path (`str`): Caminho para o config do LLM de ação.
            searchers (`list`): Lista de agentes searchers.
            weights (`list`): Pesos associados a cada searcher.
            client: Cliente para interação com LLMs externos.
        """
        super().__init__(*args, **kwargs)
        self.config = config
        # self.agent_kwargs=agent_kwargs
        self.searchers=searcher 
        self.client = InferenceClient(base_url="https://api-inference.huggingface.co/v1/", api_key="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")


    def forward(self, query: str) -> str:
        """Lógica principal para invocar os searchers e processar resultados."""
        return self.invoke(query)

    def invoke(self, query: str) -> str:
        """Invoca todos os searchers, negocia as propostas e retorna o melhor resultado."""
        logger.debug(f"Invocando searchers para a query: {query}")

        proposals = [searcher.invoke(query,json_mode=True) for searcher in self.searchers]
        logger.debug(f"Propostas recebidas: {proposals}")

        evaluations = [self.evaluate_proposal(proposal) for proposal in proposals]
        logger.debug(f"Avaliações recebidas: {evaluations}")

        negotiated_result = self.negotiate_results(proposals, evaluations)
        return negotiated_result

    def evaluate_proposal(self, proposal: str) -> float:
        prompt = self._build_manager_prompt(
            examples="",
            query=f"Avalie esta proposta: {proposal}. Retorne um score de 0 a 10."
        )
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "system", "content": "Você é um avaliador imparcial."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        evaluation_message = response.choices[0].message['content']
        score = self.extract_score_from_evaluation(evaluation_message)
        logger.debug(f"Avaliação para '{proposal}': {score}")
        return score

    def extract_score_from_evaluation(self, evaluation_message: str) -> float:
        try:
            score = float(evaluation_message.split("Score:")[1].strip())
        except Exception:
            logger.warning(f"Falha ao extrair score de: {evaluation_message}")
            score = 0.0
        return score

    def negotiate_results(self, proposals: list, evaluations: list) -> str:
        weighted_proposals = [
            (proposal, eval_score * 1)
            for proposal, eval_score in zip(proposals, evaluations)
        ]
        best_proposal = max(weighted_proposals, key=lambda x: x[1])
        logger.info(f"Proposta selecionada: {best_proposal[0]} com score {best_proposal[1]:.2f}")
        return best_proposal[0]

    def _build_manager_prompt(self, **kwargs) -> str:
        """Cria um prompt formatado para a tarefa do manager."""
        return self.manager_prompt.format(**kwargs)

    @property
    def manager_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["query", "examples"],
            template="Avalie a seguinte proposta com base na query '{query}': {examples}"
        )

