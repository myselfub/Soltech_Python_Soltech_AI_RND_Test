from __future__ import annotations
from typing import List, Union, Generator, Iterator

from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import ChatMessage
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Pipeline:
    name: str
    description: str
    valves: 'Valves'

    """
        밸브 설정
    """
    class Valves(BaseModel):
        OLLAMA_HOST: str = 'http://[LLMIP]:11434'
        OLLAMA_MODEL: str = 'qwen3:30b-instruct'

    """
        초기화
    """
    def __init__(self):
        self.name: str = 'Test Pipeline'
        self.description: str = (
            'Test Pipeline'
        )

        self.valves: Pipeline.Valves = self.Valves(
            **{
                'pipelines': ['*'],
            }
        )

    """
        서버 시작
    """
    async def on_startup(self):
        pass

    """
        서버 종료
    """
    async def on_shutdown(self):
        pass

    """
        Pipeline 실행
    """
    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.info('start')
        logger.info(body)
        logger.info('----------------------------')
        _llm: Ollama = Ollama(model=self.valves.OLLAMA_MODEL, base_url=self.valves.OLLAMA_HOST,
                              request_timeout=300.0,
                              temperature=0.35,
                              keep_alive=1,
                              streaming=True,
                              context_window=16384)
        logger.info(messages)
        logger.info('----------------------------')
        logger.info(user_message)
        _messages = body.get("messages", [])
        messages: List[ChatMessage] = [
            ChatMessage(
                role=m['role'],
                content=m['content']
            ) for m in _messages
        ]

        logger.info(messages)

        response = _llm.chat(messages)
        yield response.message.content
