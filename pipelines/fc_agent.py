"""
title: FC_AGENT
author: Soltech
version: 1.0
requirements: llama-index-embeddings-ollama==0.4.0
"""
from __future__ import annotations

import contextvars
import logging
import os
import sys
from typing import List, Union, Generator, Iterator

from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.objects import SimpleToolNodeMapping, ObjectIndex, ObjectRetriever
from llama_index.core.objects.base_node_mapping import BaseObjectNodeMapping
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fc_agent_module import fc_api_agent, fc_db_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
current_chat_memory_context: contextvars.ContextVar[ChatMemoryBuffer | None] = contextvars.ContextVar('current_chat_memory', default=None)

class Pipeline:
    name: str
    description: str
    valves: 'Valves'

    llm: Ollama = None
    db_tools: fc_db_agent.DBAgent = None
    api_tools: fc_api_agent.APIAgent = None
    obj_index: ObjectIndex = None
    embed_model: OllamaEmbedding = None

    """
        밸브 설정
    """
    class Valves(BaseModel):
        LLM_HOST: str = 'http://[LLMIP]:11434'
        LLM_MODEL: str = 'qwen3:30b-instruct'
        HISTORY_COUNT: int = 10

        EMBED_HOST: str = 'http://[EMBEDDINGIP]:11434'
        EMBED_MODEL_ID: str = 'bge-m3:latest'
        TOP_K_TOOLS: int = 5

        DB_HOST: str = '[DBIP]'
        DB_PORT: str = '1521'
        DB_USER: str = '[DBUSER]'
        DB_SCHEMA: str = '[DBSCHEMA]'
        DB_PASSWORD: str = '[DBPASSWORD]'
        DB_DATABASE: str = '[DBNAME]'
        DB_TABLES: str = '[DBTABLES]'

        DIGITS: int = 2
        ARRAY_MAX_LENGTH: int = 60

        API_URL: str = 'http://[APIIP]:8080/bizmanager/req-tag'

    """
        초기화
    """
    def __init__(self):
        self.name: str = 'Agent Pipeline'
        self.description: str = (
            'Agent Pipeline'
        )

        self.valves: Pipeline.Valves = self.Valves(
            **{
                'pipelines': ['*'],
            }
        )

    """
        서버 시작
    """
    async def on_startup(self) -> None:
        logger.debug(f'[Startup]: ---------- {self.name} initializing ----------')
        self.llm: Ollama = self._init_llm()
        self.embed_model: OllamaEmbedding = self._init_embed()
        self.db_tools: fc_db_agent.DBAgent = self._init_db_agent_tools()
        self.api_tools: fc_api_agent.APIAgent = self._init_api_agent_tools()
        self.obj_index: ObjectIndex = self._init_embed_tools()
        logger.debug(f'[Startup]: ---------- {self.name} Completed ----------')

    """
        서버 종료
    """
    async def on_shutdown(self) -> None:
        logger.debug(f'[Shutdown]: ---------- {self.name} ----------')
        if self.db_tools:
            self.db_tools.on_shutdown()
        pass

    """
        llm 초기화
    """
    def _init_llm(self) -> Ollama:
        return Ollama(model=self.valves.LLM_MODEL, base_url=self.valves.LLM_HOST,
                              request_timeout=300.0, temperature=0.1, keep_alive=1, streaming=True,
                              context_window=16384,
                              additional_kwargs={'stop': ['Observation:']})

    """
        embedding 초기화
    """
    def _init_embed(self) -> OllamaEmbedding:
        return OllamaEmbedding(
            model_name=self.valves.EMBED_MODEL_ID,
            base_url=self.valves.EMBED_HOST
        )

    """
        Agent Tool 임베딩
    """
    def _init_embed_tools(self) -> ObjectIndex:
        _agent_tools: list = self._get_all_agent_tools()
        _tool_mapping: BaseObjectNodeMapping = SimpleToolNodeMapping.from_objects(_agent_tools)

        return ObjectIndex.from_objects(
            objects=_agent_tools,
            object_mapping=_tool_mapping,
            index_cls=VectorStoreIndex,
            embed_model=self.embed_model
        )

    """
        DB Agent 초기화
    """
    def _init_db_agent_tools(self) -> fc_db_agent.DBAgent:
        return fc_db_agent.DBAgent(pipeline_valves=self.valves, llm=self.llm, embed_model=self.embed_model)

    """
        API Agent 초기화
    """
    def _init_api_agent_tools(self) -> fc_api_agent.APIAgent:
        return fc_api_agent.APIAgent(pipeline_valves=self.valves, llm=self.llm, embed_model=self.embed_model)

    """
        Agent Tool 목록
    """
    def _get_all_agent_tools(self) -> list[FunctionTool]:
        """
            DB Agent Tools
        """
        def __db_agent_tools(user_query: str) -> str:
            """
            [라우팅 도구] 사용자가 사내에 정해진 '목록 조회', '카테고리 확인', '재고 확인', '상태 변경' 등
            데이터베이스의 데이터를 조회요청 할 때 이 에이전트에게 질문(user_query)을 넘기세요.
            """
            logger.debug(f'[Call]: ---------- DB Agent Tools ----------')
            _memory: ChatMemoryBuffer = current_chat_memory_context.get()
            return self.db_tools.db_agent(user_message=user_query, memory=_memory)

        """
            API Agent Tools
        """
        def __api_agent_tools(user_query: str) -> str:
            """
            [라우팅 도구] 사용자가 사내에 정해진 '온도' 등
            태그 데이터를 조회요청 할 때 이 에이전트에게 질문(user_query)을 넘기세요.
            """
            logger.debug(f'[Call]: ---------- API Agent Tools ----------')
            _memory: ChatMemoryBuffer = current_chat_memory_context.get()
            return self.api_tools.api_agent(user_message=user_query, memory=_memory)

        return [
            FunctionTool.from_defaults(fn=__db_agent_tools),
            FunctionTool.from_defaults(fn=__api_agent_tools),
        ]

    """
        Chat History 파싱
    """
    def _parse_chat_history(
            self, user_message: str, messages: List[dict]
    ) -> list:
        _chat_history: list = []
        _history_count: int = -(abs(self.valves.HISTORY_COUNT) + 1)
        for m in messages[_history_count:-1]:
            if m['content'] == user_message:
                continue
            _role: MessageRole = MessageRole.USER if m['role'] == 'user' else MessageRole.ASSISTANT
            _chat_history.append(ChatMessage(role=_role, content=m['content']))

        return _chat_history

    """
        Pipeline 실행
    """
    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.debug(f'[Start]: ---------- {self.name} ----------')

        _chat_history: list = self._parse_chat_history(user_message=user_message, messages=messages)
        _memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(chat_history=_chat_history)
        current_chat_memory_context.set(_memory)

        _agent_tools_retriever: ObjectRetriever = self.obj_index.as_retriever(similarity_top_k=self.valves.TOP_K_TOOLS)

        _system_prompt: str = f'''
        당신은 엄격한 도구 사용 전문가다.
        1. 질문을 받으면, 절대로 스스로 추측하지 말고 반드시 필요한 도구를 호출해라.
        2. 사용자의 질문에 과제가 2개 이상 있다면, 모든 과제에 대해 각각 도구를 실행한 후에만 'Final Answer'를 작성할 수 있다.
        3. 필요한 파라메터가 부족한 경우에는 임의의 값을 스스로 지어내서 도구를 호출하지 말고, 사용자에게 "~~ 정보가 필요한데 알려주시겠어요?" 라고 질문만 해라.
        4. 알맞는 정보나 도구가 없다면 "지원하지 않는 기능입니다."라고만 답해라.
        5. [매우 중요] 반드시 정해진 포맷(Thought, Action, Action Input)만 사용하고, 절대로 응답 텍스트 전체를 마크다운 코드 블록(```)으로 감싸지 마라. 오직 평문으로만 출력해라.
        '''
        try:
            _agent: ReActAgent = ReActAgent(tools=[], tool_retriever=_agent_tools_retriever, llm=self.llm, memory=_memory, context=_system_prompt, verbose=True, max_iterations=8)
            _response: StreamingAgentChatResponse = _agent.stream_chat(message=user_message)

            logger.debug(f'[End]: ---------- {self.name} ----------')

            def __streaming_generator():
                #_full_text: str = ''
                for __token in _response.response_gen:
                    #_full_text += __token
                    yield __token
                #logger.debug(f'[Final Answer]: {_full_text}')

            return __streaming_generator()

        except Exception as e:
            logger.error(f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
            yield f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'
