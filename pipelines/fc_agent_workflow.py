"""
title: FC_AGENT_WORKFLOW
author: Soltech
version: 1.0
requirements: llama-index-embeddings-ollama==0.4.0
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from asyncio import AbstractEventLoop
from typing import Any, AsyncGenerator, Generator, Iterator, List, Union
from pydantic import BaseModel
from workflows.handler import WorkflowHandler

from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent, AgentStream
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.objects import SimpleToolNodeMapping, ObjectIndex, ObjectRetriever
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fc_agent_module import fc_db_workflow, fc_api_workflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Colors:
    """
        색상표
    """
    RESET = '\033[0m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'
    BG_PURPLE = '\033[105m'
    BG_CYAN = '\033[106m'


class Pipeline:
    name: str
    description: str
    valves: 'Valves'

    llm: Ollama = None
    db_tools: fc_db_workflow.DBAgent = None
    api_tools: fc_api_workflow.APIAgent = None
    embed_model: OllamaEmbedding = None
    tools_retriever: ObjectRetriever = None

    class Valves(BaseModel):
        """
            밸브 설정
        """
        LLM_HOST: str = 'http://[LLMIP]:11434'
        LLM_MODEL_ID: str = 'qwen3:30b-instruct'
        HISTORY_COUNT: int = 10

        EMBED_HOST: str = 'http://[EMBEDDINGIP]:11434'
        EMBED_MODEL_ID: str = 'bge-m3:latest'
        TOP_K_TOOLS: int = 5

        DB_HOST: str = '[DBIP]'
        DB_PORT: str = '1521'
        DB_DATABASE: str = '[DBNAME]'
        DB_USER: str = '[DBUSER]'
        DB_PASSWORD: str = '[DBPASSWORD]'
        DB_SCHEMA: str = '[DBSCHEMA]'
        DB_TABLES: str = '[DBTABLES]'

        API_URL: str = 'http://[APIIP]:8080/bizmanager/req-tag'

        DIGITS: int = 2
        ARRAY_MAX_LENGTH: int = 60

    def __init__(self):
        """
            초기화
        """
        self.name: str = 'Agent Workflow Pipeline'
        self.description: str = (
            'Agent Workflow Pipeline'
        )

        self.valves: Pipeline.Valves = self.Valves(
            **{
                'pipelines': ['*'],
            }
        )

    async def on_startup(self) -> None:
        """
            서버 시작
        """
        logger.debug(f'[Startup]: ---------- {self.name} initializing ----------')
        self.llm: Ollama = self._init_llm()
        self.embed_model: OllamaEmbedding = self._init_embed()
        self.db_tools: fc_db_workflow.DBAgent = self._init_db_agent_tools()
        self.api_tools: fc_api_workflow.APIAgent = self._init_api_agent_tools()
        _obj_index: ObjectIndex = self._init_embed_tools()
        self.tools_retriever: ObjectRetriever = _obj_index.as_retriever(similarity_top_k=self.valves.TOP_K_TOOLS)
        logger.debug(f'[Startup]: ---------- {self.name} Completed ----------')

    async def on_shutdown(self) -> None:
        """
            서버 종료
        """
        logger.debug(f'[Shutdown]: ---------- {self.name} ----------')
        if self.db_tools:
            self.db_tools.on_shutdown()
        pass

    def _init_llm(self) -> Ollama:
        """
            LLM 초기화
        """
        return Ollama(model=self.valves.LLM_MODEL_ID, base_url=self.valves.LLM_HOST,
                      request_timeout=300.0, temperature=0.1, keep_alive=1, streaming=True,
                      context_window=16384,
                      additional_kwargs={'stop': ['Observation:']})

    def _init_embed(self) -> OllamaEmbedding:
        """
            Embedding 초기화
        """
        return OllamaEmbedding(
            model_name=self.valves.EMBED_MODEL_ID,
            base_url=self.valves.EMBED_HOST
        )

    def _init_embed_tools(self) -> ObjectIndex:
        """
            Agent Tool 임베딩
        """
        _agent_tools: list = self._get_all_agent_tools()
        _tool_mapping = SimpleToolNodeMapping.from_objects(_agent_tools)

        return ObjectIndex.from_objects(
            objects=_agent_tools,
            object_mapping=_tool_mapping,
            index_cls=VectorStoreIndex,
            embed_model=self.embed_model
        )

    def _init_db_agent_tools(self) -> fc_db_workflow.DBAgent:
        """
            DB Agent 초기화
        """
        return fc_db_workflow.DBAgent(pipeline_valves=self.valves, llm=self.llm, embed_model=self.embed_model)

    def _init_api_agent_tools(self) -> fc_api_workflow.APIAgent:
        """
            API Agent 초기화
        """
        return fc_api_workflow.APIAgent(pipeline_valves=self.valves, llm=self.llm, embed_model=self.embed_model)

    def _get_all_agent_tools(self) -> list[FunctionTool]:
        """
            Agent Tool 목록
        """

        async def __db_agent_tools(user_query: str) -> str:
            """
            [라우팅 도구] DB Agent

            사용자가:
            - 목록 조회
            - 카테고리 조회
            - 재고 조회
            - 상태 변경

            등 데이터베이스의 데이터를 요청하면
            반드시 이 도구를 사용해라.
            """
            logger.debug(f'[Call]: ---------- DB Agent Tools ----------')
            return await self.db_tools.db_agent(user_message=user_query)

        async def __api_agent_tools(user_query: str) -> str:
            """
            [라우팅 도구] API Agent

            사용자가:
            - 온도
            - 센서값
            - 태그 데이터

            등 태그 데이터를 요청하면
            반드시 이 도구를 사용해라.
            """
            logger.debug(f'[Call]: ---------- API Agent Tools ----------')
            return await self.api_tools.api_agent(user_message=user_query)

        return [
            FunctionTool.from_defaults(async_fn=__db_agent_tools),
            FunctionTool.from_defaults(async_fn=__api_agent_tools),
        ]

    def _parse_user_message(self, messages: list[dict]) -> str:
        """
        User Message 파싱
        """
        _user_message: str = ""
        for __msg in reversed(messages):
            if __msg.get("role") == "user":
                __content: str | list = __msg.get("content", "")
                if isinstance(__content, str):
                    _user_message: str = __content
                elif isinstance(__content, list):
                    __texts: list = []
                    for __item in __content:
                        if __item.get("type") == "text":
                            __texts.append(__item.get("text", ""))
                    _user_message: str = "\n".join(__texts)
                break

        return _user_message

    def _parse_chat_history(
            self, user_message: str, messages: list[dict]
    ) -> list:
        """
            Chat History 파싱
        """
        _chat_history: list = []
        _history_count: int = -(abs(self.valves.HISTORY_COUNT) + 1)
        for m in messages[_history_count:-1]:
            if m['content'] == user_message:
                continue
            __role: MessageRole = MessageRole.USER if m['role'] == 'user' else MessageRole.ASSISTANT
            _chat_history.append(ChatMessage(role=__role, content=m['content']))

        return _chat_history

    async def _run_workflow_agent(
            self, user_message: str, tools_retriever: ObjectRetriever, system_prompt: str
    ) -> AsyncGenerator[str, Any]:
        """
            Workflow Agent 실행
        """
        _agent: ReActAgent = ReActAgent(llm=self.llm, tools=[], tool_retriever=tools_retriever,
                                        system_prompt=system_prompt, verbose=False)
        _handler: WorkflowHandler = _agent.run(user_msg=user_message, max_iterations=8)

        # 1. 스트리밍 출력을 제어하기 위한 스위치와 버퍼
        _is_streaming_final: bool = False
        _stream_buffer: str = ''

        async for __event in _handler.stream_events():
            __event_name: str = type(__event).__name__
            if isinstance(__event, AgentStream):
                if __event.delta:
                    # 2. 이미 최종 답변 구간으로 진입했다면 무조건 외부로 화면 출력(yield)
                    if _is_streaming_final:
                        yield __event.delta
                    else:
                        # 3. 아직 생각(Thought) 중이라면 버퍼에 글자 누적
                        _stream_buffer += __event.delta
                        # 4. 버퍼 안에 최종 답변 시작 키워드가 등장했는지 검사
                        __keyword: str = ''
                        if 'Final Answer:' in _stream_buffer:
                            __keyword = 'Final Answer:'
                        elif 'Answer:' in _stream_buffer:
                            __keyword: str = 'Answer:'
                        # 5. 키워드가 발견되면 스위치를 켜고, 키워드 이후의 텍스트부터 yield 시작
                        if __keyword:
                            _is_streaming_final: bool = True
                            __split_text: str = _stream_buffer.split(__keyword, 1)[1]
                            if __split_text:
                                yield __split_text
                continue

            # --- 이 아래는 각 단계(Step)가 완료되었을 때 실행되는 로깅 영역 ---
            if hasattr(__event, 'response'):
                __content: str = ''
                __event_response: ChatMessage = __event.response
                if hasattr(__event_response, 'response') and isinstance(__event_response.response, str):
                    __content: str = __event_response.response
                elif hasattr(__event_response, 'message') and hasattr(__event_response.message, 'content'):
                    __content: str = __event_response.message.content
                elif hasattr(__event_response, 'content'):
                    __content: str = __event_response.content
                else:
                    __content: str = str(__event_response)
                if __content:
                    __thought: str = str(__content).strip()
                    if __thought:
                        # 6. 중간 과정의 Thought와 텍스트는 yield 하지 않고 Logger로만 출력
                        logger.info(f'\n{Colors.MAGENTA}[Agent Thought/Log]:\n{__thought}{Colors.RESET}\n')
                # 7. 최종 답변 중이 아니라면 다음 도구 추론 단계를 위해 버퍼를 비워줌
                if not _is_streaming_final:
                    _stream_buffer: str = ''
            elif __event_name == 'ToolCallResult':
                if hasattr(__event, 'tool_call'):
                    __tool_name: str = getattr(__event.tool_call, 'tool_name', 'Unknown')
                    __tool_kwargs: dict = getattr(__event.tool_call, 'tool_kwargs', {})
                    logger.info(f'{Colors.BLUE}Action: {__tool_name}{Colors.RESET}')
                    logger.info(f'{Colors.BLUE}Action Input: {__tool_kwargs}{Colors.RESET}')
                if hasattr(__event, 'tool_output'):
                    logger.info(f'{Colors.BLUE}Observation: {str(__event.tool_output)[:500]}{Colors.RESET}')

        await _handler

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
            Pipeline 실행
        """
        logger.debug(f'[Start]: ---------- {self.name} ----------')

        _chat_history: list = self._parse_chat_history(user_message=user_message, messages=messages)
        _system_prompt: str = f'''
        당신은 엄격한 도구 사용 전문가다.
        1. 질문을 받으면, 절대로 스스로 추측하지 말고 반드시 필요한 도구를 호출해라.
        2. 사용자의 질문에 과제가 2개 이상 있다면, 모든 과제에 대해 각각 도구를 실행한 후에만 'Final Answer'를 작성할 수 있다.
        3. 필요한 파라메터가 부족한 경우에는 임의의 값을 스스로 만들어 도구를 호출하지 말고, 사용자에게 필요한 정보를 요청해라.
        4. 알맞는 정보나 도구가 없다면 "지원하지 않는 기능입니다."라고만 답해라.
        5. 반드시 plain text만 사용해라. 마크다운 코드블럭을 사용하지 마라.
        '''
        try:
            try:
                _loop: AbstractEventLoop = asyncio.get_event_loop()
                # _loop: AbstractEventLoop = asyncio.get_running_loop()
            except RuntimeError:
                _loop: AbstractEventLoop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)

            _response: AsyncGenerator[str, Any] = self._run_workflow_agent(user_message=user_message,
                                                                           tools_retriever=self.tools_retriever,
                                                                           system_prompt=_system_prompt)
            while True:
                try:
                    __chunk: str | None = _loop.run_until_complete(anext(_response, None))
                    if __chunk is None:
                        break
                    yield __chunk
                except StopAsyncIteration:
                    break
            logger.debug(f'[End]: ---------- {self.name} ----------')
        except Exception as e:
            logger.error(f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
            yield f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'
