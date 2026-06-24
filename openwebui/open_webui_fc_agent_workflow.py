"""
title: PipeTest
author: Soltech
author_url:
version: 1.0.0
icon_url:
required_open_webui_version: 0.9.0
requirements: llama-index-core==0.12.52.post1, llama-index-embeddings-ollama==0.4.0, llama-index-llms-ollama==0.4.2, oracledb==3.4.2, matplotlib==3.11.0
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import logging
import os
import oracledb
import re
import requests
import socket
import sqlalchemy
import sys
from datetime import datetime, timedelta
from io import BytesIO
from pydantic import Field
from typing import Any, AsyncGenerator, Optional, Callable

from llama_index.core import SQLDatabase, PromptTemplate, VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent, AgentOutput, AgentStream
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.objects import SimpleToolNodeMapping, ObjectIndex, ObjectRetriever
from llama_index.core.objects.base_node_mapping import BaseObjectNodeMapping
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from oracledb import Var
from pydantic import BaseModel
from sqlalchemy import (
    create_engine,
    text,
    bindparam,
    CursorResult,
    TextClause,
    PoolProxiedConnection,
)
from sqlalchemy.engine.interfaces import DBAPICursor
from workflows.handler import WorkflowHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _init_oracle_client() -> None:
    """
    Oracle 11g 접속을 위해 python-oracledb thick mode를 초기화합니다.
    """
    __candidate_dirs: list[str] = []
    __oracle_client: Optional[str] = os.getenv("ORACLE_CLIENT")
    if __oracle_client:
        __candidate_dirs.append(__oracle_client)

    __ld_library_path: Optional[str] = os.getenv("LD_LIBRARY_PATH")
    if __ld_library_path:
        __candidate_dirs.extend(
            __path for __path in __ld_library_path.split(":") if __path
        )

    __candidate_dirs.append("/opt/oracle/instantclient_19_24")

    for __lib_dir in dict.fromkeys(__candidate_dirs):
        if not os.path.exists(os.path.join(__lib_dir, "libclntsh.so")):
            continue
        try:
            oracledb.init_oracle_client(lib_dir=__lib_dir)
            logger.info(f"[Oracle Client] Thick mode 초기화 완료: {__lib_dir}")
            return
        except oracledb.ProgrammingError:
            logger.info("[Oracle Client] Thick mode가 이미 초기화되어 있습니다.")
            return
        except Exception as e:
            logger.warning(
                f"[Oracle Client] Thick mode 초기화 실패({__lib_dir}): {str(e)}"
            )


event_emitter_var: contextvars.ContextVar[Callable[[dict], Any] | None] = (
    contextvars.ContextVar("event_emitter_var", default=None)
)
user_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "user_var", default=None
)
metadata_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "metadata_var", default=None
)
body_context_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "body_context_var", default=None
)


def _contains_simm_chart_markup(content: str) -> bool:
    """
    SIMM 차트 미리보기 마크업 포함 여부를 확인합니다.
    """
    return (
        "data:image/svg+xml;base64," in content
        or "data:image/png;base64," in content
    )


class Colors:
    """
    색상표
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    BG_RED = "\033[101m"
    BG_GREEN = "\033[102m"
    BG_YELLOW = "\033[103m"
    BG_BLUE = "\033[104m"
    BG_PURPLE = "\033[105m"
    BG_CYAN = "\033[106m"


class Pipe:
    name: str
    description: str
    valves: "Valves"

    llm: Ollama = None
    db_tools: DBAgent = None
    api_tools: APIAgent = None
    simm_tools: SIMMAgent = None
    embed_model: OllamaEmbedding = None
    tools_retriever: ObjectRetriever = None

    class Valves(BaseModel):
        """
        밸브 설정
        """

        LLM_HOST: str = Field(
            default="http://[LLMIP]:11434", description="LLM Host"
        )
        HISTORY_COUNT: int = Field(default=10, description="참고할 채팅 기록 수")
        LLM_MODEL_ID: str = Field(
            default="qwen3:30b-instruct", description="LLM Model ID"
        )

        EMBED_HOST: str = Field(
            default="http://[EMBEDDINGIP]:11434", description="Embedding Host"
        )
        EMBED_MODEL_ID: str = Field(
            default="bge-m3:latest", description="Embedding Model ID"
        )
        TOP_K_TOOLS: int = Field(default=5, description="도구 Embedding TOP-K 개수")

        DB_HOST: str = Field(default="[DBIP]", description="DB Host")
        DB_PORT: str = Field(default="1521", description="DB Port")
        DB_DATABASE: str = Field(default="[DBNAME]", description="DB Database")
        DB_USER: str = Field(default="[DBUSER]", description="DB User")
        DB_PASSWORD: str = Field(default="[DBPASSWORD]", description="DB Password")
        DB_SCHEMA: str = Field(default="[DBSCHEMA]", description="DB Schema")
        DB_TABLES: str = Field(
            default="TEST_ITEM,CATEGORIES,PRODUCTS,INVENTORY",
            description="DB Table 목록(, 기준)",
        )

        API_URL: str = Field(
            default="http://[APIIP]:8080/bizmanager",
            description="API URL",
        )

        SIMM_DB_HOST: str = Field(default="[SIMM_DBIP]", description="DB Host")
        SIMM_DB_PORT: str = Field(default="8521", description="DB Port")
        SIMM_DB_DATABASE: str = Field(default="[SIMM_DBNAME]", description="DB Database")
        SIMM_DB_USER: str = Field(default="[SIMM_DBUSER]", description="DB User")
        SIMM_DB_PASSWORD: str = Field(default="[SIMM_DBPASSWORD]", description="DB Password")
        SIMM_DB_SCHEMA: str = Field(default="[SIMM_DBSCHEMA]", description="DB Schema")
        SIMM_DB_TABLES: str = Field(
            default="TB_SIM_ST_IF,TB_SIM_ST_INPUT_1",
            description="DB Table 목록(, 기준)",
        )
        SIMM_API_URL: str = Field(
            default="http://[SIMM_APIIP]:8084", description="SIMM API URL"
        )

        DIGITS: int = Field(default=2, description="표시할 소수점 자리수")
        ARRAY_MAX_LENGTH: int = Field(default=60, description="Array 최대 길이")
        pass

    def __init__(self):
        """
        초기화
        """
        self.name: str = "Agent Workflow Pipeline"
        self.description: str = "Agent Workflow Pipeline"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
            }
        )
        self.on_startup()
        pass

    def on_startup(self) -> None:
        """
        서버 시작
        """
        logger.debug(f"[Startup]: ---------- {self.name} initializing ----------")
        self.llm: Ollama = self._init_llm()
        self.embed_model: OllamaEmbedding = self._init_embed()
        self.db_tools: DBAgent = self._init_db_agent_tools()
        self.api_tools: APIAgent = self._init_api_agent_tools()
        self.simm_tools: SIMMAgent = self._init_simm_agent_tools()
        _obj_index: ObjectIndex = self._init_embed_tools()
        self.tools_retriever: ObjectRetriever = _obj_index.as_retriever(
            similarity_top_k=self.valves.TOP_K_TOOLS
        )
        logger.debug(f"[Startup]: ---------- {self.name} Completed ----------")

    def on_shutdown(self) -> None:
        """
        서버 종료
        """
        logger.debug(f"[Shutdown]: ---------- {self.name} ----------")
        if self.db_tools:
            self.db_tools.on_shutdown()

    def _init_llm(self) -> Ollama:
        """
        LLM 초기화
        """
        return Ollama(
            model=self.valves.LLM_MODEL_ID,
            base_url=self.valves.LLM_HOST,
            request_timeout=300.0,
            temperature=0.1,
            keep_alive=1,
            streaming=True,
            context_window=16384,
            additional_kwargs={"stop": ["Observation:"]},
        )

    def _init_embed(self) -> OllamaEmbedding:
        """
        Embedding 초기화
        """
        return OllamaEmbedding(
            model_name=self.valves.EMBED_MODEL_ID, base_url=self.valves.EMBED_HOST
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
            embed_model=self.embed_model,
        )

    def _init_db_agent_tools(self) -> DBAgent:
        """
        DB Agent 초기화
        """
        return DBAgent(
            pipeline_valves=self.valves, llm=self.llm, embed_model=self.embed_model
        )

    def _init_api_agent_tools(self) -> APIAgent:
        """
        API Agent 초기화
        """
        return APIAgent(
            pipeline_valves=self.valves, llm=self.llm, embed_model=self.embed_model
        )

    def _init_simm_agent_tools(self) -> SIMMAgent:
        """
        SIMM Agent 초기화
        """
        return SIMMAgent(
            pipeline_valves=self.valves, llm=self.llm, embed_model=self.embed_model
        )

    def _get_all_agent_tools(self) -> list[FunctionTool]:
        """
        Agent Tool 목록
        """

        async def db_agent_tools(
            user_query: str = Field(..., description="사용자 질의")
        ) -> str:
            """
            [라우팅 도구] DB Agent

            사용자가:
            - 목록 조회
            - 카테고리 조회
            - 재고 조회
            - 상태 변경

            등 데이터베이스의 데이터를 요청하면 반드시 이 도구를 사용해라.
            """
            logger.debug(f"[Call]: ---------- DB Agent Tools ----------")
            __event_emitter__: Optional[Callable[[dict], Any]] = event_emitter_var.get()
            __user__: Optional[dict] = user_var.get()
            __metadata__: Optional[dict] = metadata_var.get()
            body_context: Optional[dict] = body_context_var.get()
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "데이터베이스 조회중...", "done": False},
                    }
                )

            _response: str = await self.db_tools.db_agent(
                user_message=user_query,
                __user__=__user__,
                __metadata__=__metadata__,
                body_context=body_context,
                __event_emitter__=__event_emitter__,
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "데이터베이스 조회완료...", "done": True},
                    }
                )

            return _response

        async def api_agent_tools(
            user_query: str = Field(..., description="사용자 질의")
        ) -> str:
            """
            [라우팅 도구] API Agent

            사용자가:
            - 온도
            - 센서값
            - 태그 데이터

            등 태그 데이터를 요청하면 반드시 이 도구를 사용해라.
            """
            logger.debug(f"[Call]: ---------- API Agent Tools ----------")
            __event_emitter__: Optional[Callable[[dict], Any]] = event_emitter_var.get()
            __user__: Optional[dict] = user_var.get()
            __metadata__: Optional[dict] = metadata_var.get()
            body_context: Optional[dict] = body_context_var.get()
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "API 조회중...", "done": False},
                    }
                )

            _response: str = await self.api_tools.api_agent(
                user_message=user_query,
                __user__=__user__,
                __metadata__=__metadata__,
                body_context=body_context,
                __event_emitter__=__event_emitter__,
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "API 조회완료...", "done": True},
                    }
                )

            return _response

        async def simm_agent_tools(
            user_query: str = Field(..., description="사용자 질의")
        ) -> str:
            """
            [라우팅 도구] SIMM Agent

            사용자가 다음과 같은 요청을 하면 이 도구를 사용합니다.
            - 시뮬레이션 실행
            - 모의운영 실행
            - 강우량 증가/감소/설정 후 모의운영 실행
            - 특정 날짜, 시간, 댐 구간의 모의운영 실행
            """
            logger.debug(f"[Call]: ---------- SIMM Agent Tools ----------")
            __event_emitter__: Optional[Callable[[dict], Any]] = event_emitter_var.get()
            __user__: Optional[dict] = user_var.get()
            __metadata__: Optional[dict] = metadata_var.get()
            body_context: Optional[dict] = body_context_var.get()
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "SIMM 조회중...", "done": False},
                    }
                )

            _response: str = await self.simm_tools.simm_agent(
                user_message=user_query,
                __user__=__user__,
                __metadata__=__metadata__,
                body_context=body_context,
                __event_emitter__=__event_emitter__,
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "SIMM 조회완료...", "done": True},
                    }
                )

            if not _response.startswith("[시스템 알림]") and (
                "더 필요합니다" in _response or _response.endswith("겠습니까?")
            ):
                return (
                    "[시스템 알림] SIMM 모의운영 처리가 사용자 확인 또는 추가 입력을 필요로 합니다. "
                    f"더 이상 도구를 찾지 말고 즉시 사용자에게 '{_response}'라고 답변하세요."
                )
            return _response

        return [
            FunctionTool.from_defaults(async_fn=db_agent_tools),
            FunctionTool.from_defaults(async_fn=api_agent_tools),
            FunctionTool.from_defaults(async_fn=simm_agent_tools),
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

    def _parse_chat_history(self, user_message: str, messages: list[dict]) -> list:
        """
        Chat History 파싱
        """
        _chat_history: list = []
        _history_count: int = -(abs(self.valves.HISTORY_COUNT) + 1)
        for m in messages[_history_count:-1]:
            if m["content"] == user_message:
                continue
            __role: MessageRole = (
                MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
            )
            _chat_history.append(ChatMessage(role=__role, content=m["content"]))

        return _chat_history

    async def _run_workflow_agent(
        self,
        user_message: str,
        tools_retriever: ObjectRetriever,
        system_prompt: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        body_context: Optional[dict] = None,
    ) -> AsyncGenerator[str, Any]:
        """
        Workflow Agent 실행
        """
        _agent: ReActAgent = ReActAgent(
            llm=self.llm,
            tools=[],
            tool_retriever=tools_retriever,
            system_prompt=system_prompt,
            verbose=False,
        )
        _handler: WorkflowHandler = _agent.run(user_msg=user_message, max_iterations=8)

        _is_streaming_final: bool = False
        _stream_buffer: str = ""

        async for __event in _handler.stream_events():
            __event_name: str = type(__event).__name__
            if isinstance(__event, AgentStream):
                if __event.delta:
                    if _is_streaming_final:
                        yield __event.delta
                    else:
                        _stream_buffer += __event.delta
                        __keyword: str = ""
                        if "Final Answer:" in _stream_buffer:
                            __keyword = "Final Answer:"
                        elif "Answer:" in _stream_buffer:
                            __keyword: str = "Answer:"
                        if __keyword:
                            _is_streaming_final: bool = True
                            __split_text: str = _stream_buffer.split(__keyword, 1)[1]
                            if __split_text:
                                yield __split_text
                continue

            if hasattr(__event, "response"):
                __content: str = ""
                __event_response: ChatMessage = __event.response
                if hasattr(__event_response, "response") and isinstance(
                    __event_response.response, str
                ):
                    __content: str = __event_response.response
                elif hasattr(__event_response, "message") and hasattr(
                    __event_response.message, "content"
                ):
                    __content: str = __event_response.message.content
                elif hasattr(__event_response, "content"):
                    __content: str = __event_response.content
                else:
                    __content: str = str(__event_response)
                if __content:
                    __thought: str = str(__content).strip()
                    if __thought:
                        logger.info(
                            f"\n{Colors.MAGENTA}[Agent Thought/Log]:\n{__thought}{Colors.RESET}\n"
                        )
                if not _is_streaming_final:
                    _stream_buffer: str = ""
            elif __event_name == "ToolCallResult":
                if hasattr(__event, "tool_call"):
                    __tool_name: str = getattr(
                        __event.tool_call, "tool_name", "Unknown"
                    )
                    __tool_kwargs: dict = getattr(__event.tool_call, "tool_kwargs", {})
                    logger.info(f"{Colors.BLUE}Action: {__tool_name}{Colors.RESET}")
                    logger.info(
                        f"{Colors.BLUE}Action Input: {__tool_kwargs}{Colors.RESET}"
                    )
                if hasattr(__event, "tool_output"):
                    __tool_output_text: str = str(__event.tool_output)
                    logger.info(
                        f"{Colors.BLUE}Observation: {__tool_output_text[:500]}{Colors.RESET}"
                    )
                    if _contains_simm_chart_markup(__tool_output_text):
                        yield __tool_output_text
                        return

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "응답생성중...", "done": False},
                }
            )

        await _handler

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> AsyncGenerator[str, Any]:
        """
        Pipeline 실행
        """
        logger.debug(f"[Start]: ---------- {self.name} ----------")
        _messages: list[dict] = body.get("messages", [])
        _user_message: str = self._parse_user_message(_messages)
        _chat_history: list = self._parse_chat_history(
            user_message=_user_message, messages=_messages
        )

        event_emitter_var.set(__event_emitter__)
        user_var.set(__user__ or {})
        metadata_var.set(__metadata__ or {})
        body_context_var.set(body or {})

        _system_prompt: str = f"""
        당신은 엄격한 도구 사용 전문가다.
        1. 질문을 받으면, 절대로 스스로 추측하지 말고 반드시 필요한 도구를 호출해라.
        2. 사용자의 질문에 과제가 2개 이상 있다면, 모든 과제에 대해 각각 도구를 실행한 후에만 'Final Answer'를 작성할 수 있다.
        3. 필요한 파라메터가 부족한 경우에는 임의의 값을 스스로 만들어 도구를 호출하지 말고, 사용자에게 필요한 정보를 요청해라.
        4. 알맞는 정보나 도구가 없다면 "지원하지 않는 기능입니다."라고만 답해라.
        5. 기본적으로 plain text만 사용하고 마크다운 코드블럭을 사용하지 마라. 단, 도구 결과에 SVG/PNG Markdown 이미지가 포함되어 있으면 삭제하거나 요약하지 말고 그대로 최종 답변에 포함해라.
        6. Observation이 "[시스템 알림]"으로 시작하면 다른 도구를 절대 호출하지 말고, 알림에 적힌 사용자 응답 문구만 Final Answer로 답해라.
        """

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"처리중...", "done": False},
                    }
                )

            _response: AsyncGenerator[str, Any] = self._run_workflow_agent(
                user_message=_user_message,
                tools_retriever=self.tools_retriever,
                system_prompt=_system_prompt,
                __event_emitter__=__event_emitter__,
                __user__=__user__ or {},
                __metadata__=__metadata__ or {},
                body_context=body or {},
            )

            async for __chunk in _response:
                yield __chunk

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "", "done": True},
                    }
                )
        except Exception as e:
            _error_message: str = str(e)
            _is_max_iteration_error: bool = (
                "Max iterations" in _error_message
                or "parse_agent_output" in _error_message
            )

            if _is_max_iteration_error:
                logger.info(
                    f"[Agent Max Iteration] 오케스트레이션 에이전트 최대 반복 도달: {_error_message}"
                )
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "", "done": True},
                        }
                    )
                yield "요청 처리 중 응답을 완성하지 못했습니다. 필요한 정보를 조금 더 구체적으로 다시 입력해 주세요."
                return

            logger.error(
                f"[Agent Error] 오케스트레이션 에이전트 처리 중 오류가 발생했습니다: {_error_message}"
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "", "done": True},
                    }
                )
            yield "요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


class DBAgent:
    valves: "Valves"
    engine: sqlalchemy.engine.base.Engine = None
    llm: Ollama = None

    class Valves(BaseModel):
        """
        밸브 설정
        """

        DB_HOST: str = None
        DB_PORT: str = None
        DB_DATABASE: str = None
        DB_USER: str = None
        DB_PASSWORD: str = None
        DB_SCHEMA: str = None
        DB_TABLES: str = None

        TOP_K_TOOLS: int = 0

    def __init__(
        self, pipeline_valves: BaseModel, llm: Ollama, embed_model: OllamaEmbedding
    ) -> None:
        """
        초기화
        """
        self.name: str = "Database Agent"

        _valves_dict: dict = pipeline_valves.model_dump()
        self.valves: DBAgent.Valves = self.Valves(**_valves_dict)
        self.engine: sqlalchemy.engine.base.Engine = self._init_db_connection()
        logger.debug(f"[DataBase Connected]: {self.valves.DB_HOST}")

        self.llm: Ollama = llm
        self.embed_model: OllamaEmbedding = embed_model
        self.obj_index: ObjectIndex = self._init_embed_tools()

    def on_shutdown(self):
        """
        서버 종료
        """
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()
        pass

    def _init_embed_tools(self) -> ObjectIndex:
        """
        Agent Tool 임베딩
        """
        _db_tools: list = [
            FunctionTool.from_defaults(fn=self.get_table_comments_in_database),
            FunctionTool.from_defaults(fn=self.select_query),
            FunctionTool.from_defaults(fn=self.get_data_procedure),
        ]
        _tool_mapping: BaseObjectNodeMapping = SimpleToolNodeMapping.from_objects(
            _db_tools
        )

        return ObjectIndex.from_objects(
            objects=_db_tools,
            object_mapping=_tool_mapping,
            index_cls=VectorStoreIndex,
            embed_model=self.embed_model,
        )

    def _get_upper_tables_list(self) -> list:
        """
        Upper 테이블명 목록
        """
        return [t.strip().upper() for t in self.valves.DB_TABLES.split(",")]

    def _get_lower_tables_list(self) -> list:
        """
        Lower 테이블명 목록
        """
        return [t.strip().lower() for t in self.valves.DB_TABLES.split(",")]

    def _init_db_connection(self) -> sqlalchemy.engine.base.Engine:
        """
        Database 연결
        Oracle + oracledb 드라이버 사용 설정 (형식: oracle+oracledb://user:pass@host:port/?service_name=db)
        """
        _connection_url: str = (
            f"oracle+oracledb://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}@"
            f"{self.valves.DB_HOST}:{self.valves.DB_PORT}/"
            f"?service_name={self.valves.DB_DATABASE}"
        )
        _engine: sqlalchemy.engine.base.Engine = create_engine(
            _connection_url
        )  # , echo=True)

        return _engine

    def get_table_comments_in_table(self) -> str:
        """
        사용자가 데이터베이스 테이블 구조, 컬럼 정보, 데이터 타입 또는 테이블의 의미(코멘트)에 대해 질문할 때 호출합니다.

        현재 설정된 스키마와 대상 테이블 목록을 기준으로, DB 딕셔너리를 조회하여 테이블명, 컬럼명,
        데이터 타입 및 코멘트(설명)를 마크다운(Markdown) 형식의 텍스트로 반환합니다.
        반환된 결과를 바탕으로 사용자에게 데이터베이스 구조를 설명하거나 SQL 쿼리를 작성할 수 있습니다.
        Returns:
            str: 테이블 및 컬럼의 메타데이터가 정리된 마크다운 문자열
        """
        ### --------------------------------------------------
        ### Table에 정의된 테이블 설명 조회
        ### --------------------------------------------------
        logger.debug(f"----- get_table_comments_in_table -----")
        _schema_info: str = ""
        with self.engine.connect() as _conn:
            _query: TextClause = text("""
                          SELECT TBLS.table_name  AS table_name,
                                 TBLS.descrt      AS table_descrt,
                                 COLS.column_name AS column_name,
                                 COLS.data_type   AS data_type,
                                 COLS.descrt      AS column_descrt
                          FROM (SELECT TABLE_NAME,
                                       DESCRT
                                FROM [DBSCHEMA].db_info
                                WHERE div = 'TABLE') TBLS
                                   LEFT JOIN (SELECT table_name,
                                                     column_name,
                                                     data_type,
                                                     descrt
                                              FROM [DBSCHEMA].db_info
                                              WHERE div = 'COLUMN') COLS
                                             ON
                                                 TBLS.table_name = COLS.table_name
                          WHERE TBLS.table_name IN :tables
                          ORDER BY table_name, column_name
                          """).bindparams(bindparam("tables", expanding=True))

            _tables_upper_list: list = self._get_upper_tables_list()

            _result: CursorResult = _conn.execute(
                _query, {"tables": _tables_upper_list}
            )

            _current_table: str = ""
            for (
                __table_name,
                __table_descr,
                __column_name,
                __data_type,
                __col_descrt,
            ) in _result.fetchall():
                if __table_name != _current_table:
                    _current_table: str = __table_name
                    __table_desc: str = f" ({__table_descr})" if __table_descr else ""
                    _schema_info += f"\n### TABLE: {__table_name}{__table_desc}\n"

                __col_desc: str = f" - {__col_descrt}" if __col_descrt else ""
                _schema_info += f"  * {__column_name} [{__data_type}]{__col_desc}\n"

        return _schema_info

    def get_table_comments_in_database(self) -> str:
        """
        데이터베이스의 스키마, 테이블 구조, 컬럼 정보, 데이터 타입 및 코멘트(설명)를 조회하는 도구입니다.

        다음 상황에서 이 도구를 호출하세요:
        - 사용자가 데이터베이스 구조나 테이블/컬럼의 의미에 대해 질문할 때
        - 정확한 SQL 쿼리를 작성하기 위해 테이블의 메타데이터(컬럼명, 데이터 타입 등)가 필요할 때

        Returns:
            str: 대상 테이블과 컬럼의 메타데이터가 정리된 마크다운 형식의 문자
        """
        ### --------------------------------------------------
        ### DB에 정의된 테이블 설명 조회
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- get_table_comments_in_database ----------")
        schema: str = self.valves.DB_SCHEMA.strip().upper()

        _schema_info: str = ""
        with self.engine.connect() as _conn:
            _query = text("""
                              SELECT T1.table_name,
                                     T3.comments AS table_comment,
                                     T1.column_name,
                                     T1.data_type,
                                     T2.comments AS column_comment
                              FROM all_tab_columns T1
                                       JOIN all_col_comments T2
                                            ON
                                                T1.table_name = T2.table_name
                                                    AND T1.column_name = T2.column_name
                                       JOIN all_tab_comments T3
                                            ON
                                                T1.table_name = T3.table_name
                              WHERE T1.owner = :schema
                                AND T1.table_name IN :tables
                              ORDER BY T1.table_name,
                                       T1.column_id
                              """).bindparams(bindparam("tables", expanding=True))

            _tables_upper_list: list = self._get_upper_tables_list()

            _result: CursorResult = _conn.execute(
                _query, {"schema": schema, "tables": _tables_upper_list}
            )

            _current_table: str = ""
            for _row in _result:
                if _row.table_name != _current_table:
                    _current_table = _row.table_name
                    _table_desc: str = (
                        f" ({_row.table_comment})" if _row.table_comment else ""
                    )
                    _schema_info += f"\n### TABLE: {_row.table_name}{_table_desc}\n"

                _column_desc = (
                    f" - {_row.column_comment}" if _row.column_comment else ""
                )
                _schema_info += (
                    f"  * {_row.column_name} [{_row.data_type}]{_column_desc}\n"
                )

        return _schema_info

    def select_query(
        self,
        message: str = Field(
            ...,
            description="사용자의 자연어 데이터 조회 요청 (예: '카테고리가 전자인 상품들을 보여줘')",
        ),
    ) -> str:
        """
        사용자가 데이터베이스의 실제 데이터(예: 특정 카테고리의 상품 목록, 재고 수량 등)나 집계를 조회해 달라고 요청할 때 호출합니다.
        자연어 질문(message)을 입력받아, 내부적으로 Oracle SQL 쿼리를 생성하고 실행한 뒤 그 결과값을 문자열로 반환합니다.

        Args:
            message (str): 사용자의 자연어 데이터 조회 요청 (예: '카테고리가 전자인 상품들을 보여줘')
        """
        ### --------------------------------------------------
        ### Query 실행
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- select_query ----------")
        _tables_lower_list: list = self._get_lower_tables_list()
        _sql_database: SQLDatabase = SQLDatabase(
            self.engine,
            schema=self.valves.DB_USER.lower().strip(),
            include_tables=_tables_lower_list,
        )
        _schema_info: str = self.get_table_comments_in_table()
        logger.debug(f"[Schema Info]: {_schema_info}")

        _sql_prompt: str = f"""
        당신은 {{dialect}} SQL 전문가입니다. 
        제공된 테이블 스키마를 참고하여 사용자의 질문에 최적화된 SQL 쿼리를 생성하고 결과에 기반해 답변하십시오.

        [스키마 정보]
        {_schema_info}

        [작성 규칙]
        1. **문법**: 반드시 Oracle SQL(Oracle 12c 이상) 문법을 사용하십시오. (LIMIT 대신 'FETCH FIRST n ROWS ONLY' 사용)
        2. **별칭**: SQL 쿼리를 작성할 때, 모든 AS(Alias) 키워드 뒤의 별칭은 반드시 쌍따옴표("")로 감싸야 합니다. 예: SELECT column_name AS "alias_name".
        3. **제한**: SELECT 쿼리만 허용합니다. (DELETE, UPDATE, DROP 등 금지). 사용자가 명시하지 않는 한 최대 100건만 조회하십시오.
        4. **효율성**: SELECT * 사용 금지. 필요한 컬럼만 명시하십시오. 필요한 경우 DISTINCT를 사용하십시오.
        5. **금지**: SQL 쿼리 생성 시 서술형 설명이나 주석을 붙이지 말고 오직 실행 가능한 SQL만 출력하십시오.

        [출력 형식]
        모든 대답은 한글로 답변하며, 반드시 아래 형식을 유지하며 각 항목은 한 줄씩 작성하십시오:

        Question: 사용자의 질문 내용
        SQLQuery: 실행할 Oracle SQL 쿼리
        SQLResult: SQL 실행 결과
        Answer: 결과에 기반한 최종 답변

        질문: {{query_str}}
        SQLQuery: 
        """
        logger.debug(f"[Prompt]: {_sql_prompt}")
        _sql_template: PromptTemplate = PromptTemplate(_sql_prompt)

        _query_engine: NLSQLTableQueryEngine = NLSQLTableQueryEngine(
            sql_database=_sql_database,
            tables=_tables_lower_list,
            llm=self.llm,
            embed_model="local",
            text_to_sql_prompt=_sql_template,
            streaming=True,
        )

        _response: StreamingResponse = _query_engine.query(message)
        logger.debug(_response.metadata)
        """
        _full_text = ''
        for _token in _response.response_gen:
            _full_text += _token
        """

        return str(_response)

    def get_data_procedure(
        self,
        category_code: str = Field(
            description="조회할 데이터의 카테고리 코드 (예: 'ELEC', 'WEAR')"
        ),
    ) -> str:
        """
        사용자가 특정 카테고리의 전체 목록이나 상세 데이터를 나열해 달라고 할 때만 사용합니다.
        단, 카테고리가 명확히 지정되지 않은 질문에는 이 도구를 쓰지 마세요

        [경고: 절대 사용 금지 조건]
        질문에 '합계', '총합', '평균', '개수', '통계' 같은 계산/집계 요구사항이 포함되어 있다면
        이 도구를 절대 사용하지 말고, 반드시 _call_query 도구를 사용하십시오.

        Args:
            category_code (str): 조회할 데이터의 카테고리 코드 (예: 'ELEC', 'WEAR')

        Returns:
            str: 조회된 데이터의 목록 (Markdown 표 또는 텍스트 형식)
        """
        ### --------------------------------------------------
        ### get_data 프로시저 실행
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- get_data_procedure ----------")
        if category_code == "ALL":
            return f"[시스템 알림] 구체적인 카테고리 코드가 없습니다. 더 이상 도구를 찾지 말고 즉시 사용자에게 '어떤 카테고리를 조회할까요?'라고 답변하세요."
        _result_text: str = ""

        _conn: PoolProxiedConnection | None = None
        _cursor: DBAPICursor | None = None
        _ref_cursor: oracledb.Cursor | None = None

        try:
            _conn: PoolProxiedConnection = self.engine.raw_connection()
            _cursor: DBAPICursor = _conn.cursor()

            _out_cursor: Var = _cursor.var(oracledb.CURSOR)

            _procedure_name: str = f"{self.valves.DB_SCHEMA.upper()}.GET_DATA_CURSOR"
            _cursor.callproc(_procedure_name, [category_code, _out_cursor])

            _ref_cursor: oracledb.Cursor = _out_cursor.getvalue()

            if _ref_cursor:
                _columns: list = [col[0] for col in _ref_cursor.description]
                _result_text += " | ".join(_columns) + "\n"
                _result_text += "-" * 50 + "\n"

                _rows = _ref_cursor.fetchmany(100)
                if not _rows:
                    return "[시스템 알림] 조회된 데이터가 0건입니다. 절대로 임의의 데이터를 지어내지 마세요. 사용자에게 '요청하신 조건에 맞는 데이터가 없습니다. 카테고리를 다시 확인해 주세요.'라고 답변하세요."

                for row in _rows:
                    _result_text += " | ".join(str(item) for item in row) + "\n"
            else:
                _result_text: str = "조회된 데이터가 없습니다."

            logger.debug(f"----- {_result_text} -----")

            return _result_text
        except Exception as e:
            logger.error(f"[Procedure Error]: {str(e)}")
            return "[시스템 알림] 프로시저 실행 중 내부 오류가 발생했습니다. 더 이상 도구를 찾지 말고 즉시 사용자에게 '요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.'라고 답변하세요."
        finally:
            try:
                if _ref_cursor:
                    _ref_cursor.close()
            except:
                pass
            try:
                if _cursor:
                    _cursor.close()
            except:
                pass
            try:
                if _conn:
                    _conn.close()
            except:
                pass

    async def db_agent(
        self,
        user_message: str,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        body_context: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        DB Tools Agent
        """
        logger.debug(f"[Start]: ---------- {self.name} ----------")
        _db_tools_retriever: ObjectRetriever = self.obj_index.as_retriever(
            similarity_top_k=self.valves.TOP_K_TOOLS
        )

        _system_prompt: str = f"""
        당신은 엄격한 도구 사용 전문가다.
        1. 질문을 받으면, 절대로 스스로 추측하지 말고 반드시 필요한 도구를 호출해라.
        2. 사용자의 질문에 과제가 2개 이상 있다면, 모든 과제에 대해 각각 도구를 실행한 후에만 'Final Answer'를 작성할 수 있다.
        3. 필요한 파라메터가 부족한 경우에는 임의의 값을 스스로 지어내서 도구를 호출하지 말고, 사용자에게 "~~ 정보가 필요한데 알려주시겠어요?" 라고 질문만 해라.
        4. 알맞는 정보나 도구가 없다면 "지원하지 않는 기능입니다."라고만 답해라.
        5. [매우 중요] 반드시 정해진 포맷(Thought, Action, Action Input)만 사용하고, 절대로 응답 텍스트 전체를 마크다운 코드 블록(```)으로 감싸지 마라. 오직 평문으로만 출력해라.
        """
        try:
            _agent: ReActAgent = ReActAgent(
                llm=self.llm,
                tools=[],
                tool_retriever=_db_tools_retriever,
                system_prompt=_system_prompt,
                verbose=False,
            )
            _handler: WorkflowHandler = _agent.run(
                user_msg=user_message, max_iterations=8
            )

            _is_streaming_final: bool = False
            _stream_buffer: str = ""
            _final_answer_buffer: str = ""

            async for __event in _handler.stream_events():
                __event_name: str = type(__event).__name__
                if isinstance(__event, AgentStream):
                    if __event.delta:
                        if _is_streaming_final:
                            _final_answer_buffer += __event.delta
                        else:
                            _stream_buffer += __event.delta
                            __keyword: str = ""
                            if "Final Answer:" in _stream_buffer:
                                __keyword = "Final Answer:"
                            elif "Answer:" in _stream_buffer:
                                __keyword: str = "Answer:"
                            if __keyword:
                                _is_streaming_final: bool = True
                                __split_text: str = _stream_buffer.split(__keyword, 1)[
                                    1
                                ]
                                if __split_text:
                                    _final_answer_buffer += __split_text
                    continue
                if hasattr(__event, "response"):
                    __content: str = ""
                    __event_response: ChatMessage = __event.response
                    if hasattr(__event_response, "response") and isinstance(
                        __event_response.response, str
                    ):
                        __content: str = __event_response.response
                    elif hasattr(__event_response, "message") and hasattr(
                        __event_response.message, "content"
                    ):
                        __content: str = __event_response.message.content
                    elif hasattr(__event_response, "content"):
                        __content: str = __event_response.content
                    else:
                        __content: str = str(__event_response)
                    if __content:
                        __thought: str = str(__content).strip()
                        if __thought:
                            logger.info(
                                f"\n{Colors.YELLOW}[Agent Thought/Log]:\n{__thought}{Colors.RESET}\n"
                            )
                    if not _is_streaming_final:
                        _stream_buffer: str = ""
                elif __event_name == "ToolCallResult":
                    if hasattr(__event, "tool_call"):
                        __tool_name: str = getattr(
                            __event.tool_call, "tool_name", "Unknown"
                        )
                        __tool_kwargs: dict = getattr(
                            __event.tool_call, "tool_kwargs", {}
                        )
                        logger.info(f"{Colors.CYAN}Action: {__tool_name}{Colors.RESET}")
                        logger.info(
                            f"{Colors.CYAN}Action Input: {__tool_kwargs}{Colors.RESET}"
                        )
                    if hasattr(__event, "tool_output"):
                        logger.info(
                            f"{Colors.CYAN}Observation: {str(__event.tool_output)[:500]}{Colors.RESET}"
                        )

            _result: AgentOutput = await _handler
            if _final_answer_buffer:
                logger.info(
                    f"\n{Colors.GREEN}[Sub Agent Final Answer]:\n{_final_answer_buffer.strip()}{Colors.RESET}\n"
                )

            logger.debug(f"[End]: ---------- {self.name} ----------")
            if hasattr(_result, "response"):
                return str(_result.response)
            return str(_result)
        except Exception as e:
            logger.error(
                f"[Agent Error] 데이터베이스 에이전트 처리 중 오류가 발생했습니다: {str(e)}"
            )
            return "요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


class APIAgent:
    valves: "Valves"
    llm: Ollama = None

    class Valves(BaseModel):
        """
        밸브 설정
        """

        API_URL: str = None

        DIGITS: int = 0
        ARRAY_MAX_LENGTH: int = 0

        TOP_K_TOOLS: int = None

    def __init__(
        self, pipeline_valves: BaseModel, llm: Ollama, embed_model: OllamaEmbedding
    ) -> None:
        """
        초기화
        """
        self.name: str = "API Agent"

        _valves_dict: dict = pipeline_valves.model_dump()
        self.valves: APIAgent.Valves = self.Valves(**_valves_dict)

        self.llm: Ollama = llm
        self.embed_model: OllamaEmbedding = embed_model
        self.obj_index: ObjectIndex = self._init_embed_tools()

    def on_shutdown(self):
        """
        서버 종료
        """
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()
        pass

    def _init_embed_tools(self) -> ObjectIndex:
        """
        Agent Tool 임베딩
        """
        _api_tools: list = [
            # FunctionTool.from_defaults(fn=self._call_api),
            FunctionTool.from_defaults(fn=self.get_current_time),
            # FunctionTool.from_defaults(fn=self.get_tag_list),
            FunctionTool.from_defaults(fn=self.find_tag_list),
            FunctionTool.from_defaults(fn=self.get_factory_data),
            FunctionTool.from_defaults(fn=self.slice_list),
        ]
        _tool_mapping: BaseObjectNodeMapping = SimpleToolNodeMapping.from_objects(
            _api_tools
        )

        return ObjectIndex.from_objects(
            objects=_api_tools,
            object_mapping=_tool_mapping,
            index_cls=VectorStoreIndex,
            embed_model=self.embed_model,
        )

    def _call_api_req_tag(
        self, params: dict = Field(..., description="API 호출에 필요한 body 데이터")
    ) -> dict:
        """
        BizNexus API 통신을 위한 도구
        Args:
            params (str): API 호출에 필요한 body 데이터
        """
        ### --------------------------------------------------
        ### API 통신
        ### --------------------------------------------------
        logger.debug(f"[Start]: ---------- _call_api ----------")
        __connection_url: str = self.valves.API_URL + "/req-tag"
        __headers: dict = {"Content-Type": "application/json"}
        __response: requests.Response = requests.post(
            __connection_url, json=params, headers=__headers
        )
        if not (200 <= __response.status_code < 300):
            logger.error(
                f"[API Error] API 호출 중 오류가 발생했습니다: {__response.status_code}"
            )
            raise Exception(
                f"API 호출 중 오류가 발생했습니다: {__response.text}({__response.status_code})"
            )

        __result = __response.json()
        logger.debug(f"[API Result]: {__result}")
        return __result

    def get_current_time(self) -> str:
        """
        현재 시스템의 날짜와 시간을 'YYYY-MM-DD HH:mm:ss' 형식으로 반환합니다.
        LLM의 내부 지식으로 시간을 말하지 마세요. 반드시 이 도구를 호출해서 얻은 결과만 답변에 사용하세요.
        Returns:
            string: '2026-01-01 00:00:00'
        """
        ### --------------------------------------------------
        ### 현재 시간
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- get_current_time ----------")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_tag_list(self) -> str:
        """
        공장에서 관리하는 모든 태그(온도, 전압 등)의 이름과 설명 목록을 반환합니다.
        """
        ### --------------------------------------------------
        ### 태그 목록 전체 조회
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- get_tag_list ----------")
        __params: dict = {
            "cmd": "selectTags",
            "param": {"category": "byName", "val": ""},
        }
        try:
            __response: dict = self._call_api_req_tag(__params)
            __tag_list = [
                {"name": __item.get("name", ""), "desc": __item.get("desc", "")}
                for __item in __response.get("param", [])
                if not __item.get("name", "$").startswith("$")
            ]
            return json.dumps(__tag_list, ensure_ascii=False)
        except Exception as e:
            logger.error(
                f"[API Error] 태그 데이터 API 조회 중 오류가 발생했습니다: {str(e)}"
            )
            return "[시스템 알림] 태그 데이터 API 조회 중 내부 오류가 발생했습니다. 더 이상 도구를 찾지 말고 즉시 사용자에게 '요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.'라고 답변하세요."

    def find_tag_list(
        self,
        keyword: str = Field(
            default="", description="검색할 키워드 (예: '1층', '온도', 'ROOM1')"
        ),
    ) -> list:
        """
        공장에서 관리하는 모든 태그(온도, 전압 등) 목록에서 특정 키워드가 포함된 태그만 검색하여 이름과 설명 목록을 반환합니다.
        (예: '1층 온도' => 키워드에 '1층' 혹은 '온도' 중 하나)
        Args:
            keyword: 검색할 키워드 (예: '1층', '온도', 'ROOM1')
        """
        ### --------------------------------------------------
        ### 태그 목록 키워드 조회
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- find_tag_list ----------")
        _all_tags: list = json.loads(self.get_tag_list())
        _filtered = [
            _item
            for _item in _all_tags
            if keyword in _item["name"] or keyword in _item["desc"]
        ]

        return _filtered[: self.valves.ARRAY_MAX_LENGTH]

    def get_factory_data(
        self,
        cmd: str = Field(
            default="reqValues",
            description="'reqValues'(실시간) 또는 'fetchValues'(과거)",
        ),
        tag_list: list[str] = Field(
            default=None, description="조회할 태그명 리스트 (예: ['ROOM1_TEMP'])"
        ),
        start: str = Field(
            default="",
            description="과거 조회 시 시작시간 (YYYY-MM-DD HH:mm:ss). 실시간 조회 시 빈 문자열.",
        ),
        end: str = Field(
            default="",
            description="과거 조회 시 종료시간 (YYYY-MM-DD HH:mm:ss). 실시간 조회 시 빈 문자열.",
        ),
    ) -> str:
        """
        공장의 실시간 수치나 과거 기록을 조회합니다.
        Args:
            cmd: 'reqValues'(실시간) 또는 'fetchValues'(과거)
            tag_list: 조회할 태그명 리스트 (예: ['ROOM1_TEMP'])
            start: 과거 조회 시 시작시간 (YYYY-MM-DD HH:mm:ss). 실시간 조회 시 빈 문자열.
            end: 과거 조회 시 종료시간 (YYYY-MM-DD HH:mm:ss). 실시간 조회 시 빈 문자열.
        Returns:
            JSON string: {
                "ROOM1_TEMP": {"count": 1111, "sum": 21264.10, "min": 17.5, "max": 21.2, "avg": 19.14, "raw_data": [{"time": "2026-05-06 00:00:00", "val": 18.1}, ...]},
                ...
            }
            - 사용자가 '평균', '합계', '최고/최저' 등 통계를 물으면 'summary'를 사용하세요.
            - 사용자가 '데이터 보여줘', '기록 알려줘' 등 상세 내역을 물으면 'raw_data'를 표(Table) 형태로 정리해 답변하세요.
        """
        ### --------------------------------------------------
        ### API 조회
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- get_factory_data ----------")
        __param: dict = {"cmd": cmd, "param": {"tagList": tag_list}}
        if start and end:
            __param["param"].update({"start": start, "end": end, "span": 10})

        try:
            logger.debug(f"param: {__param}")
            __response: dict = self._call_api_req_tag(__param)
            __process: dict = self._process_data(__response)
            logger.debug(f"process: {__process}")

            return json.dumps(__process, ensure_ascii=False)
        except Exception as e:
            logger.error(
                f"[API Error] API 데이터 호출 중 오류가 발생했습니다: {str(e)}"
            )
            return "[시스템 알림] API 데이터 호출 중 내부 오류가 발생했습니다. 더 이상 도구를 찾지 말고 즉시 사용자에게 '요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.'라고 답변하세요."

    def _process_data(self, response: dict) -> dict:
        """
        통계 계산
        """
        __results: dict = {}
        for __item in response.get("param", []):
            __name: str = __item.get("name", "")
            __vals: list = []
            if __item.get("val", ""):
                # fetchValues
                __vals: list = [float(__item.get("val"))]
            elif __item.get("values", []):
                __vals: list = [
                    self.safe_convert_float(__v.get("val", ""))
                    for __v in __item.get("values", [])
                    if self.safe_convert_float(__v.get("val", "")) is not None
                ]

            if __vals:
                __digits: int = self.valves.DIGITS
                __sum: float = sum(__vals)
                __count: int = len(__vals)
                __results[__name] = {
                    "count": __count,
                    "sum": __sum,
                    "min": round(min(__vals, default=0), __digits),
                    "max": round(max(__vals, default=0), __digits),
                    "avg": round(__sum / __count, __digits) if __count > 0 else 0,
                }
                if __item.get("val", ""):
                    __results[__name].update({"raw_data": __vals})
                elif __item.get("values", []):
                    __results[__name].update(
                        {
                            "raw_data": self.slice_list(
                                __item.get("values", []), self.valves.ARRAY_MAX_LENGTH
                            )
                        }
                    )

        return __results

    def safe_convert_float(self, num: str) -> float | None:
        """
        float 형변환
        """
        try:
            return float(num)
        except Exception as e:
            logger.error(f"Float 변환 중 오류가 발생했습니다: {str(e)}")
            return None

    def slice_list(self, target_list: list | str = None, num: int = 20) -> list:
        """
        List를 num의 개수만큼 잘라 반환
        Args:
            target_list: 자를 리스트
            num: 자를 리스트 개수
        """
        ### --------------------------------------------------
        ### list 자르기
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- slice_list ----------")
        if target_list is None:
            return []
        elif type(target_list) is str:
            target_list: list = json.loads(target_list)

        return target_list[-num:]

    async def api_agent(
        self,
        user_message: str,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        body_context: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        API Tools Agent
        """
        logger.debug(f"[Start]: ---------- {self.name} ----------")
        _api_tools_retriever: ObjectRetriever = self.obj_index.as_retriever(
            similarity_top_k=self.valves.TOP_K_TOOLS
        )

        __max_length: int = self.valves.ARRAY_MAX_LENGTH
        _system_prompt: str = f"""
        당신은 공장 데이터 관리 전문가입니다. 절대로 사용자의 의도를 추측하지 마세요.
        사용자의 질문에 답하기 위해 반드시 다음 행동 강령과 단계를 엄격하게 지키세요.

        [핵심 행동 강령 및 처리 단계]
        1. 시간 기준 설정
            - 현재 시간을 모른다면 'get_current_time'을 먼저 호출하여 기준을 잡으세요.
        2. 태그 조회를 위한 키워드 추출 및 확인
            - 사용자가 질문하면 가장 먼저 'find_tag_list' (또는 필요시 'get_tag_list')를 호출하여 공장에 어떤 장소(구역/설비)와 측정 항목이 있는지 파악하세요.
            - 'find_tag_list' 호출 시 질문에서 핵심 명사(예: '1층', '온도') 중 한 단어만 추출하여 keyword 파라미터에 넣어 검색 효율을 높이세요.
            - [주의] 사용자가 "태그 목록을 보여달라"고 명시적으로 요청하지 않는 한, 조회된 태그 리스트를 답변에 나열하지 말고 내부 참조용으로만 사용하세요.
        3. 장소(구역/설비) 명시 여부 검증 및 재질문 (가장 중요)
            - 사용자의 질문에 특정 장소(예: '1층', 'A구역', '1번 펌프' 등)가 명확히 언급되지 않았다면(예: 그냥 "온도 알려줘", "어제 데이터 보여줘"), 절대로 'get_factory_data'를 호출하지 마세요.
            - 장소를 모를 때는 데이터를 임의로 조회하지 말고, 반드시 "어느 장소의 데이터를 알려드릴까요?"라고 질문만 한 뒤 사용자의 답변을 기다리세요.
            - 검색된 태그 리스트의 첫 번째 항목을 임의로 기본값으로 지정하여 조회해서는 안 됩니다.
            - 사용자가 명시한 장소가 태그 목록에 존재하지 않는다면 "해당 구역(설비)은 등록되어 있지 않습니다."라고 명확히 안내하세요.
        4. 실제 데이터 조회 ('get_factory_data' 호출)
            - 사용자가 명확하게 장소(구역/설비)를 언급한 것이 확인된 시점에만 'get_factory_data'를 사용하여 실제 수치를 조회하세요.
            - 오늘/어제 등 상대적인 날짜는 현재 시간을 기준으로 계산해서 YYYY-MM-DD HH:mm:ss 형식으로 변환하여 요청하세요.
            - [데이터 활용 규칙]
                -- 통계적인 질문(평균, 최댓값 등)에는 'summary' 데이터를 바탕으로 문장으로 답하세요.
                -- 상세 내역 조회 요청(데이터 보여줘, 기록 알려줘 등)에는 'raw_data'의 내용을 사용하여 답변하세요.
                -- 만약 'raw_data'가 너무 많다면({__max_length}개 초과), "최근 데이터 {__max_length}개만 표시합니다"라는 안내와 함께 상위 {__max_length}개만 표(Markdown Table) 형태로 깔끔하게 보여주세요.
        5. 최종 답변 규칙
            - 어떤 상황에서도 최종 답변은 한국어로 작성하세요. (분석 결과나 시스템 메시지가 영어라도 한국어로 번역할 것)
            - 공장 데이터 관리 업무와 무관한 질문은 단호히 거절하세요.
        """
        try:
            _agent: ReActAgent = ReActAgent(
                llm=self.llm,
                tools=[],
                tool_retriever=_api_tools_retriever,
                system_prompt=_system_prompt,
                verbose=False,
            )
            _handler: WorkflowHandler = _agent.run(
                user_msg=user_message, max_iterations=8
            )

            _is_streaming_final: bool = False
            _stream_buffer: str = ""
            _final_answer_buffer: str = ""

            async for __event in _handler.stream_events():
                __event_name: str = type(__event).__name__
                if isinstance(__event, AgentStream):
                    if __event.delta:
                        if _is_streaming_final:
                            _final_answer_buffer += __event.delta
                        else:
                            _stream_buffer += __event.delta
                            __keyword: str = ""
                            if "Final Answer:" in _stream_buffer:
                                __keyword = "Final Answer:"
                            elif "Answer:" in _stream_buffer:
                                __keyword: str = "Answer:"
                            if __keyword:
                                _is_streaming_final: bool = True
                                __split_text: str = _stream_buffer.split(__keyword, 1)[
                                    1
                                ]
                                if __split_text:
                                    _final_answer_buffer += __split_text
                    continue
                if hasattr(__event, "response"):
                    __content: str = ""
                    __event_response: ChatMessage = __event.response
                    if hasattr(__event_response, "response") and isinstance(
                        __event_response.response, str
                    ):
                        __content: str = __event_response.response
                    elif hasattr(__event_response, "message") and hasattr(
                        __event_response.message, "content"
                    ):
                        __content: str = __event_response.message.content
                    elif hasattr(__event_response, "content"):
                        __content: str = __event_response.content
                    else:
                        __content: str = str(__event_response)
                    if __content:
                        __thought: str = str(__content).strip()
                        if __thought:
                            logger.info(
                                f"\n{Colors.YELLOW}[Agent Thought/Log]:\n{__thought}{Colors.RESET}\n"
                            )
                    if not _is_streaming_final:
                        _stream_buffer: str = ""
                elif __event_name == "ToolCallResult":
                    if hasattr(__event, "tool_call"):
                        __tool_name: str = getattr(
                            __event.tool_call, "tool_name", "Unknown"
                        )
                        __tool_kwargs: dict = getattr(
                            __event.tool_call, "tool_kwargs", {}
                        )
                        logger.info(f"{Colors.CYAN}Action: {__tool_name}{Colors.RESET}")
                        logger.info(
                            f"{Colors.CYAN}Action Input: {__tool_kwargs}{Colors.RESET}"
                        )
                    if hasattr(__event, "tool_output"):
                        logger.info(
                            f"{Colors.CYAN}Observation: {str(__event.tool_output)[:500]}{Colors.RESET}"
                        )

            _result: AgentOutput = await _handler
            if _final_answer_buffer:
                logger.info(
                    f"\n{Colors.GREEN}[Sub Agent Final Answer]:\n{_final_answer_buffer.strip()}{Colors.RESET}\n"
                )

            logger.debug(f"[End]: ---------- {self.name} ----------")
            if hasattr(_result, "response"):
                return str(_result.response)
            return str(_result)
        except Exception as e:
            logger.error(
                f"[API Error] API 에이전트 처리 중 오류가 발생했습니다: {str(e)}"
            )
            return "요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


class SIMMAgent:
    valves: "Valves"
    engine: Optional[sqlalchemy.engine.base.Engine] = None
    llm: Optional[Ollama] = None
    SIMM_MODEL_TYPE: str = "OSS"
    SIMM_USER_ID: str = "[SIMM_USER_ID]"
    SIMM_USER_IP: str = "[SIMM_USER_IP]"
    SIMM_EXECUTION_TIMEOUT_SECONDS: int = 180
    SIMM_DAM_ORDER: tuple[dict[str, Any], ...] = (
        {"name": "평화의댐", "dam_cd": "1009710", "dam_scd": "PH", "order": 0},
        {"name": "화천", "dam_cd": "1010310", "dam_scd": "HC", "order": 1},
        {"name": "춘천", "dam_cd": "1010320", "dam_scd": "CC", "order": 2},
        {"name": "소양강댐", "dam_cd": "1012110", "dam_scd": "SY", "order": 3},
        {"name": "의암", "dam_cd": "1013310", "dam_scd": "UA", "order": 4},
        {"name": "청평", "dam_cd": "1015310", "dam_scd": "CP", "order": 5},
        {"name": "도암", "dam_cd": "1001310", "dam_scd": "DA", "order": 6},
        {"name": "충주", "dam_cd": "1003110", "dam_scd": "CJ", "order": 7},
        {"name": "괴산", "dam_cd": "1004310", "dam_scd": "GS", "order": 8},
        {"name": "충주(조)", "dam_cd": "1003611", "dam_scd": "JJ", "order": 9},
        {"name": "횡성", "dam_cd": "1006110", "dam_scd": "HS", "order": 10},
        {"name": "팔당", "dam_cd": "1017310", "dam_scd": "PD", "order": 11},
        {"name": "섬진강", "dam_cd": "4001110", "dam_scd": "SJ", "order": 12},
        {"name": "보성강", "dam_cd": "4007310", "dam_scd": "BS", "order": 13},
    )
    SIMM_DOWNSTREAM_PATH_MAP: dict[str, tuple[str, ...]] = {
        "1009710": ("1009710", "1010310", "1010320", "1013310", "1015310", "1017310"),
        "1010310": ("1010310", "1010320", "1013310", "1015310", "1017310"),
        "1010320": ("1010320", "1013310", "1015310", "1017310"),
        "1013310": ("1013310", "1015310", "1017310"),
        "1015310": ("1015310", "1017310"),
        "1017310": ("1017310",),
        "1012110": ("1012110", "1013310", "1015310", "1017310"),
        "1001310": ("1001310", "1003110", "1003611", "1017310"),
        "1003110": ("1003110", "1003611", "1017310"),
        "1003611": ("1003611", "1017310"),
        "1004310": ("1004310", "1017310"),
        "1006110": ("1006110", "1017310"),
        "4001110": ("4001110", "4007310"),
        "4007310": ("4007310",),
    }
    SIMM_CONFLUENCE_MAP: dict[str, tuple[str, ...]] = {
        "1010310": ("1009710",),
        "1010320": ("1010310",),
        "1013310": ("1010320", "1012110"),
        "1015310": ("1013310",),
        "1017310": ("1015310", "1006110", "1003611"),
        "1003110": ("1001310",),
        "1003611": ("1003110", "1004310"),
    }
    SIMM_DAM_LMT_WL_MAP: dict[str, float] = {
        "1009710": 264.6,
        "1010310": 175.0,
        "1010320": 102.0,
        "1012110": 190.3,
        "1013310": 70.5,
        "1015310": 50.0,
        "1001310": 707.0,
        "1003110": 138.0,
        "1004310": 134.0,
        "1003611": 65.1,
        "1006110": 178.2,
        "1017310": 25.5,
        "4001110": 194.0,
        "4007310": 126.9,
    }
    SIMM_HC_DAM_CD: str = "1010310"
    SIMM_HC_OTF_TAGS: tuple[str, ...] = (
        "D1010310FCOTF501",
        "D1010310FCOTF502",
        "D1010310FCOTF503",
        "D1010310FCOTF504",
        "D1010310FCOTF505",
    )
    SIMM_DAM_ALIASES: dict[str, str] = {
        "평화": "평화의댐",
        "평화댐": "평화의댐",
        "평화의댐": "평화의댐",
        "화천댐": "화천",
        "화천": "화천",
        "춘천댐": "춘천",
        "춘천": "춘천",
        "소양강댐": "소양강댐",
        "소양댐": "소양강댐",
        "소양강": "소양강댐",
        "의암댐": "의암",
        "의암": "의암",
        "청평댐": "청평",
        "청평": "청평",
        "팔당댐": "팔당",
        "팔당": "팔당",
        "괴산댐": "괴산",
        "괴산": "괴산",
        "충주댐": "충주",
        "충주": "충주",
        "충주(조)댐": "충주(조)",
        "충주조댐": "충주(조)",
        "충주조정지댐": "충주(조)",
        "충주(조)": "충주(조)",
        "횡성댐": "횡성",
        "횡성": "횡성",
        "도암": "도암",
        "도암댐": "도암",
        "섬진강댐": "섬진강",
        "섬진강": "섬진강",
        "보성강댐": "보성강",
        "보성강": "보성강",
    }
    SIMM_DAM_TAG_MAP: dict[str, tuple[str, ...]] = {
        "1009710": ("D1009710FCPCP112", "D1009710FCPCP113", "D1009710FCPCP114"),
        "1010310": (
            "D1010310FCPCP115",
            "D1010310FCPCP116",
            "D1010310FCPCP117",
            "D1010310FCPCP118",
            "D1010310FCPCP119",
            "D1010310FCPCP120",
            "D1010310FCPCP121",
        ),
        "1010320": (
            "D1010320FCPCP122",
            "D1010320FCPCP123",
            "D1010320FCPCP124",
            "D1010320FCPCP125",
            "D1010320FCPCP126",
        ),
        "1012110": (
            "D1012110FCPCP128",
            "D1012110FCPCP129",
            "D1012110FCPCP130",
            "D1012110FCPCP131",
            "D1012110FCPCP132",
            "D1012110FCPCP133",
            "D1012110FCPCP134",
            "D1012110FCPCP135",
            "D1012110FCPCP136",
            "D1012110FCPCP137",
            "D1012110FCPCP138",
        ),
        "1013310": (
            "D1013310FCPCP127",
            "D1013310FCPCP139",
            "D1013310FCPCP140",
            "D1013310FCPCP141",
        ),
        "1015310": (
            "D1015310FCPCP142",
            "D1015310FCPCP143",
            "D1015310FCPCP144",
            "D1015310FCPCP145",
            "D1015310FCPCP146",
            "D1015310FCPCP147",
            "D1015310FCPCP148",
            "D1015310FCPCP149",
            "D1015310FCPCP150",
            "D1015310FCPCP151",
            "D1015310FCPCP152",
            "D1015310FCPCP153",
            "D1015310FCPCP154",
            "D1015310FCPCP155",
            "D1015310FCPCP156",
            "D1015310FCPCP157",
            "D1015310FCPCP158",
            "D1015310FCPCP159",
            "D1015310FCPCP160",
            "D1015310FCPCP161",
            "D1015310FCPCP162",
        ),
        "1017310": (
            "D1017310FCPCP066",
            "D1017310FCPCP067",
            "D1017310FCPCP068",
            "D1017310FCPCP069",
            "D1017310FCPCP071",
            "D1017310FCPCP072",
            "D1017310FCPCP073",
            "D1017310FCPCP074",
            "D1017310FCPCP075",
            "D1017310FCPCP076",
            "D1017310FCPCP077",
            "D1017310FCPCP078",
            "D1017310FCPCP079",
            "D1017310FCPCP080",
            "D1017310FCPCP081",
            "D1017310FCPCP082",
            "D1017310FCPCP083",
            "D1017310FCPCP084",
            "D1017310FCPCP085",
            "D1017310FCPCP086",
            "D1017310FCPCP087",
            "D1017310FCPCP088",
            "D1017310FCPCP089",
            "D1017310FCPCP090",
            "D1017310FCPCP091",
            "D1017310FCPCP092",
            "D1017310FCPCP093",
            "D1017310FCPCP094",
            "D1017310FCPCP095",
            "D1017310FCPCP096",
            "D1017310FCPCP097",
            "D1017310FCPCP098",
            "D1017310FCPCP163",
            "D1017310FCPCP164",
            "D1017310FCPCP165",
            "D1017310FCPCP166",
            "D1017310FCPCP167",
            "D1017310FCPCP168",
            "D1017310FCPCP169",
            "D1017310FCPCP170",
            "D1017310FCPCP171",
            "D1017310FCPCP172",
            "D1017310FCPCP173",
            "D1017310FCPCP174",
        ),
        "1001310": ("D1001310FCPCP005",),
        "1004310": (
            "D1004310FCPCP051",
            "D1004310FCPCP052",
            "D1004310FCPCP053",
            "D1004310FCPCP054",
            "D1004310FCPCP055",
            "D1004310FCPCP056",
        ),
        "1003110": (
            "D1003110FCPCP001",
            "D1003110FCPCP002",
            "D1003110FCPCP003",
            "D1003110FCPCP004",
            "D1003110FCPCP006",
            "D1003110FCPCP007",
            "D1003110FCPCP008",
            "D1003110FCPCP009",
            "D1003110FCPCP010",
            "D1003110FCPCP011",
            "D1003110FCPCP012",
            "D1003110FCPCP013",
            "D1003110FCPCP014",
            "D1003110FCPCP015",
            "D1003110FCPCP016",
            "D1003110FCPCP017",
            "D1003110FCPCP018",
            "D1003110FCPCP019",
            "D1003110FCPCP020",
            "D1003110FCPCP021",
            "D1003110FCPCP022",
            "D1003110FCPCP023",
            "D1003110FCPCP024",
            "D1003110FCPCP025",
            "D1003110FCPCP026",
            "D1003110FCPCP027",
            "D1003110FCPCP028",
            "D1003110FCPCP029",
            "D1003110FCPCP030",
            "D1003110FCPCP031",
            "D1003110FCPCP032",
            "D1003110FCPCP033",
            "D1003110FCPCP034",
            "D1003110FCPCP035",
            "D1003110FCPCP036",
            "D1003110FCPCP037",
            "D1003110FCPCP038",
            "D1003110FCPCP039",
            "D1003110FCPCP040",
            "D1003110FCPCP041",
            "D1003110FCPCP042",
            "D1003110FCPCP043",
            "D1003110FCPCP044",
            "D1003110FCPCP045",
            "D1003110FCPCP046",
            "D1003110FCPCP047",
            "D1003110FCPCP048",
            "D1003110FCPCP049",
        ),
        "1003611": (
            "D1003611FCPCP050",
            "D1003611FCPCP057",
            "D1003611FCPCP058",
            "D1003611FCPCP059",
            "D1003611FCPCP060",
            "D1003611FCPCP061",
            "D1003611FCPCP062",
            "D1003611FCPCP063",
            "D1003611FCPCP064",
            "D1003611FCPCP065",
        ),
        "1006110": ("D1006110FCPCP070",),
        "4001110": (
            "D4001110FCPCP700",
            "D4001110FCPCP701",
            "D4001110FCPCP702",
            "D4001110FCPCP703",
            "D4001110FCPCP704",
            "D4001110FCPCP705",
            "D4001110FCPCP706",
            "D4001110FCPCP707",
            "D4001110FCPCP708",
        ),
        "4007310": ("D4007310FCPCP726", "D4007310FCPCP727"),
    }

    class Valves(BaseModel):
        """
        밸브 설정
        """

        SIMM_DB_HOST: str
        SIMM_DB_PORT: str
        SIMM_DB_DATABASE: str
        SIMM_DB_USER: str
        SIMM_DB_PASSWORD: str
        SIMM_API_URL: str
        SIMM_CHART_FORMAT: str = "PNG"

        TOP_K_TOOLS: int = 0

    def __init__(
        self, pipeline_valves: BaseModel, llm: Ollama, embed_model: OllamaEmbedding
    ) -> None:
        """
        초기화
        """
        self.name: str = "SIMM Agent"

        _valves_dict: dict = pipeline_valves.model_dump()
        self.valves: SIMMAgent.Valves = self.Valves(**_valves_dict)
        self.engine: sqlalchemy.engine.base.Engine = self._init_db_connection()
        logger.debug(f"[DataBase Connected]: {self.valves.SIMM_DB_HOST}")

        self.llm: Ollama = llm
        self.embed_model: OllamaEmbedding = embed_model
        self.obj_index: ObjectIndex = self._init_embed_tools()

    def on_shutdown(self):
        """
        서버 종료
        """
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()
        pass

    def _init_embed_tools(self) -> ObjectIndex:
        """
        Agent Tool 임베딩
        """
        _simm_tools: list = [
            FunctionTool.from_defaults(fn=self.run_simm_simulation),
        ]
        _tool_mapping: BaseObjectNodeMapping = SimpleToolNodeMapping.from_objects(
            _simm_tools
        )

        return ObjectIndex.from_objects(
            objects=_simm_tools,
            object_mapping=_tool_mapping,
            index_cls=VectorStoreIndex,
            embed_model=self.embed_model,
        )

    def _has_simm_adjustment_condition(self, user_message: str) -> bool:
        """
        SIMM 요청에 강우량 변경 조건이 명시되어 있는지 확인합니다.
        """
        __message: str = user_message.lower()
        __keywords: tuple[str, ...] = (
            "%",
            "increase",
            "decrease",
            "set",
            "raise",
            "lower",
            "plus",
            "minus",
            "증가",
            "감소",
            "설정",
            "늘려",
            "줄여",
            "올려",
            "낮춰",
            "더해",
            "빼",
        )
        return any(__keyword in __message for __keyword in __keywords)

    def _normalize_simm_dam_name(self, dam_name: str) -> Optional[str]:
        """
        사용자 입력 댐명을 내부 표준 댐명으로 변환합니다.
        """
        __name: str = dam_name.strip().replace(" ", "")
        return self.SIMM_DAM_ALIASES.get(__name)

    def _get_simm_dam_by_name(self, dam_name: str) -> Optional[dict[str, Any]]:
        """
        표준 댐명에 해당하는 댐 메타데이터를 반환합니다.
        """
        __normalized_name: Optional[str] = self._normalize_simm_dam_name(dam_name)
        if not __normalized_name:
            return None
        for __dam in self.SIMM_DAM_ORDER:
            if __dam["name"] == __normalized_name:
                return __dam
        return None

    def _find_simm_dam_mentions(self, user_message: str) -> list[dict[str, Any]]:
        """
        사용자 문장에서 댐명 언급을 위치와 함께 찾습니다.
        """
        __mentions: list[dict[str, Any]] = []
        __seen: set[tuple[str, int]] = set()
        for __alias, __name in sorted(
            self.SIMM_DAM_ALIASES.items(), key=lambda x: len(x[0]), reverse=True
        ):
            for __match in re.finditer(re.escape(__alias), user_message):
                __key: tuple[str, int] = (__name, __match.start())
                if __key in __seen or any(
                    __item["start"] <= __match.start() < __item["end"]
                    for __item in __mentions
                ):
                    continue
                __dam: Optional[dict[str, Any]] = self._get_simm_dam_by_name(__name)
                if __dam:
                    __mentions.append(
                        {
                            "name": __dam["name"],
                            "dam_cd": __dam["dam_cd"],
                            "dam_scd": __dam["dam_scd"],
                            "order": __dam["order"],
                            "start": __match.start(),
                            "end": __match.end(),
                        }
                    )
                    __seen.add(__key)
        return sorted(__mentions, key=lambda x: x["start"])

    def _expand_simm_dam_range(self, user_message: str) -> dict[str, Any]:
        """
        사용자 문장의 댐 시작/종료 표현을 실제 댐 목록으로 확장합니다.
        """
        __mentions: list[dict[str, Any]] = self._find_simm_dam_mentions(user_message)
        if not __mentions:
            return {"dams": [], "missing": ["댐 구간 또는 대상 댐"]}

        if len(__mentions) >= 2:
            __start_dam: dict[str, Any] = __mentions[0]
            __end_dam: dict[str, Any] = __mentions[1]
        else:
            __only_dam: dict[str, Any] = __mentions[0]
            __after_text: str = user_message[__only_dam["end"] : __only_dam["end"] + 8]
            if "까지" in __after_text:
                __start_dam = self._get_simm_dam_by_name("평화의댐")
                __end_dam = __only_dam
            elif "부터" in __after_text:
                __start_dam = __only_dam
                __end_dam = self._get_simm_dam_by_name("팔당")
            else:
                __start_dam = __only_dam
                __end_dam = __only_dam

        if not __start_dam or not __end_dam:
            return {"dams": [], "missing": ["댐 구간"]}

        __start_cd: str = __start_dam["dam_cd"]
        __end_cd: str = __end_dam["dam_cd"]
        __downstream_path: tuple[str, ...] = self.SIMM_DOWNSTREAM_PATH_MAP.get(
            __start_cd, (__start_cd,)
        )
        if __end_cd not in __downstream_path:
            return {
                "dams": [],
                "start_dam": __start_dam,
                "end_dam": __end_dam,
                "missing": ["종료 댐은 시작 댐보다 하위 댐이어야 합니다"],
            }
        __end_index: int = __downstream_path.index(__end_cd)

        __selected_codes: list[str] = list(__downstream_path[: __end_index + 1])
        __search_index: int = 0
        while __search_index < len(__selected_codes):
            __flow_cd: str = __selected_codes[__search_index]
            for __join_cd in self.SIMM_CONFLUENCE_MAP.get(__flow_cd, ()):
                if __join_cd not in __selected_codes:
                    __selected_codes.append(__join_cd)
            __search_index += 1

        __selected_code_set: set[str] = set(__selected_codes)
        __dams: list[dict[str, Any]] = [
            __dam
            for __dam in sorted(self.SIMM_DAM_ORDER, key=lambda x: x["order"])
            if __dam["dam_cd"] in __selected_code_set
        ]
        return {
            "dams": __dams,
            "start_dam": __start_dam,
            "end_dam": __end_dam,
            "missing": [],
        }

    def _parse_simm_datetime(self, user_message: str) -> dict[str, Any]:
        """
        사용자 문장에서 날짜, 시간, 조회 기간을 파싱합니다.
        """
        __now: datetime = datetime.now().replace(second=0, microsecond=0)
        __suggested_start: datetime = __now.replace(
            minute=0 if __now.minute < 30 else 30
        )
        __date_match = re.search(
            r"(\d{2,4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", user_message
        )
        if not __date_match:
            __date_match = re.search(
                r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", user_message
            )
        if not __date_match:
            return {
                "missing": ["조회 날짜"],
                "start": None,
                "end": None,
                "std_date": None,
                "suggested_start": __suggested_start.strftime("%Y-%m-%d %H:%M:%S"),
            }

        __year: int = int(__date_match.group(1))
        if __year < 100:
            __year += 2000
        __month: int = int(__date_match.group(2))
        __day: int = int(__date_match.group(3))

        __time_match = re.search(r"(\d{1,2})\s*[:시]\s*(\d{1,2})?", user_message)
        __duration_match = None
        for __match in re.finditer(
            r"(?:예측|기간|동안|간)?\s*(\d{1,2})\s*일(?:간|치|예측)?", user_message
        ):
            if __match.start() > __date_match.end():
                __duration_match = __match
                break

        __missing: list[str] = []
        if not __time_match:
            __missing.append("조회 시작 시간")

        __start_dt: Optional[datetime] = None
        __end_dt: Optional[datetime] = None
        if __time_match:
            __hour: int = int(__time_match.group(1))
            __minute: int = int(__time_match.group(2) or 0)
            __start_dt = datetime(__year, __month, __day, __hour, __minute, 0)
            __duration_days: int = (
                int(__duration_match.group(1)) if __duration_match else 3
            )
            __end_dt = __start_dt + timedelta(days=__duration_days)

        return {
            "missing": __missing,
            "start": __start_dt.strftime("%Y-%m-%d %H:%M:%S") if __start_dt else None,
            "end": __end_dt.strftime("%Y-%m-%d %H:%M:%S") if __end_dt else None,
            "std_date": (
                __start_dt.strftime("%Y-%m-%d %H:%M:%S") if __start_dt else None
            ),
            "suggested_start": __suggested_start.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _parse_simm_adjustment(self, user_message: str) -> dict[str, Any]:
        """
        사용자 문장에서 강우량 조정 조건을 파싱합니다.
        """
        __operation: Optional[str] = None
        if any(
            __keyword in user_message
            for __keyword in ("증가", "늘려", "올려", "더해", "상향")
        ):
            __operation = "increase"
        elif any(
            __keyword in user_message
            for __keyword in ("감소", "줄여", "낮춰", "빼", "하향")
        ):
            __operation = "decrease"
        elif any(__keyword in user_message for __keyword in ("설정", "지정", "변경")):
            __operation = "set"

        __amount_match = re.search(r"(\d+(?:\.\d+)?)\s*%", user_message)
        __amount_type: str = "percent"
        if not __amount_match:
            __amount_match = re.search(r"값\s*(\d+(?:\.\d+)?)", user_message)
            if not __amount_match:
                __amount_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:로|으로)\s*(?:설정|지정|변경)", user_message
                )
            __amount_type = "value"

        __missing: list[str] = []
        if not __operation:
            __missing.append("강우량 증가/감소/설정 조건")
        if not __amount_match:
            __missing.append("강우량 조정 값")

        return {
            "missing": __missing,
            "operation": __operation,
            "amount": float(__amount_match.group(1)) if __amount_match else None,
            "amount_type": __amount_type,
        }

    def _parse_simm_adjustment_dams(
        self, user_message: str, selected_dams: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        사용자 문장에서 강우량 값을 수정할 댐을 별도로 파싱합니다.
        """
        __mentions: list[dict[str, Any]] = self._find_simm_dam_mentions(user_message)
        __target_dams: list[dict[str, Any]] = []
        __selected_codes: set[str] = {__dam["dam_cd"] for __dam in selected_dams}
        __explicit_mentions: list[dict[str, Any]] = __mentions[2:] if len(__mentions) >= 3 else []
        __is_hc_otf_request: bool = "비상방류" in user_message

        if __is_hc_otf_request:
            if self.SIMM_HC_DAM_CD not in __selected_codes:
                return {
                    "dams": [],
                    "missing": ["화천댐 비상방류량은 운영 범위에 화천댐이 포함되어야 합니다"],
                    "otf_adjustment": True,
                    "rainfall_adjustment": False,
                }
            __hc_dam: Optional[dict[str, Any]] = self._get_simm_dam_by_name("화천")
            __rainfall_adjustment: bool = any(
                __keyword in user_message
                for __keyword in ("강우", "기존 태그", "태그도", "값도")
            )
            return {
                "dams": [__hc_dam] if __hc_dam else [],
                "missing": [],
                "otf_adjustment": True,
                "rainfall_adjustment": __rainfall_adjustment,
            }

        if __explicit_mentions:
            __segment: str = user_message[
                __explicit_mentions[0]["start"] : __explicit_mentions[-1]["end"] + 16
            ]
            if any(__keyword in __segment for __keyword in ("만", "값", "증가", "감소", "설정")):
                for __mention in __explicit_mentions:
                    if __mention["dam_cd"] not in __selected_codes:
                        return {
                            "dams": [],
                            "missing": ["수정 대상 댐은 운영 범위에 포함되어야 합니다"],
                        }
                    if not any(
                        __dam["dam_cd"] == __mention["dam_cd"]
                        for __dam in __target_dams
                    ):
                        __target_dams.append(__mention)
                return {
                    "dams": __target_dams,
                    "missing": [],
                    "otf_adjustment": False,
                    "rainfall_adjustment": True,
                }

        for __idx, __mention in enumerate(__mentions):
            __next_start: int = (
                __mentions[__idx + 1]["start"]
                if __idx + 1 < len(__mentions)
                else __mention["end"] + 16
            )
            __after_text: str = user_message[__mention["end"] : __next_start]
            if "값" in __after_text and not any(
                __keyword in __after_text for __keyword in ("부터", "까지")
            ):
                if __mention["dam_cd"] not in __selected_codes:
                    return {
                        "dams": [],
                        "missing": ["수정 대상 댐은 운영 범위에 포함되어야 합니다"],
                    }
                if not any(
                    __dam["dam_cd"] == __mention["dam_cd"] for __dam in __target_dams
                ):
                    __target_dams.append(__mention)

        return {
            "dams": __target_dams or selected_dams,
            "missing": [],
            "otf_adjustment": False,
            "rainfall_adjustment": True,
        }

    def _parse_simm_user_request(self, user_message: str) -> dict[str, Any]:
        """
        SIMM 자연어 요청을 실행 가능한 구조로 변환합니다.
        """
        __datetime_info: dict[str, Any] = self._parse_simm_datetime(user_message)
        __range_info: dict[str, Any] = self._expand_simm_dam_range(user_message)
        __adjustment_info: dict[str, Any] = self._parse_simm_adjustment(user_message)
        __dams: list[dict[str, Any]] = __range_info.get("dams", [])
        __adjustment_dam_info: dict[str, Any] = self._parse_simm_adjustment_dams(
            user_message, __dams
        )
        __adjustment_dams: list[dict[str, Any]] = __adjustment_dam_info.get(
            "dams", []
        )
        __otf_adjustment: bool = bool(__adjustment_dam_info.get("otf_adjustment"))
        __rainfall_adjustment: bool = bool(
            __adjustment_dam_info.get("rainfall_adjustment", True)
        )
        __tag_list: list[str] = []
        __tag_dam_map: dict[str, str] = {}
        for __dam in __adjustment_dams:
            for __tag in self.SIMM_DAM_TAG_MAP.get(__dam["dam_cd"], ()):
                __tag_list.append(__tag)
                __tag_dam_map[__tag] = __dam["dam_cd"]

        __missing: list[str] = []
        __missing.extend(__datetime_info.get("missing", []))
        __missing.extend(__range_info.get("missing", []))
        __missing.extend(__adjustment_dam_info.get("missing", []))
        __missing.extend(__adjustment_info.get("missing", []))

        return {
            "missing": __missing,
            "dams": __dams,
            "adjustment_dams": __adjustment_dams,
            "otf_adjustment": __otf_adjustment,
            "rainfall_adjustment": __rainfall_adjustment,
            "start_dam": __range_info.get("start_dam"),
            "end_dam": __range_info.get("end_dam"),
            "tag_list": __tag_list,
            "tag_dam_map": __tag_dam_map,
            "start": __datetime_info.get("start"),
            "end": __datetime_info.get("end"),
            "std_date": __datetime_info.get("std_date"),
            "suggested_start": __datetime_info.get("suggested_start"),
            "operation": __adjustment_info.get("operation"),
            "amount": __adjustment_info.get("amount"),
            "amount_type": __adjustment_info.get("amount_type"),
        }

    def _build_simm_missing_message(self, parsed_request: dict[str, Any]) -> str:
        """
        부족한 SIMM 요청 정보를 사용자 재질의 문장으로 변환합니다.
        """
        __missing: list[str] = parsed_request.get("missing", [])
        if not __missing:
            return ""
        __messages: list[str] = []
        if "조회 날짜" in __missing:
            __suggested_start: Optional[str] = parsed_request.get("suggested_start")
            if __suggested_start:
                __suggested_dt: datetime = datetime.strptime(
                    __suggested_start, "%Y-%m-%d %H:%M:%S"
                )
                __messages.append(
                    "조회 날짜 데이터가 없습니다. "
                    f"{__suggested_dt.strftime('%Y년 %m월 %d일 %H시 %M분')}으로 설정하시겠습니까?"
                )
        if "조회 시작 시간" in __missing:
            __messages.append("조회 시작 시간이 필요합니다. 예: 09시 30분")
        if "댐 구간 또는 대상 댐" in __missing or "댐 구간" in __missing:
            __messages.append("모의운영 대상 댐 구간이 필요합니다. 예: 평화부터 팔당까지")
        if "종료 댐은 시작 댐보다 하위 댐이어야 합니다" in __missing:
            __messages.append("종료 댐은 시작 댐보다 하위 댐으로 입력해 주세요.")
        if "수정 대상 댐은 운영 범위에 포함되어야 합니다" in __missing:
            __messages.append("수정 대상 댐은 선택한 모의운영 범위 안에 포함되어야 합니다.")
        if "화천댐 비상방류량은 운영 범위에 화천댐이 포함되어야 합니다" in __missing:
            __messages.append("화천댐 비상방류량 수정은 모의운영 범위에 화천댐이 포함되어야 합니다.")
        if "강우량 증가/감소/설정 조건" in __missing:
            __messages.append("강우량 증가, 감소, 설정 중 하나를 입력해 주세요.")
        if "강우량 조정 값" in __missing:
            __messages.append("강우량 조정 값이 필요합니다. 예: 10% 증가 또는 0.1로 설정")

        return " ".join(__messages) if __messages else (
            "모의운영 실행을 위해 다음 정보가 더 필요합니다: "
            + ", ".join(__missing)
            + "."
        )

    def _build_simm_target_mask(self, __selected_dams: list[dict[str, Any]]) -> str:
        """
        PH, HC, CC, SY, UA, CP, DA, CJ, GS, JJ, HS, PD, SJ, BS 순서의 14자리 대상 댐 마스크를 생성합니다.
        """
        __selected_dam_codes: set[str] = {
            __dam.get("dam_cd", "") for __dam in __selected_dams
        }
        return "".join(
            "1" if __dam["dam_cd"] in __selected_dam_codes else "0"
            for __dam in sorted(self.SIMM_DAM_ORDER, key=lambda __item: __item["order"])
        )

    def _build_simm_id(
        self,
        __parsed_request: dict[str, Any],
        __user_id: str,
        __user_ip: str,
    ) -> str:
        """
        SIMM_ID를 모형 타입, 대상 댐 마스크, 기본 구분값, 시작시간, 사용자 정보로 생성합니다.
        """
        __selected_dams: list[dict[str, Any]] = __parsed_request.get("dams", [])
        __target_mask: str = self._build_simm_target_mask(__selected_dams)
        __start_dt: datetime = datetime.strptime(
            __parsed_request.get("start"), "%Y-%m-%d %H:%M:%S"
        )
        __start_text: str = __start_dt.strftime("%Y%m%d%H%M")
        return (
            f"{self.SIMM_MODEL_TYPE}{__target_mask}0"
            f"{__start_text}{__user_id}:{__user_ip}"
        )

    def _extract_context_user_id(self, __user__: Optional[dict]) -> str:
        """
        Open-WebUI 사용자 컨텍스트에서 SIMM 저장용 사용자 ID를 추출합니다.
        """
        if not __user__:
            return f"ai_{self.SIMM_USER_ID}"
        __raw_user_id: str = str(
            __user__.get("name")
            or __user__.get("username")
            or __user__.get("email")
            or __user__.get("id")
            or self.SIMM_USER_ID
        )
        __safe_user_id: str = re.sub(r"[^0-9A-Za-z_.@-]+", "_", __raw_user_id).strip("_")
        return f"ai_{__safe_user_id or self.SIMM_USER_ID}"

    def _get_local_ip(self) -> str:
        """
        현재 Python 프로세스가 실행 중인 서버/컨테이너의 로컬 IP를 반환합니다.
        """
        try:
            __host_name: str = socket.gethostname()
            __addr_infos: list = socket.getaddrinfo(
                __host_name, None, socket.AF_INET
            )
            for __addr_info in __addr_infos:
                __ip: str = str(__addr_info[4][0])
                if __ip and not __ip.startswith("127."):
                    return __ip
        except OSError:
            pass
        try:
            __ip: str = socket.gethostbyname(socket.gethostname())
            if __ip and not __ip.startswith("127."):
                return __ip
        except OSError:
            pass
        return self.SIMM_USER_IP

    def _init_db_connection(self) -> sqlalchemy.engine.base.Engine:
        """
        Database 연결
        Oracle + oracledb 드라이버 사용 설정 (형식: oracle+oracledb://user:pass@host:port/?service_name=db)
        """
        _init_oracle_client()

        _connection_url: str = (
            f"oracle+oracledb://{self.valves.SIMM_DB_USER}:{self.valves.SIMM_DB_PASSWORD}@"
            f"{self.valves.SIMM_DB_HOST}:{self.valves.SIMM_DB_PORT}/"
            f"?service_name={self.valves.SIMM_DB_DATABASE}"
        )
        _engine: sqlalchemy.engine.base.Engine = create_engine(
            _connection_url
        )  # , echo=True)

        return _engine

    def _get_simm_api_data(
        self,
        start: str = Field(
            ..., description="조회 시작 일시 (형식: 'YYYY-MM-DD HH:MM:SS')"
        ),
        end: str = Field(
            ..., description="조회 종료 일시 (형식: 'YYYY-MM-DD HH:MM:SS')"
        ),
        tag_list: str = Field(
            ...,
            description="조회할 태그 리스트 (쉼표로 구분된 문자열. 예: 'D1009710FCPCP112,D1009710FCPCP113')",
        ),
        ip: str = Field(default="[CLIENTIP]", description="요청 IP 주소"),
        channel: str = Field(default="/queue/phdWeb", description="요청 채널"),
    ) -> list[Any]:
        """
        지정한 기간과 태그 목록으로 SIMM 강우 트렌드 API를 호출합니다.
        """
        ### --------------------------------------------------
        ### 강우 데이터 조회
        ### --------------------------------------------------
        logger.debug(f"[Start]: ---------- _get_simm_api_data ----------")

        if hasattr(ip, "default"):
            ip = ip.default
        if hasattr(channel, "default"):
            channel = channel.default

        _tags: list[str] = [
            t.strip().strip("\"'")
            for t in tag_list.replace("[", "").replace("]", "").split(",")
            if t.strip()
        ]
        _request_tags: list[str] = [
            __tag for __tag in _tags if not __tag.endswith("FCPCP000")
        ]
        if not _request_tags:
            logger.warning("[SIMM API] 실제 조회 가능한 강우 태그가 없습니다.")
            return []
        if len(_request_tags) != len(_tags):
            logger.debug(
                "[SIMM API] 임시 태그 제외: "
                + ",".join(__tag for __tag in _tags if __tag.endswith("FCPCP000"))
            )

        _connection_url: str = self.valves.SIMM_API_URL + "/KHNP/EA/EaiRequest_get"
        _headers: dict[str, str] = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
        }

        def __request_tags(
            __tags: list[str],
            __retry_count: int = 3,
        ) -> list[Any]:
            __json: dict[str, Any] = {
                "cmd": "fetch",
                "param": {
                    "start": start,
                    "end": end,
                    "function": "getTrend",
                    "tagList": __tags,
                },
            }
            __form_data: dict[str, str] = {
                "json": json.dumps(__json, separators=(",", ":")),
                "ip": ip,
                "channel": channel,
            }
            __result: list[Any] = []
            for __attempt in range(__retry_count):
                __response: requests.Response = requests.post(
                    _connection_url, data=__form_data, headers=_headers
                )
                if not (200 <= __response.status_code < 300):
                    logger.error(
                        f"[API Error] API 호출 중 오류가 발생했습니다: {__response.status_code}"
                    )
                    raise Exception(
                        f"API 호출 중 오류가 발생했습니다: {__response.text}({__response.status_code})"
                    )
                __response_data: Any = __response.json()
                if isinstance(__response_data, list):
                    __result = __response_data
                elif __response_data:
                    __result = [__response_data]
                else:
                    __result = []
                if __result:
                    return __result
                if __attempt < __retry_count - 1:
                    logger.debug(
                        f"[SIMM API] 빈 응답 재시도 {__attempt + 1}/{__retry_count - 1}: {','.join(__tags)}"
                    )
            return __result

        _result: list[Any] = __request_tags(_request_tags)
        if not _result and len(_request_tags) > 1:
            _merged_result: list[Any] = []
            for __tag in _request_tags:
                __tag_result: list[Any] = __request_tags([__tag])
                if __tag_result:
                    _merged_result.extend(__tag_result)
            _result = _merged_result

        logger.debug(f"[API Result]: {_result}")

        return _result

    def _apply_simm_rainfall_adjustment(
        self,
        api_data: Any = Field(
            ..., description="_get_simm_api_data 함수가 반환한 강우 데이터"
        ),
        operation: str = Field(
            ...,
            description="강우량 조정 방식. increase/decrease/set 또는 증가/감소/설정 중 하나",
        ),
        amount: float = Field(..., description="조정 값. 10% 조정이면 10을 입력"),
        amount_type: str = Field(
            default="percent",
            description="조정 값 유형. percent/value 또는 퍼센트/값 중 하나",
        ),
        value_keys: Optional[str] = Field(
            default=None, description="조정할 강우량 값 키 목록. 여러 개면 쉼표로 구분"
        ),
    ) -> dict[str, Any]:
        """
        SIMM API 응답 구조를 유지한 채 강우량 값만 증가, 감소 또는 고정값으로 설정합니다.
        """
        logger.debug(f"[Call]: ---------- _apply_simm_rainfall_adjustment ----------")

        if isinstance(api_data, str):
            try:
                _data: Any = json.loads(api_data)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"api_data는 JSON 객체/배열 또는 JSON 문자열이어야 합니다: {str(e)}"
                )
        else:
            _data: Any = json.loads(json.dumps(api_data))

        if hasattr(amount_type, "default"):
            amount_type = amount_type.default
        if hasattr(value_keys, "default"):
            value_keys = value_keys.default
        try:
            if isinstance(amount, str):
                _amount_match = re.search(r"-?\d+(?:\.\d+)?", amount)
                if not _amount_match:
                    raise ValueError
                _amount = float(_amount_match.group(0))
            else:
                _amount: float = float(amount)
        except (TypeError, ValueError):
            raise ValueError("amount는 숫자여야 합니다. 예: 10% 증가이면 amount=10")

        def _normalize_operation(value: Any) -> str:
            _value: str = str(value).strip().lower()
            _operation_aliases: dict[str, str] = {
                "increase": "increase",
                "증가": "increase",
                "늘려": "increase",
                "올려": "increase",
                "더해": "increase",
                "상향": "increase",
                "decrease": "decrease",
                "감소": "decrease",
                "줄여": "decrease",
                "낮춰": "decrease",
                "빼": "decrease",
                "하향": "decrease",
                "set": "set",
                "설정": "set",
                "지정": "set",
                "변경": "set",
            }
            for _alias, _normalized in _operation_aliases.items():
                if _alias in _value:
                    return _normalized
            return _value

        def _normalize_amount_type(value: Any) -> str:
            _value: str = str(value).strip().lower()
            if _value in {"percent", "%", "퍼센트", "비율"}:
                return "percent"
            if _value in {"value", "값", "절대값", "절대"}:
                return "value"
            return _value

        _operation: str = _normalize_operation(operation)
        _amount_type: str = _normalize_amount_type(amount_type)
        if _operation not in {"increase", "decrease", "set"}:
            raise ValueError(
                "operation은 increase/decrease/set 또는 증가/감소/설정 중 하나여야 합니다."
            )
        if _amount_type not in {"percent", "value"}:
            raise ValueError(
                "amount_type은 percent/value 또는 퍼센트/값 중 하나여야 합니다."
            )

        _default_value_keys: set[str] = {
            "value",
            "val",
            "y",
            "rain",
            "rainfall",
            "pcp",
            "prcp",
            "rf",
            "obsrvalue",
            "data_value",
            "data",
        }
        _value_keys: set[str] = _default_value_keys
        if value_keys:
            _value_keys = {
                _key.strip().lower() for _key in value_keys.split(",") if _key.strip()
            }

        _exclude_key_parts: tuple[str, ...] = (
            "date",
            "time",
            "timestamp",
            "tag",
            "id",
            "code",
            "cd",
            "dam",
            "year",
            "month",
            "day",
            "hour",
            "min",
            "sec",
            "index",
        )
        _changed_count: int = 0

        def _is_number(value: Any) -> bool:
            if isinstance(value, bool):
                return False
            if isinstance(value, (int, float)):
                return True
            if isinstance(value, str):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False
            return False

        def _looks_temporal(value: Any) -> bool:
            if not isinstance(value, str):
                return False
            return "-" in value or ":" in value or "T" in value

        def _looks_json(value: str) -> bool:
            _value: str = value.strip()
            return (_value.startswith("{") and _value.endswith("}")) or (
                _value.startswith("[") and _value.endswith("]")
            )

        def _should_adjust_key(key: str) -> bool:
            _key: str = key.strip().lower()
            if any(_part in _key for _part in _exclude_key_parts):
                return False
            return _key in _value_keys or any(
                _part in _key for _part in ("rain", "pcp", "prcp", "rf")
            )

        def _adjust_value(value: Any) -> Any:
            nonlocal _changed_count
            _original_is_str: bool = isinstance(value, str)
            _original_is_int: bool = isinstance(value, int) and not isinstance(
                value, bool
            )
            _number: float = float(value)

            if _operation == "increase":
                _adjusted: float = (
                    _number * (1 + _amount / 100)
                    if _amount_type == "percent"
                    else _number + _amount
                )
            elif _operation == "decrease":
                _adjusted: float = (
                    _number * (1 - _amount / 100)
                    if _amount_type == "percent"
                    else _number - _amount
                )
            else:
                _adjusted: float = _amount

            _adjusted = round(_adjusted, 6)
            _changed_count += 1
            if _original_is_str:
                return str(_adjusted).rstrip("0").rstrip(".")
            if _original_is_int and _adjusted.is_integer():
                return int(_adjusted)
            return _adjusted

        def _walk(value: Any, key_hint: Optional[str] = None) -> Any:
            if isinstance(value, dict):
                return {_key: _walk(_value, _key) for _key, _value in value.items()}
            if isinstance(value, list):
                if value and _looks_temporal(value[0]):
                    return [
                        value[0],
                        *[
                            _adjust_value(_item) if _is_number(_item) else _walk(_item)
                            for _item in value[1:]
                        ],
                    ]
                return [_walk(_item, key_hint) for _item in value]
            if isinstance(value, str) and _looks_json(value):
                try:
                    _parsed_value: Any = json.loads(value)
                    _adjusted_value: Any = _walk(_parsed_value, key_hint)
                    return json.dumps(_adjusted_value, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            if key_hint and _should_adjust_key(key_hint) and _is_number(value):
                return _adjust_value(value)
            return value

        _adjusted_data: Any = _walk(_data)
        return {
            "operation": _operation,
            "amount": _amount,
            "amount_type": _amount_type,
            "changed_count": _changed_count,
            "data": _adjusted_data,
        }

    def _extract_simm_data_sets(self, api_data: Any) -> list[dict[str, Any]]:
        """
        SIMM API 응답에서 모든 body.dataSet 배열을 찾아 반환합니다.
        """
        if isinstance(api_data, str):
            __data: Any = json.loads(api_data)
        else:
            __data: Any = json.loads(json.dumps(api_data))

        if isinstance(__data, dict) and "data" in __data:
            __data = __data["data"]

        __data_sets: list[dict[str, Any]] = []

        def __walk(value: Any):
            if isinstance(value, str):
                __text: str = value.strip()
                if (__text.startswith("{") and __text.endswith("}")) or (
                    __text.startswith("[") and __text.endswith("]")
                ):
                    try:
                        __walk(json.loads(__text))
                    except json.JSONDecodeError:
                        pass
                return
            if isinstance(value, list):
                for __item in value:
                    __walk(__item)
                return
            if isinstance(value, dict):
                __body: Any = value.get("body")
                if isinstance(__body, dict) and isinstance(__body.get("dataSet"), list):
                    __data_sets.extend(__body["dataSet"])
                for __item in value.values():
                    __walk(__item)

        __walk(__data)
        return __data_sets

    def _build_simm_tag_val_columns(
        self,
        values: list[dict[str, Any]],
        start_date: Optional[str] = None,
        fixed_value: Optional[float] = None,
    ) -> dict[str, Optional[str]]:
        """
        API values 배열을 TAG_VAL1~TAG_VAL12 저장 문자열로 변환합니다.
        """
        __min_time: Optional[datetime] = None
        if start_date:
            try:
                __min_time = datetime.strptime(
                    start_date[:19], "%Y-%m-%d %H:%M:%S"
                ) + timedelta(minutes=10)
            except ValueError:
                __min_time = None

        __pairs: list[str] = []
        for __item in values:
            if not isinstance(__item, dict):
                continue
            __timestamp: Optional[str] = (
                __item.get("timeStamp") or __item.get("timestamp") or __item.get("time")
            )
            if not __timestamp:
                continue
            __timestamp_text: str = str(__timestamp).replace("T", " ")
            try:
                __time_dt: datetime = datetime.strptime(
                    __timestamp_text[:19], "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                continue
            if __min_time and __time_dt < __min_time:
                continue
            __time_text: str = __time_dt.strftime("%Y-%m-%d %H:%M")
            try:
                if fixed_value is not None:
                    __value_text = str(round(float(fixed_value), 6)).rstrip("0").rstrip(".")
                else:
                    __value_text = f"{float(__item.get('value', 0)):0.2f}"
            except (TypeError, ValueError):
                __value_text: str = "0.00"
            __pairs.append(f"{__time_text},{__value_text};")

        __columns: dict[str, Optional[str]] = {
            f"tag_val{__idx}": None for __idx in range(1, 13)
        }
        for __idx in range(12):
            __chunk: list[str] = __pairs[__idx * 100 : (__idx + 1) * 100]
            if __chunk:
                __columns[f"tag_val{__idx + 1}"] = "".join(__chunk)
        return __columns

    def _insert_sim_st_inputs(
        self,
        adjusted_api_data: Any = Field(
            ...,
            description="_apply_simm_rainfall_adjustment 함수가 반환한 조정 강우 데이터 또는 그 data 값",
        ),
        simm_id: str = Field(..., max_length=100, description="SIMM ID"),
        ins_date: str = Field(
            ...,
            pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
            description="저장일시 (형식: YYYY-MM-DD HH:MM:SS)",
        ),
        tag_dam_map: str | dict[str, str] | None = Field(
            default=None,
            description='tagName별 dam_cd JSON 문자열. 예: {"D100...":"1013310"}',
        ),
        start_date: Optional[str] = Field(
            default=None,
            description="모의운영 시작일시. TAG_VAL 저장 시 시작일시 10분 후부터 저장합니다.",
        ),
        otf_adjustment: bool = False,
        rainfall_adjustment: bool = True,
        ai_type: str = Field(default="TM", description="관측 강우 타입. 기본값 TM"),
    ) -> str:
        """
        조정된 강우 시계열을 [DBSCHEMA].TB_SIM_ST_INPUT_1 테이블에 등록하거나 갱신합니다.

        _apply_simm_rainfall_adjustment 결과의 body.dataSet[].values[]를
        'YYYY-MM-DD HH:MM,value;' 형식으로 변환하고, 100개 시점 단위로
        TAG_VAL1~TAG_VAL12에 나누어 저장합니다. TAG_CD에는 API 응답의
        tagName을 그대로 저장하고, AI_TYPE은 기본값 TM을 사용합니다.
        같은 트랜잭션에서 전체 댐 기준의 TB_SIM_ST_INPUT_2도 함께 저장합니다.
        화천댐 비상방류량 OTF 태그는 항상 함께 저장합니다.
        """
        logger.debug(f"[Call]: ---------- _insert_sim_st_inputs ----------")

        if hasattr(tag_dam_map, "default"):
            tag_dam_map = tag_dam_map.default
        if hasattr(start_date, "default"):
            start_date = start_date.default
        if hasattr(otf_adjustment, "default"):
            otf_adjustment = otf_adjustment.default
        if hasattr(rainfall_adjustment, "default"):
            rainfall_adjustment = rainfall_adjustment.default
        if hasattr(ai_type, "default"):
            ai_type = ai_type.default

        _source_data: Any = adjusted_api_data
        if isinstance(adjusted_api_data, dict) and "data" in adjusted_api_data:
            _source_data = adjusted_api_data["data"]

        _data_sets: list[dict[str, Any]] = self._extract_simm_data_sets(_source_data)
        if not _data_sets:
            return "[시스템 알림] 저장할 강우 시계열 데이터가 없습니다. 절대로 임의의 데이터를 지어내지 마세요. 사용자에게 '저장할 강우 시계열 데이터가 없습니다.'라고 답변하세요."

        _tag_dam_map: dict[str, str] = {}
        if isinstance(tag_dam_map, dict):
            _tag_dam_map = tag_dam_map
        elif tag_dam_map:
            try:
                _tag_dam_map = json.loads(tag_dam_map)
            except (TypeError, json.JSONDecodeError) as e:
                logger.error(
                    f"[검증 실패] tag_dam_map JSON 형식이 올바르지 않습니다: {str(e)}"
                )
                return "[시스템 알림] JSON 형식이 올바르지 않아 검증에 실패하였습니다. 절대로 임의의 데이터를 지어내지 마세요. 사용자에게 'JSON 검증에 실패하였습니다.'라고 답변하세요."

        _query: TextClause = text("""
            MERGE INTO [DBSCHEMA].TB_SIM_ST_INPUT_1 T
            USING (
                SELECT
                    :simm_id AS SIMM_ID,
                    :dam_cd AS DAM_CD,
                    :tag_cd AS TAG_CD,
                    :tag_val1 AS TAG_VAL1,
                    :tag_val2 AS TAG_VAL2,
                    :tag_val3 AS TAG_VAL3,
                    :tag_val4 AS TAG_VAL4,
                    :tag_val5 AS TAG_VAL5,
                    :tag_val6 AS TAG_VAL6,
                    :tag_val7 AS TAG_VAL7,
                    :tag_val8 AS TAG_VAL8,
                    :tag_val9 AS TAG_VAL9,
                    :tag_val10 AS TAG_VAL10,
                    :tag_val11 AS TAG_VAL11,
                    :tag_val12 AS TAG_VAL12,
                    TO_DATE(:ins_date, 'YYYY-MM-DD HH24:MI:SS') AS INS_DATE,
                    :ai_type AS AI_TYPE
                FROM DUAL
            ) S
            ON (
                T.SIMM_ID = S.SIMM_ID AND
                T.DAM_CD = S.DAM_CD AND
                T.TAG_CD = S.TAG_CD AND
                T.INS_DATE = S.INS_DATE
            )
            WHEN MATCHED THEN
                UPDATE SET
                    T.TAG_VAL1 = S.TAG_VAL1,
                    T.TAG_VAL2 = S.TAG_VAL2,
                    T.TAG_VAL3 = S.TAG_VAL3,
                    T.TAG_VAL4 = S.TAG_VAL4,
                    T.TAG_VAL5 = S.TAG_VAL5,
                    T.TAG_VAL6 = S.TAG_VAL6,
                    T.TAG_VAL7 = S.TAG_VAL7,
                    T.TAG_VAL8 = S.TAG_VAL8,
                    T.TAG_VAL9 = S.TAG_VAL9,
                    T.TAG_VAL10 = S.TAG_VAL10,
                    T.TAG_VAL11 = S.TAG_VAL11,
                    T.TAG_VAL12 = S.TAG_VAL12,
                    T.AI_TYPE = S.AI_TYPE
            WHEN NOT MATCHED THEN
                INSERT (
                    SIMM_ID, DAM_CD, TAG_CD,
                    TAG_VAL1, TAG_VAL2, TAG_VAL3, TAG_VAL4, TAG_VAL5, TAG_VAL6,
                    TAG_VAL7, TAG_VAL8, TAG_VAL9, TAG_VAL10, TAG_VAL11, TAG_VAL12,
                    INS_DATE, AI_TYPE
                )
                VALUES (
                    S.SIMM_ID, S.DAM_CD, S.TAG_CD,
                    S.TAG_VAL1, S.TAG_VAL2, S.TAG_VAL3, S.TAG_VAL4, S.TAG_VAL5, S.TAG_VAL6,
                    S.TAG_VAL7, S.TAG_VAL8, S.TAG_VAL9, S.TAG_VAL10, S.TAG_VAL11, S.TAG_VAL12,
                    S.INS_DATE, S.AI_TYPE
                )
        """)

        _rows: list[dict[str, Any]] = []
        for _data_set in _data_sets:
            _tag_cd: Optional[str] = _data_set.get("tagName")
            if not _tag_cd:
                continue
            if not rainfall_adjustment:
                continue
            _dam_cd: Optional[str] = _tag_dam_map.get(_tag_cd)
            if not _dam_cd:
                return f"[시스템 알림] 태그 '{_tag_cd}'에 해당하는 댐관리코드가 없어 검증에 실패하였습니다. 절대로 임의의 데이터를 지어내지 마세요. 사용자에게 '댐관리코드 검증에 실패하였습니다.'라고 답변하세요."
            _values: list[dict[str, Any]] = _data_set.get("values", [])
            _tag_val_columns: dict[str, Optional[str]] = (
                self._build_simm_tag_val_columns(_values, start_date=start_date)
            )
            _rows.append(
                {
                    "simm_id": simm_id,
                    "dam_cd": _dam_cd,
                    "tag_cd": _tag_cd,
                    "ins_date": ins_date,
                    "ai_type": ai_type,
                    **_tag_val_columns,
                }
            )

        _timeline_values: list[dict[str, Any]] = []
        for _data_set in _data_sets:
            _values: list[dict[str, Any]] = _data_set.get("values", [])
            if _values:
                _timeline_values = _values
                break

        if _timeline_values:
            _otf_value: float = 0.0
            if otf_adjustment:
                _operation: str = str(adjusted_api_data.get("operation", "set"))
                _amount: float = float(adjusted_api_data.get("amount", 0))
                _amount_type: str = str(adjusted_api_data.get("amount_type", "value"))
                if _operation == "set":
                    _otf_value = _amount
                elif _operation == "increase" and _amount_type == "value":
                    _otf_value = _amount
                elif _operation == "decrease" and _amount_type == "value":
                    _otf_value = -_amount

            _otf_columns: dict[str, Optional[str]] = self._build_simm_tag_val_columns(
                _timeline_values,
                start_date=start_date,
                fixed_value=_otf_value,
            )
            for _otf_tag in self.SIMM_HC_OTF_TAGS:
                _rows.append(
                    {
                        "simm_id": simm_id,
                        "dam_cd": self.SIMM_HC_DAM_CD,
                        "tag_cd": _otf_tag,
                        "ins_date": ins_date,
                        "ai_type": ai_type,
                        **_otf_columns,
                    }
                )

        if not _rows:
            return "[시스템 알림] 저장할 강우 시계열 데이터가 없습니다. 절대로 임의의 데이터를 지어내지 마세요. 사용자에게 '저장할 태그 데이터가 없어 실패하였습니다.'라고 답변하세요."

        try:
            with self.engine.connect() as _conn:
                with _conn.begin():
                    for _row in _rows:
                        _conn.execute(_query, _row)
                    self._insert_sim_st_input_2(
                        simm_id=simm_id,
                        rom_type="1",
                        connection=_conn,
                    )
            return (
                f"TB_SIM_ST_INPUT_1에 강우 시계열 {len(_rows)}건을 등록/갱신했습니다. "
                "TB_SIM_ST_INPUT_2 제한수위도 함께 등록/갱신했습니다. "
                "다음으로 TB_SIM_ST_IF 실행 요청을 등록하세요."
            )
        except Exception as e:
            logger.error(
                f"[Insert Error] TB_SIM_ST_INPUT_1 저장 중 오류가 발생했습니다: {str(e)}"
            )
            return "[시스템 알림] 강우 시계열 저장 중 내부 오류가 발생했습니다. 더 이상 도구를 찾지 말고 즉시 사용자에게 '모의운영 실행 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.'라고 답변하세요."

    def _insert_sim_st_input_2(
        self,
        simm_id: str,
        rom_type: str = "1",
        connection: Optional[sqlalchemy.engine.Connection] = None,
    ) -> str:
        """
        전체 댐의 제한수위(LMT_WL)를 [DBSCHEMA].TB_SIM_ST_INPUT_2에 등록하거나 갱신합니다.

        TARGET_WL에는 댐별 LMT_WL을 저장하고, ROM_TYPE은 현재 기준값 "1"을 사용합니다.
        WTD_LIMIT는 1000000, QPF_YN은 "N"으로 저장합니다.
        """
        logger.debug(f"[Call]: ---------- _insert_sim_st_input_2 ----------")

        __rows: list[dict[str, Any]] = []
        for __dam in sorted(self.SIMM_DAM_ORDER, key=lambda __item: __item["order"]):
            __dam_cd: str = __dam.get("dam_cd", "")
            __target_wl: Optional[float] = self.SIMM_DAM_LMT_WL_MAP.get(__dam_cd)
            if __target_wl is None:
                if connection:
                    raise ValueError(f"댐관리코드 '{__dam_cd}'의 제한수위가 없습니다.")
                return (
                    f"[시스템 알림] 댐관리코드 '{__dam_cd}'의 제한수위가 없어 "
                    "TB_SIM_ST_INPUT_2 저장을 중단했습니다."
                )
            __row: dict[str, Any] = {
                "simm_id": simm_id,
                "dam_cd": __dam_cd,
                "rom_type": rom_type,
                "target_wl": __target_wl,
                "rom_sim": None,
                "stop_plan": None,
                "wtd_limit": 1000000,
                "bgin_wl": None,
                "qpf_yn": "N",
            }
            for __idx in range(1, 13):
                __row[f"otf_val{__idx}"] = None
            __rows.append(__row)

        if not __rows:
            if connection:
                raise ValueError("TB_SIM_ST_INPUT_2에 저장할 활성화 댐이 없습니다.")
            return "[시스템 알림] TB_SIM_ST_INPUT_2에 저장할 활성화 댐이 없습니다."

        __query: TextClause = text("""
            MERGE INTO [DBSCHEMA].TB_SIM_ST_INPUT_2 T
            USING (
                SELECT
                    :simm_id AS SIMM_ID,
                    :dam_cd AS DAM_CD,
                    :rom_type AS ROM_TYPE,
                    :target_wl AS TARGET_WL,
                    :rom_sim AS ROM_SIM,
                    :otf_val1 AS OTF_VAL1,
                    :otf_val2 AS OTF_VAL2,
                    :otf_val3 AS OTF_VAL3,
                    :otf_val4 AS OTF_VAL4,
                    :otf_val5 AS OTF_VAL5,
                    :otf_val6 AS OTF_VAL6,
                    :otf_val7 AS OTF_VAL7,
                    :otf_val8 AS OTF_VAL8,
                    :otf_val9 AS OTF_VAL9,
                    :otf_val10 AS OTF_VAL10,
                    :otf_val11 AS OTF_VAL11,
                    :otf_val12 AS OTF_VAL12,
                    :stop_plan AS STOP_PLAN,
                    :wtd_limit AS WTD_LIMIT,
                    :bgin_wl AS BGIN_WL,
                    :qpf_yn AS QPF_YN
                FROM DUAL
            ) S
            ON (
                T.SIMM_ID = S.SIMM_ID AND
                T.DAM_CD = S.DAM_CD
            )
            WHEN MATCHED THEN
                UPDATE SET
                    T.ROM_TYPE = S.ROM_TYPE,
                    T.TARGET_WL = S.TARGET_WL,
                    T.ROM_SIM = S.ROM_SIM,
                    T.OTF_VAL1 = S.OTF_VAL1,
                    T.OTF_VAL2 = S.OTF_VAL2,
                    T.OTF_VAL3 = S.OTF_VAL3,
                    T.OTF_VAL4 = S.OTF_VAL4,
                    T.OTF_VAL5 = S.OTF_VAL5,
                    T.OTF_VAL6 = S.OTF_VAL6,
                    T.OTF_VAL7 = S.OTF_VAL7,
                    T.OTF_VAL8 = S.OTF_VAL8,
                    T.OTF_VAL9 = S.OTF_VAL9,
                    T.OTF_VAL10 = S.OTF_VAL10,
                    T.OTF_VAL11 = S.OTF_VAL11,
                    T.OTF_VAL12 = S.OTF_VAL12,
                    T.STOP_PLAN = S.STOP_PLAN,
                    T.WTD_LIMIT = S.WTD_LIMIT,
                    T.BGIN_WL = S.BGIN_WL,
                    T.QPF_YN = S.QPF_YN
            WHEN NOT MATCHED THEN
                INSERT (
                    SIMM_ID, DAM_CD, ROM_TYPE, TARGET_WL, ROM_SIM,
                    OTF_VAL1, OTF_VAL2, OTF_VAL3, OTF_VAL4, OTF_VAL5, OTF_VAL6,
                    OTF_VAL7, OTF_VAL8, OTF_VAL9, OTF_VAL10, OTF_VAL11, OTF_VAL12,
                    STOP_PLAN, WTD_LIMIT, BGIN_WL, QPF_YN
                )
                VALUES (
                    S.SIMM_ID, S.DAM_CD, S.ROM_TYPE, S.TARGET_WL, S.ROM_SIM,
                    S.OTF_VAL1, S.OTF_VAL2, S.OTF_VAL3, S.OTF_VAL4, S.OTF_VAL5, S.OTF_VAL6,
                    S.OTF_VAL7, S.OTF_VAL8, S.OTF_VAL9, S.OTF_VAL10, S.OTF_VAL11, S.OTF_VAL12,
                    S.STOP_PLAN, S.WTD_LIMIT, S.BGIN_WL, S.QPF_YN
                )
        """)

        try:
            if connection:
                for __row in __rows:
                    connection.execute(__query, __row)
            else:
                with self.engine.connect() as __conn:
                    with __conn.begin():
                        for __row in __rows:
                            __conn.execute(__query, __row)
            return f"TB_SIM_ST_INPUT_2에 제한수위 {len(__rows)}건을 등록/갱신했습니다."
        except Exception as e:
            logger.error(
                f"[Insert Error] TB_SIM_ST_INPUT_2 저장 중 오류가 발생했습니다: {str(e)}"
            )
            if connection:
                raise
            return "[시스템 알림] 제한수위 저장 중 내부 오류가 발생했습니다. 더 이상 도구를 찾지 말고 즉시 사용자에게 '모의운영 실행 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.'라고 답변하세요."

    def _insert_sim_st_if(
        self,
        user_id: str = Field(..., max_length=100, description="사용자 ID"),
        dam_cd: str = Field(
            ..., min_length=7, max_length=7, description="댐관리코드 (7자리 고정)"
        ),
        std_date: str = Field(
            ...,
            pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
            description="기준시간 (형식: YYYY-MM-DD HH:MM:SS)",
        ),
        simm_id: str = Field(..., max_length=100, description="SIMM ID"),
        ins_date: str = Field(
            ...,
            pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
            description="저장일시 (형식: YYYY-MM-DD HH:MM:SS)",
        ),
        recv_date: Optional[str] = Field(
            default=None, description="수신일시 (형식: YYYY-MM-DD HH:MM:SS)"
        ),
        start_date: Optional[str] = Field(
            default=None, description="모형시작 일시 (형식: YYYY-MM-DD HH:MM:SS)"
        ),
        end_date: Optional[str] = Field(
            default=None, description="모형종료 일시 (형식: YYYY-MM-DD HH:MM:SS)"
        ),
        proc_cd: Optional[str] = Field(default=None, description="분석코드"),
        proc_rslt: Optional[str] = Field(default=None, description="모형처리결과"),
        was_rslt: Optional[str] = Field(default=None, description="WAS처리결과"),
        start_date_diff: Optional[int] = Field(
            default=None, description="기준시간 대비 시뮬레이션 시작 시간 차이"
        ),
        end_date_diff: Optional[int] = Field(
            default=None, description="기준시간 대비 시뮬레이션 종료 시간 차이"
        ),
        batch_start: Optional[str] = Field(default=None, description="배치 시작"),
        batch_end: Optional[str] = Field(default=None, description="배치 종료"),
        mg_se_id: Optional[str] = Field(
            default=None, description="설정한 매개변수 CD_NO"
        ),
    ) -> str:
        """
        [DBSCHEMA].TB_SIM_ST_IF 테이블에 SIMM 모의운영 실행 요청 row를 등록하거나 갱신합니다.

        이 도구는 강우 시계열 데이터 자체를 저장하는 도구가 아닙니다.
        _get_simm_api_data로 조회하고 _apply_simm_rainfall_adjustment로 조정한 강우 데이터를
        _insert_sim_st_inputs로 TB_SIM_ST_INPUT_1과 TB_SIM_ST_INPUT_2에 저장한 뒤,
        해당 모의운영을 실행 대기/연계 대상으로 올리기 위한 인터페이스 상태 정보를 등록할 때만 호출합니다.

        동일한 SIMM_ID와 INS_DATE가 이미 있으면 기존 row를 갱신하고, 없으면 새 row를 생성합니다.

        호출 전 확인해야 할 사항:
        - 사용자가 강우량 증가/감소/설정 조건을 명시했거나, 기존 강우량 그대로 실행하겠다고 명확히 확인했습니다.
        - 필수 값(user_id, dam_cd, std_date, simm_id, ins_date)이 모두 확보되었습니다.
        - start_date와 end_date가 함께 주어진 경우 start_date가 end_date보다 늦지 않습니다.

        Args:
            user_id (str): 모의운영 요청 사용자 ID
            dam_cd (str): 대상 댐관리코드. 7자리 문자열
            std_date (str): 모의운영 기준시간. YYYY-MM-DD HH:MM:SS 형식
            simm_id (str): 모의운영 요청을 식별하는 SIMM ID
            ins_date (str): 인터페이스 row 저장일시. YYYY-MM-DD HH:MM:SS 형식
            recv_date (Optional[str]): 연계 수신일시
            start_date (Optional[str]): 모형 실행 시작일시
            end_date (Optional[str]): 모형 실행 종료일시
            proc_cd (Optional[str]): 분석/처리 상태 코드
            proc_rslt (Optional[str]): 모형 처리 결과 메시지 또는 상태값
            was_rslt (Optional[str]): WAS 처리 결과 메시지 또는 상태값
            start_date_diff (Optional[int]): 기준시간 대비 모형 시작 시간 차이
            end_date_diff (Optional[int]): 기준시간 대비 모형 종료 시간 차이
            batch_start (Optional[str]): 배치 시작 정보
            batch_end (Optional[str]): 배치 종료 정보
            mg_se_id (Optional[str]): 적용한 설정/매개변수 구분 CD_NO

        Returns:
            str: 인터페이스 row 등록/갱신 결과를 설명하는 시스템 메시지
        """
        ### --------------------------------------------------
        ### TB_SIM_ST_IF Insert 쿼리 실행
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- _insert_sim_st_if ----------")

        if hasattr(recv_date, "default"):
            recv_date = recv_date.default
        if hasattr(start_date, "default"):
            start_date = start_date.default
        if hasattr(end_date, "default"):
            end_date = end_date.default
        if hasattr(proc_cd, "default"):
            proc_cd = proc_cd.default
        if hasattr(proc_rslt, "default"):
            proc_rslt = proc_rslt.default
        if hasattr(was_rslt, "default"):
            was_rslt = was_rslt.default
        if hasattr(start_date_diff, "default"):
            start_date_diff = start_date_diff.default
        if hasattr(end_date_diff, "default"):
            end_date_diff = end_date_diff.default
        if hasattr(batch_start, "default"):
            batch_start = batch_start.default
        if hasattr(batch_end, "default"):
            batch_end = batch_end.default
        if hasattr(mg_se_id, "default"):
            mg_se_id = mg_se_id.default

        was_rslt = None
        mg_se_id = "SCH_1"

        def _validate_date_logic(date_str: str, field_name: str):
            if date_str:
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    raise ValueError(
                        f"'{field_name}'에 달력에 존재하지 않는 잘못된 날짜가 입력되었습니다: {date_str}"
                    )
            return None

        try:
            _dt_std = _validate_date_logic(std_date, "std_date")
            _dt_ins = _validate_date_logic(ins_date, "ins_date")
            _dt_recv = _validate_date_logic(recv_date, "recv_date")
            _dt_start = _validate_date_logic(start_date, "start_date")
            _dt_end = _validate_date_logic(end_date, "end_date")

            if _dt_start and _dt_end:
                if _dt_start > _dt_end:
                    return f"[검증 실패] 모형종료 일시({end_date})가 모형시작 일시({start_date})보다 빠를 수 없습니다."

            if start_date_diff is not None and end_date_diff is not None:
                if start_date_diff > end_date_diff:
                    return f"[검증 실패] START_DATE_DIFF({start_date_diff})가 END_DATE_DIFF({end_date_diff})보다 클 수 없습니다."

        except ValueError as ve:
            logger.error(
                f"[Value Error] 데이터 추가 중 검증 오류가 발생했습니다: {str(ve)}"
            )
            return f"[시스템 알림] 데이터 검증에 실패하였습니다. 더 이상 도구를 찾지 말고 즉시 사용자에게 '검증에 실패하였습니다. 확인 후 다시 입력해주세요.'라고 답변하세요."

        _query: TextClause = text("""
            MERGE INTO [DBSCHEMA].TB_SIM_ST_IF T
            USING (
                SELECT 
                    :user_id AS USER_ID,
                    :dam_cd AS DAM_CD,
                    TO_DATE(:std_date, 'YYYY-MM-DD HH24:MI:SS') AS STD_DATE,
                    :simm_id AS SIMM_ID,
                    TO_DATE(:recv_date, 'YYYY-MM-DD HH24:MI:SS') AS RECV_DATE,
                    TO_DATE(:start_date, 'YYYY-MM-DD HH24:MI:SS') AS START_DATE,
                    TO_DATE(:end_date, 'YYYY-MM-DD HH24:MI:SS') AS END_DATE,
                    :proc_cd AS PROC_CD,
                    :proc_rslt AS PROC_RSLT,
                    :was_rslt AS WAS_RSLT,
                    TO_DATE(:ins_date, 'YYYY-MM-DD HH24:MI:SS') AS INS_DATE,
                    :start_date_diff AS START_DATE_DIFF,
                    :end_date_diff AS END_DATE_DIFF,
                    :batch_start AS BATCH_START,
                    :batch_end AS BATCH_END,
                    :mg_se_id AS MG_SE_ID
                FROM DUAL
            ) S
            ON (
                T.SIMM_ID = S.SIMM_ID AND
                T.INS_DATE = S.INS_DATE
            )
            WHEN MATCHED THEN
                UPDATE SET 
                    T.RECV_DATE = S.RECV_DATE,
                    T.START_DATE = S.START_DATE,
                    T.END_DATE = S.END_DATE,
                    T.PROC_CD = S.PROC_CD,
                    T.PROC_RSLT = S.PROC_RSLT,
                    T.WAS_RSLT = S.WAS_RSLT,
                    T.START_DATE_DIFF = S.START_DATE_DIFF,
                    T.END_DATE_DIFF = S.END_DATE_DIFF,
                    T.BATCH_START = S.BATCH_START,
                    T.BATCH_END = S.BATCH_END,
                    T.MG_SE_ID = S.MG_SE_ID
            WHEN NOT MATCHED THEN
                INSERT (
                    USER_ID, DAM_CD, STD_DATE, SIMM_ID, RECV_DATE, 
                    START_DATE, END_DATE, PROC_CD, PROC_RSLT, WAS_RSLT, 
                    INS_DATE, START_DATE_DIFF, END_DATE_DIFF, BATCH_START, BATCH_END, MG_SE_ID
                ) 
                VALUES (
                    S.USER_ID, S.DAM_CD, S.STD_DATE, S.SIMM_ID, S.RECV_DATE, 
                    S.START_DATE, S.END_DATE, S.PROC_CD, S.PROC_RSLT, S.WAS_RSLT, 
                    S.INS_DATE, S.START_DATE_DIFF, S.END_DATE_DIFF, S.BATCH_START, S.BATCH_END, S.MG_SE_ID
                )
        """)

        try:
            with self.engine.connect() as _conn:
                with _conn.begin():
                    _conn.execute(
                        _query,
                        {
                            "user_id": user_id,
                            "dam_cd": dam_cd,
                            "std_date": std_date,
                            "simm_id": simm_id,
                            "recv_date": recv_date,
                            "start_date": start_date,
                            "end_date": end_date,
                            "proc_cd": proc_cd,
                            "proc_rslt": proc_rslt,
                            "was_rslt": was_rslt,
                            "ins_date": ins_date,
                            "start_date_diff": start_date_diff,
                            "end_date_diff": end_date_diff,
                            "batch_start": batch_start,
                            "batch_end": batch_end,
                            "mg_se_id": mg_se_id,
                        },
                    )

            _success_msg: str = (
                f"사용자 '{user_id}'의 시뮬레이션(SIMM ID: {simm_id}) 인터페이스 데이터가 성공적으로 추가되었습니다."
            )
            logger.debug(_success_msg)
            return _success_msg
        except Exception as e:
            logger.error(f"[Insert Error] 데이터 추가 중 오류가 발생했습니다: {str(e)}")
            return "[시스템 알림] 모의운영 실행 요청 등록 중 내부 오류가 발생했습니다. 더 이상 도구를 찾지 말고 즉시 사용자에게 '모의운영 실행 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.'라고 답변하세요."

    async def _emit_simm_status(
        self,
        event_emitter: Optional[Callable[[dict], Any]],
        description: str,
        done: bool = False,
    ) -> None:
        """
        모의운영 진행 상태를 Open-WebUI status 이벤트로 전달합니다.
        """
        if not event_emitter:
            return

        try:
            __result: Any = event_emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )
            if asyncio.iscoroutine(__result):
                await __result
        except Exception as e:
            logger.info(f"[SIMM Status Event Error] 상태 이벤트 전송 실패: {str(e)}")

    async def _emit_simm_replace(
        self,
        event_emitter: Optional[Callable[[dict], Any]],
        content: str,
    ) -> None:
        """
        모의운영 진행 문구를 Open-WebUI 응답 영역에 replace 이벤트로 전달합니다.
        """
        if not event_emitter:
            return

        try:
            __result: Any = event_emitter(
                {
                    "type": "replace",
                    "data": {"content": content},
                }
            )
            if asyncio.iscoroutine(__result):
                await __result
        except Exception as e:
            logger.info(f"[SIMM Replace Event Error] replace 이벤트 전송 실패: {str(e)}")

    def _get_simm_status(self, simm_id: str) -> dict[str, Any]:
        """
        [DBSCHEMA].TB_SIM_ST_IF에서 모의운영 처리 상태를 조회합니다.
        """
        __query: TextClause = text("""
            SELECT *
            FROM (
                SELECT
                    PROC_CD,
                    PROC_RSLT,
                    WAS_RSLT
                FROM [DBSCHEMA].TB_SIM_ST_IF
                WHERE SIMM_ID = :simm_id
                ORDER BY INS_DATE DESC
            )
            WHERE ROWNUM = 1
        """)

        try:
            with self.engine.connect() as __conn:
                __row = __conn.execute(__query, {"simm_id": simm_id}).fetchone()
            if not __row:
                return {
                    "state": "pending",
                    "proc_cd": None,
                    "proc_rslt": None,
                    "was_rslt": None,
                }

            __mapping = __row._mapping
            __proc_cd: Optional[str] = __mapping.get("proc_cd") or __mapping.get("PROC_CD")
            __proc_rslt: Any = __mapping.get("proc_rslt") or __mapping.get("PROC_RSLT")
            __was_rslt: Optional[str] = __mapping.get("was_rslt") or __mapping.get("WAS_RSLT")
            return {
                "state": "ok",
                "proc_cd": str(__proc_cd).strip() if __proc_cd is not None else None,
                "proc_rslt": __proc_rslt,
                "was_rslt": str(__was_rslt).strip() if __was_rslt is not None else None,
            }
        except Exception as e:
            logger.error(f"[SIMM Status Error] 모의운영 상태 조회 중 오류가 발생했습니다: {str(e)}")
            return {
                "state": "error",
                "proc_cd": None,
                "proc_rslt": None,
                "was_rslt": None,
            }

    def _is_simm_completed(self, status: dict[str, Any]) -> bool:
        """
        PROC_RSLT 100과 PROC_CD의 ZZ; 포함 여부를 모두 만족하면 완료로 판단합니다.
        """
        __proc_cd: str = str(status.get("proc_cd") or "")
        __proc_rslt: Any = status.get("proc_rslt")
        try:
            __proc_rslt_value: float = float(__proc_rslt)
        except (TypeError, ValueError):
            __proc_rslt_value = 0.0
        return __proc_rslt_value >= 100 and "ZZ;" in __proc_cd

    def _build_simm_replace_content(self, status: dict[str, Any]) -> str:
        """
        PROC_RSLT와 PROC_CD를 기준으로 응답 영역에 보여줄 진행 문구를 만듭니다.
        """
        __proc_cd: str = str(status.get("proc_cd") or "").strip()
        __proc_rslt: Any = status.get("proc_rslt")
        try:
            __proc_rslt_value: float = (
                float(__proc_rslt) if __proc_rslt is not None else 0.0
            )
        except (TypeError, ValueError):
            __proc_rslt_value = 0.0

        __dam_name_by_scd: dict[str, str] = {}
        for __dam in self.SIMM_DAM_ORDER:
            __dam_name: str = str(__dam["name"])
            if not __dam_name.endswith("댐"):
                __dam_name = f"{__dam_name}댐"
            __dam_name_by_scd[str(__dam["dam_scd"])] = __dam_name

        __proc_codes: list[str] = [
            __code.strip() for __code in __proc_cd.split(";") if __code.strip()
        ]
        __completed_dam_names: list[str] = []
        for __proc_code in __proc_codes:
            __dam_name: Optional[str] = __dam_name_by_scd.get(__proc_code)
            if __dam_name and __dam_name not in __completed_dam_names:
                __completed_dam_names.append(__dam_name)

        if __proc_rslt_value == 0 and __proc_cd and not __completed_dam_names:
            __messages: list[str] = [
                __message for __message in __proc_codes if __message != "ZZ"
            ]
            if __messages:
                return "\n".join(__messages)

        __progress_text: str = f"{__proc_rslt_value:g}"
        __lines: list[str] = [f"모의운영 실행 중... {__progress_text}%"]
        __lines.extend(
            f"{__dam_name} 완료" for __dam_name in __completed_dam_names
        )
        return "\n".join(__lines)

    async def _wait_simm_completion(
        self,
        simm_id: str,
        event_emitter: Optional[Callable[[dict], Any]] = None,
        timeout_seconds: int = 180,
        interval_seconds: int = 3,
    ) -> dict[str, Any]:
        """
        모의운영 인터페이스 처리 완료를 지정 시간 동안 대기합니다.
        """
        __loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        __started_at: float = __loop.time()
        __last_status: dict[str, Any] = {
            "state": "pending",
            "proc_cd": None,
            "proc_rslt": None,
            "was_rslt": None,
        }
        __running_status_emitted: bool = False
        await self._emit_simm_status(event_emitter, "모의운영 실행 대기 중...", done=False)

        while __loop.time() - __started_at <= timeout_seconds:
            __last_status = self._get_simm_status(simm_id)
            if __last_status.get("state") == "error":
                return {**__last_status, "done": False, "timed_out": False}

            if __last_status.get("was_rslt") == "C":
                await self._emit_simm_status(event_emitter, "모의운영 실행이 취소되었습니다.", done=False)
                return {**__last_status, "done": False, "cancelled": True, "timed_out": False}

            if not __running_status_emitted:
                await self._emit_simm_status(event_emitter, "모의운영 실행 중...", done=False)
                __running_status_emitted = True

            await self._emit_simm_replace(
                event_emitter,
                self._build_simm_replace_content(__last_status),
            )

            if self._is_simm_completed(__last_status):
                await self._emit_simm_status(event_emitter, "모의운영 실행 완료...", done=False)
                await self._emit_simm_replace(event_emitter, "")
                return {**__last_status, "done": True, "timed_out": False}

            await asyncio.sleep(interval_seconds)

        return {**__last_status, "done": False, "timed_out": True}

    def _get_simm_output_1_rows(self, simm_id: str) -> list[dict[str, Any]]:
        """
        [DBSCHEMA].TB_SIM_ST_OUTPUT_1에서 SIMM_ID 기준 전체 결과 row를 조회합니다.
        """
        __query: TextClause = text("""
            SELECT
                O.SIMM_ID,
                O.DAM_CD,
                O.TAG_CD,
                T.DSCR AS TAG_DSCR,
                O.TAG_VAL1,
                O.TAG_VAL2,
                O.TAG_VAL3,
                O.TAG_VAL4,
                O.TAG_VAL5,
                O.TAG_VAL6,
                O.TAG_VAL7,
                O.TAG_VAL8,
                O.TAG_VAL9,
                O.TAG_VAL10,
                O.TAG_VAL11,
                O.TAG_VAL12,
                O.INS_DATE,
                O.STD_DATE
            FROM
                [DBSCHEMA].TB_SIM_ST_OUTPUT_1 O
            LEFT JOIN
                [TAG_DBSCHEMA].IP_TAG T
            ON
                TRIM(O.TAG_CD) = TRIM(T.TAGNAME)
            WHERE
                O.SIMM_ID = :simm_id
            ORDER BY
                O.STD_DATE, O.DAM_CD, O.TAG_CD
        """)

        try:
            with self.engine.connect() as __conn:
                __rows = __conn.execute(__query, {"simm_id": simm_id}).fetchall()
            return [
                {str(__key).lower(): __value for __key, __value in __row._mapping.items()}
                for __row in __rows
            ]
        except Exception as e:
            logger.error(f"[SIMM Output Error] 모의운영 결과 조회 중 오류가 발생했습니다: {str(e)}")
            return []

    def _get_simm_output_2_rows(self, simm_id: str) -> list[dict[str, Any]]:
        """
        [DBSCHEMA].TB_SIM_ST_OUTPUT_2에서 SIMM_ID 기준 전체 결과 row를 조회합니다.

        현재 모의운영 응답과 차트 생성 흐름에서는 호출하지 않으며,
        추후 OUTPUT_2 전용 차트를 생성할 때 사용합니다.
        """
        __query: TextClause = text("""
            SELECT
                O.SIMM_ID,
                O.DAM_CD,
                O.TAG_CD,
                T.DSCR AS TAG_DSCR,
                O.TAG_VAL1,
                O.TAG_VAL2,
                O.TAG_VAL3,
                O.TAG_VAL4,
                O.TAG_VAL5,
                O.TAG_VAL6,
                O.TAG_VAL7,
                O.TAG_VAL8,
                O.TAG_VAL9,
                O.TAG_VAL10,
                O.TAG_VAL11,
                O.TAG_VAL12,
                O.INS_DATE,
                O.STD_DATE
            FROM
                [DBSCHEMA].TB_SIM_ST_OUTPUT_2 O
            LEFT JOIN
                [TAG_DBSCHEMA].IP_TAG T
            ON
                TRIM(O.TAG_CD) = TRIM(T.TAGNAME)
            WHERE
                O.SIMM_ID = :simm_id
            ORDER BY
                O.STD_DATE, O.DAM_CD, O.TAG_CD
        """)

        try:
            with self.engine.connect() as __conn:
                __rows = __conn.execute(__query, {"simm_id": simm_id}).fetchall()
            return [
                {str(__key).lower(): __value for __key, __value in __row._mapping.items()}
                for __row in __rows
            ]
        except Exception as e:
            logger.error(
                f"[SIMM Output Error] TB_SIM_ST_OUTPUT_2 조회 중 오류가 발생했습니다: {str(e)}"
            )
            return []

    def _parse_simm_chart_points(
        self,
        output_row: dict[str, Any],
        max_points: int = 1200,
    ) -> list[dict[str, Any]]:
        """
        TAG_VAL1~TAG_VAL12를 차트 표시용 시계열 포인트로 변환합니다.
        """
        __points: list[dict[str, Any]] = []
        for __index in range(1, 13):
            __raw_value: Optional[str] = output_row.get(f"tag_val{__index}")
            if not __raw_value:
                continue
            for __item in str(__raw_value).split(";"):
                __item = __item.strip()
                if not __item or "," not in __item:
                    continue
                __time_text, __value_text = __item.split(",", 1)
                try:
                    __points.append(
                        {
                            "time": __time_text.strip(),
                            "value": float(__value_text.strip()),
                        }
                    )
                except ValueError:
                    continue
                if len(__points) >= max_points:
                    return __points
        return __points

    def _get_simm_dam_chart_title(self, dam_cd: str) -> str:
        """
        차트 제목에 사용할 댐 한국어 명칭을 반환합니다.
        """
        __dam_name: str = next(
            (
                str(__dam["name"])
                for __dam in self.SIMM_DAM_ORDER
                if str(__dam["dam_cd"]) == str(dam_cd)
            ),
            str(dam_cd),
        )
        if __dam_name and not __dam_name.endswith("댐"):
            __dam_name = f"{__dam_name}댐"
        return f"{__dam_name}({dam_cd})"

    def _build_simm_chart_series(
        self,
        output_rows: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        OUTPUT_1 row를 DAM_CD별 다중 series 차트 데이터로 묶습니다.
        """
        __groups: dict[str, list[dict[str, Any]]] = {}
        for __row in output_rows:
            __points: list[dict[str, Any]] = self._parse_simm_chart_points(__row)
            if not __points:
                continue
            __dam_cd: str = str(__row.get("dam_cd") or "").strip()
            __tag_cd: str = str(__row.get("tag_cd") or "").strip()
            __tag_dscr: str = str(__row.get("tag_dscr") or "").split(",", 1)[
                0
            ].strip()
            __groups.setdefault(__dam_cd, []).append(
                {
                    "label": __tag_dscr or __tag_cd,
                    "points": __points,
                    "std_date": __row.get("std_date"),
                }
            )

        __dam_order_map: dict[str, int] = {
            str(__dam["dam_cd"]): int(__dam["order"]) for __dam in self.SIMM_DAM_ORDER
        }
        return dict(
            sorted(
                __groups.items(),
                key=lambda __item: (__dam_order_map.get(__item[0], 9999), __item[0]),
            )
        )

    def _flatten_simm_series_points(
        self,
        series_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        여러 series의 포인트를 차트 범위 계산용으로 평탄화합니다.
        """
        return [
            __point
            for __series in series_list
            for __point in __series.get("points", [])
        ]

    def _render_simm_matplotlib_chart(
        self,
        series_list: list[dict[str, Any]],
        title: str,
        image_format: str,
    ) -> bytes:
        """
        저장 없이 여러 series를 하나의 matplotlib 차트 이미지로 생성합니다.
        """
        __all_points: list[dict[str, Any]] = self._flatten_simm_series_points(series_list)
        if not __all_points:
            return b""

        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
        except Exception as e:
            logger.exception("[SIMM Chart Error] matplotlib 로드 실패: %s", e)
            return b""

        __korean_font_names: tuple[str, ...] = (
            "Malgun Gothic",
            "AppleGothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
            "NanumGothic",
        )
        __korean_font_paths: tuple[str, ...] = (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            "[FONT_PATH]/malgun.ttf",
            "[FONT_PATH]/malgunbd.ttf",
        )
        __available_font_names: set[str] = {
            __font.name for __font in font_manager.fontManager.ttflist
        }
        __title_font_properties: Optional[font_manager.FontProperties] = None
        for __font_name in __korean_font_names:
            if __font_name in __available_font_names:
                plt.rcParams["font.family"] = __font_name
                __title_font_properties = font_manager.FontProperties(
                    family=__font_name,
                    size=24,
                    weight="bold",
                )
                break
        if __title_font_properties is None:
            for __font_path in __korean_font_paths:
                if os.path.exists(__font_path):
                    font_manager.fontManager.addfont(__font_path)
                    __title_font_properties = font_manager.FontProperties(
                        fname=__font_path,
                        size=24,
                        weight="bold",
                    )
                    plt.rcParams["font.family"] = __title_font_properties.get_name()
                    break
        plt.rcParams["axes.unicode_minus"] = False
        __title_text: str = title
        if __title_font_properties is None:
            logger.warning("[SIMM Chart Warning] 한글 폰트를 찾지 못해 차트 제목을 댐코드로 대체합니다.")
            __title_match: Optional[re.Match] = re.search(r"\(([^)]+)\)", title)
            __title_text = __title_match.group(1) if __title_match else title

        def __parse_chart_time(value: Any) -> Optional[datetime]:
            __time_text: str = str(value or "").strip().replace("T", " ")
            for __format in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(__time_text[:19], __format)
                except ValueError:
                    continue
            return None

        __all_times: list[datetime] = sorted(
            {
                __time
                for __series in series_list
                for __point in __series.get("points", [])
                if (__time := __parse_chart_time(__point.get("time"))) is not None
            }
        )
        if not __all_times:
            return b""
        __reference_times: list[datetime] = [
            __reference_time
            for __series in series_list
            if (
                __reference_time := __parse_chart_time(
                    __series.get("std_date")
                )
            )
            is not None
        ]
        __reference_time: Optional[datetime] = (
            min(__reference_times) if __reference_times else None
        )
        __x_count: int = len(__all_times)
        __figure_height: float = max(5.2, min(9.6, 4.0 + (len(series_list) * 0.14)))

        __fig, __axis = plt.subplots(figsize=(12.0, __figure_height), dpi=200)
        __water_level_series: list[dict[str, Any]] = [
            __series
            for __series in series_list
            if "저수위" in str(__series.get("label") or "")
        ]
        __other_series: list[dict[str, Any]] = [
            __series
            for __series in series_list
            if "저수위" not in str(__series.get("label") or "")
        ]
        __right_axis = __axis.twinx() if __water_level_series else None
        __color_map = plt.get_cmap("tab20")
        __label_color_rules: tuple[tuple[str, str], ...] = (
            ("강우량", "#0000FF"),
            ("유입량", "#FF0000"),
            ("관측", "#FF0000"),
            ("보정", "#808080"),
            ("저수위", "#000000"),
            ("수위", "#FF00FF"),
            ("사용수량", "#F59E0B"),
            ("수문방류량", "#00A651"),
            ("총방류량", "#800080"),
        )

        if (
            __reference_time is not None
            and __reference_time < __all_times[-1]
        ):
            __axis.axvspan(
                max(__reference_time, __all_times[0]),
                __all_times[-1],
                facecolor="#F3F4F6",
                edgecolor="none",
                zorder=0,
            )

        for __series_index, __series in enumerate(series_list):
            __points: list[dict[str, Any]] = __series.get("points", [])
            if not __points:
                continue
            __time_value_pairs: list[tuple[datetime, float]] = []
            for __point in __points:
                __point_time: Optional[datetime] = __parse_chart_time(
                    __point.get("time")
                )
                if __point_time is None:
                    continue
                __time_value_pairs.append(
                    (__point_time, float(__point["value"]))
                )
            if not __time_value_pairs:
                continue
            __time_value_pairs.sort(key=lambda __pair: __pair[0])
            __x_values: list[datetime] = [
                __pair[0] for __pair in __time_value_pairs
            ]
            __y_values: list[float] = [
                __pair[1] for __pair in __time_value_pairs
            ]
            __label: str = str(__series.get("label") or "")
            __plot_axis = (
                __right_axis
                if __right_axis is not None and "저수위" in __label
                else __axis
            )
            __series_color: Any = next(
                (
                    __color
                    for __keyword, __color in __label_color_rules
                    if __keyword in __label
                ),
                __color_map(__series_index % 20),
            )
            __plot_axis.plot(
                __x_values,
                __y_values,
                linewidth=2.0,
                color=__series_color,
                label=__label,
                zorder=2,
            )

        __tick_count: int = min(8, __x_count)
        if __tick_count > 1:
            __tick_indexes: list[int] = sorted(
                {
                    round(__index * (__x_count - 1) / (__tick_count - 1))
                    for __index in range(__tick_count)
                }
            )
        else:
            __tick_indexes = [0]
        __tick_values: list[datetime] = [
            __all_times[__index] for __index in __tick_indexes
        ]
        __tick_labels: list[str] = []
        __date_label_positions: set[int] = {0, max(len(__tick_indexes) // 2, 0), len(__tick_indexes) - 1}
        for __label_index, __tick_time in enumerate(__tick_values):
            __time_label: str = __tick_time.strftime("%H:%M")
            if __label_index in __date_label_positions:
                __date_label: str = __tick_time.strftime("%Y-%m-%d")
                __tick_labels.append(f"{__time_label}\n{__date_label}")
            else:
                __tick_labels.append(__time_label)

        __axis.set_title(
            __title_text,
            fontsize=24,
            fontweight="bold",
            fontproperties=__title_font_properties,
            pad=18,
        )
        __axis.title.set_fontweight("bold")
        __axis.title.set_fontsize(24)
        __axis.set_xlabel("")
        __axis.set_ylabel("[m³/s]", rotation=0, fontsize=10, labelpad=18)
        if len(__all_times) == 1:
            __axis.set_xlim(
                __all_times[0] - timedelta(minutes=5),
                __all_times[0] + timedelta(minutes=5),
            )
        else:
            __axis.set_xlim(__all_times[0], __all_times[-1])
        __axis.set_xticks(__tick_values)
        __axis.set_xticklabels(__tick_labels, rotation=0, ha="center", fontsize=9)
        __axis.tick_params(axis="y", labelsize=9)
        if not __other_series:
            __axis.tick_params(axis="y", left=False, labelleft=False)
            __axis.set_ylabel("")
        if __right_axis is not None:
            __right_axis.set_ylabel("[El.m]", rotation=0, fontsize=10, labelpad=20)
            __right_axis.tick_params(axis="y", labelsize=9)
        __axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        __axis.margins(x=0.01)
        __legend_handles, __legend_labels = __axis.get_legend_handles_labels()
        if __right_axis is not None:
            __right_handles, __right_labels = __right_axis.get_legend_handles_labels()
            __legend_handles.extend(__right_handles)
            __legend_labels.extend(__right_labels)
        __axis.legend(
            __legend_handles,
            __legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.10, 1.0),
            fontsize=9,
            frameon=False,
        )
        __fig.tight_layout()

        __buffer: BytesIO = BytesIO()
        __fig.savefig(__buffer, format=image_format, dpi=200, bbox_inches="tight")
        plt.close(__fig)
        return __buffer.getvalue()

    def _build_simm_chart_preview(self, output_rows: list[dict[str, Any]]) -> str:
        """
        저장 없이 전체 OUTPUT_1 row의 SVG/PNG(base64) 차트 미리보기를 생성합니다.
        """
        if not output_rows:
            return ""
        __groups: dict[str, list[dict[str, Any]]] = self._build_simm_chart_series(output_rows)
        __sections: list[str] = []
        __chart_format: str = str(
            getattr(getattr(self, "valves", None), "SIMM_CHART_FORMAT", "PNG") or "PNG"
        ).strip().upper()
        if __chart_format not in {"PNG", "SVG"}:
            __chart_format = "PNG"

        for __dam_cd, __series_list in __groups.items():
            __title: str = self._get_simm_dam_chart_title(__dam_cd)
            __chart_parts: list[str] = []

            if __chart_format == "SVG":
                __svg_bytes: bytes = self._render_simm_matplotlib_chart(
                    __series_list,
                    __title,
                    "svg",
                )
                if __svg_bytes:
                    __svg_base64: str = base64.b64encode(__svg_bytes).decode("ascii")
                    __chart_parts.append(
                        f"![SIMM 결과 SVG {__dam_cd}](data:image/svg+xml;base64,{__svg_base64})"
                    )

            if __chart_format == "PNG":
                __png_bytes: bytes = self._render_simm_matplotlib_chart(
                    __series_list,
                    __title,
                    "png",
                )
                if __png_bytes:
                    __png_base64: str = base64.b64encode(__png_bytes).decode("ascii")
                    __chart_parts.append(
                        f"![SIMM 결과 PNG {__dam_cd}](data:image/png;base64,{__png_base64})"
                    )

            if not __chart_parts:
                continue
            __sections.append("\n\n".join(__chart_parts))
        return "\n\n".join(__sections)

    async def _execute_simm_simulation(
        self,
        __parsed_request: dict[str, Any],
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None,
        event_emitter: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        파싱된 SIMM 요청을 실제 API 조회, 강우 조정, DB 저장, 실행 요청 등록 순서로 처리합니다.
        """
        __loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        __execution_started_at: float = __loop.time()
        __ins_date: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        __selected_dams: list[dict[str, Any]] = __parsed_request.get("dams", [])
        __target_mask: str = self._build_simm_target_mask(__selected_dams)
        #TODO: 추후 변경(테스트용)
        #__user_id: str = user_id or f"ai_{self.SIMM_USER_ID}"
        __user_id: str = "[USER_ID]"
        __user_ip: str = user_ip or self.SIMM_USER_IP
        __simm_id: str = self._build_simm_id(
            __parsed_request,
            __user_id,
            __user_ip,
        )
        __start_dam: Optional[dict[str, Any]] = __parsed_request.get("start_dam")
        __representative_dam_cd: str = (
            __start_dam["dam_cd"]
            if __start_dam
            else (__selected_dams[0]["dam_cd"] if __selected_dams else "")
        )
        __tag_list: str = ",".join(__parsed_request.get("tag_list", []))
        __tag_dam_map: str = json.dumps(
            __parsed_request.get("tag_dam_map", {}), ensure_ascii=False
        )
        logger.debug(
            "[SIMM 실행 입력]: "
            + json.dumps(
                {
                    "start": __parsed_request.get("start"),
                    "end": __parsed_request.get("end"),
                    "std_date": __parsed_request.get("std_date"),
                    "operation": __parsed_request.get("operation"),
                    "amount": __parsed_request.get("amount"),
                    "amount_type": __parsed_request.get("amount_type"),
                    "tag_list": __tag_list,
                    "tag_dam_map": __parsed_request.get("tag_dam_map", {}),
                    "otf_adjustment": __parsed_request.get("otf_adjustment"),
                    "rainfall_adjustment": __parsed_request.get("rainfall_adjustment"),
                    "target_mask": __target_mask,
                    "simm_id": __simm_id,
                    "ins_date": __ins_date,
                    "user_id": __user_id,
                    "user_ip": __user_ip,
                    "dam_cd": __representative_dam_cd,
                    "ai_type": "TM",
                },
                ensure_ascii=False,
            )
        )

        __api_data: list[Any] = self._get_simm_api_data(
            start=__parsed_request.get("start"),
            end=__parsed_request.get("end"),
            tag_list=__tag_list,
        )
        __adjusted_data: dict = self._apply_simm_rainfall_adjustment(
            api_data=__api_data,
            operation=__parsed_request.get("operation"),
            amount=__parsed_request.get("amount"),
            amount_type=__parsed_request.get("amount_type"),
        )
        if int(__adjusted_data.get("changed_count", 0)) == 0:
            return "[시스템 알림] 조정된 강우량 값이 없습니다. 더 이상 도구를 찾지 말고 사용자에게 '조정할 강우량 값이 없어 모의운영 실행을 중단했습니다.'라고 답변하세요."

        __input_result: str = self._insert_sim_st_inputs(
            adjusted_api_data=__adjusted_data,
            simm_id=__simm_id,
            ins_date=__ins_date,
            tag_dam_map=__tag_dam_map,
            start_date=__parsed_request.get("start"),
            otf_adjustment=bool(__parsed_request.get("otf_adjustment")),
            rainfall_adjustment=bool(__parsed_request.get("rainfall_adjustment", True)),
            ai_type="TM",
        )
        if "오류" in __input_result or "실패" in __input_result:
            return __input_result

        __start_dt: datetime = datetime.strptime(
            __parsed_request.get("start"), "%Y-%m-%d %H:%M:%S"
        )
        __end_dt: datetime = datetime.strptime(
            __parsed_request.get("end"), "%Y-%m-%d %H:%M:%S"
        )
        __date_diff_hours: int = int((__end_dt - __start_dt).total_seconds() // 3600)

        __if_result: str = self._insert_sim_st_if(
            user_id=__user_id,
            dam_cd=__representative_dam_cd,
            std_date=__parsed_request.get("std_date"),
            simm_id=__simm_id,
            ins_date=__ins_date,
            start_date=__parsed_request.get("start"),
            end_date=__parsed_request.get("end"),
            was_rslt=None,
            start_date_diff=-__date_diff_hours,
            end_date_diff=__date_diff_hours,
            mg_se_id="SCH_1",
        )
        if "오류" in __if_result or "실패" in __if_result:
            return __if_result

        __elapsed_seconds: float = __loop.time() - __execution_started_at
        __remaining_timeout_seconds: int = max(
            0,
            int(self.SIMM_EXECUTION_TIMEOUT_SECONDS - __elapsed_seconds),
        )
        __completion_status: dict[str, Any] = await self._wait_simm_completion(
            simm_id=__simm_id,
            event_emitter=event_emitter,
            timeout_seconds=__remaining_timeout_seconds,
        )

        if __completion_status.get("cancelled"):
            await self._emit_simm_status(event_emitter, "", done=True)
            return f"모의운영 실행이 취소되었습니다. SIMM ID는 {__simm_id}입니다."

        if __completion_status.get("state") == "error":
            await self._emit_simm_status(event_emitter, "", done=True)
            return (
                "모의운영 실행 상태를 확인하는 중 오류가 발생했습니다. "
                f"SIMM ID는 {__simm_id}입니다."
            )

        if __completion_status.get("timed_out"):
            await self._emit_simm_status(event_emitter, "모의운영 결과 조회 중...", done=False)
            __timeout_output_rows: list[dict[str, Any]] = self._get_simm_output_1_rows(__simm_id)
            if not __timeout_output_rows:
                logger.info(
                    f"[SIMM Timeout] 180초 내 완료되지 않았고 OUTPUT_1 조회 결과가 없습니다. SIMM_ID={__simm_id}"
                )
                await self._emit_simm_status(event_emitter, "", done=True)
                return (
                    "모의운영이 아직 진행 중이며, 현재 조회 가능한 결과 데이터가 없어 차트를 생성하지 못했습니다. "
                    f"SIMM ID는 {__simm_id}입니다."
                )
            __timeout_chart_preview: str = self._build_simm_chart_preview(
                __timeout_output_rows
            )
            return (
                "모의운영 완료 상태는 아직 확인되지 않았지만 결과 데이터가 조회되었습니다. "
                f"SIMM ID는 {__simm_id}입니다. "
                f"결과 데이터 {len(__timeout_output_rows)}건을 조회했습니다. "
                f"{__timeout_chart_preview}"
            )

        await self._emit_simm_status(event_emitter, "모의운영 결과 조회 중...", done=False)
        __output_rows: list[dict[str, Any]] = self._get_simm_output_1_rows(__simm_id)
        if not __output_rows:
            return (
                "모의운영은 완료되었지만 조회 가능한 결과 데이터가 없습니다. "
                f"SIMM ID는 {__simm_id}입니다."
            )

        __chart_preview: str = self._build_simm_chart_preview(__output_rows)
        return (
            "모의운영이 완료되었습니다. "
            f"SIMM ID는 {__simm_id}입니다. "
            f"결과 데이터 {len(__output_rows)}건을 조회했습니다. "
            f"{__chart_preview}"
        )

    async def run_simm_simulation(
        self,
        user_query: str = Field(
            ..., description="모의운영 실행을 요청한 사용자 원문 질의"
        ),
    ) -> str:
        """
        사용자 자연어 요청을 받아 SIMM 모의운영을 실행합니다.

        날짜, 시작 시간, 댐 구간, 강우량 증가/감소/설정 조건을 검증한 뒤
        강우 API 조회, 강우량 조정, TB_SIM_ST_INPUT_1 저장, TB_SIM_ST_IF 실행 요청 등록을
        한 번에 처리합니다.
        """
        logger.debug("[Call]: ---------- run_simm_simulation ----------")
        if not self._has_simm_adjustment_condition(user_query):
            return (
                "[시스템 알림] SIMM 모의운영 실행 조건 확인이 필요합니다. "
                "더 이상 도구를 찾지 말고 즉시 사용자에게 "
                "'기존 강우량 데이터를 그대로 사용하여 모의운영을 실행할까요? "
                "아니면 강우량 증가/감소/설정 조건을 추가하시겠습니까?'라고 답변하세요."
            )

        parsed_request: dict[str, Any] = self._parse_simm_user_request(user_query)
        if parsed_request.get("missing"):
            return self._build_simm_missing_message(parsed_request)
        __context_user: Optional[dict] = user_var.get()
        __user_id: str = self._extract_context_user_id(__context_user)
        __user_ip: str = self._get_local_ip()
        __event_emitter: Optional[Callable[[dict], Any]] = event_emitter_var.get()
        return await self._execute_simm_simulation(
            parsed_request,
            user_id=__user_id,
            user_ip=__user_ip,
            event_emitter=__event_emitter,
        )

    async def simm_agent(
        self,
        user_message: str,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        body_context: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        SIMM 모의운영 Agent
        """
        logger.debug(f"[Start]: ---------- {self.name} ----------")
        user_var.set(__user__ or {})
        simm_tools_retriever: ObjectRetriever = self.obj_index.as_retriever(
            similarity_top_k=self.valves.TOP_K_TOOLS
        )
        system_prompt: str = """
        당신은 SIMM 모의운영 요청을 처리하는 도구 사용 전문가입니다.
        사용자가 시뮬레이션, 모의운영, SIMM 실행을 요청하면 run_simm_simulation 도구를 호출하세요.
        필요한 값이 부족한 경우에도 직접 추측하지 말고 선택한 도구의 결과를 그대로 따르세요.
        도구 Observation에 없는 영향, 예측, 해석, 원인 추정은 덧붙이지 마세요.
        Observation이 "[시스템 알림]"으로 시작하면 다른 도구를 호출하지 말고, 알림에 적힌 사용자 응답 문구만 Final Answer로 답하세요.
        도구 결과에 SVG/PNG Markdown 이미지가 포함되어 있으면 삭제하거나 요약하지 말고 그대로 Final Answer에 포함하세요.
        차트 마크업이 없는 최종 답변은 한국어 평문 한 문장으로 작성하고, 마크다운 문법을 사용하지 마세요.
        """

        try:
            agent: ReActAgent = ReActAgent(
                llm=self.llm,
                tools=[],
                tool_retriever=simm_tools_retriever,
                system_prompt=system_prompt,
                verbose=False,
            )
            handler: WorkflowHandler = agent.run(
                user_msg=user_message, max_iterations=4
            )

            _is_streaming_final: bool = False
            _stream_buffer: str = ""
            _final_answer_buffer: str = ""

            async for __event in handler.stream_events():
                __event_name: str = type(__event).__name__
                if isinstance(__event, AgentStream):
                    if __event.delta:
                        if _is_streaming_final:
                            _final_answer_buffer += __event.delta
                        else:
                            _stream_buffer += __event.delta
                            __keyword: str = ""
                            if "Final Answer:" in _stream_buffer:
                                __keyword = "Final Answer:"
                            elif "Answer:" in _stream_buffer:
                                __keyword = "Answer:"
                            if __keyword:
                                _is_streaming_final = True
                                __split_text: str = _stream_buffer.split(__keyword, 1)[
                                    1
                                ]
                                if __split_text:
                                    _final_answer_buffer += __split_text
                    continue

                if hasattr(__event, "response"):
                    __content: str = ""
                    __event_response: ChatMessage = __event.response
                    if hasattr(__event_response, "response") and isinstance(
                        __event_response.response, str
                    ):
                        __content = __event_response.response
                    elif hasattr(__event_response, "message") and hasattr(
                        __event_response.message, "content"
                    ):
                        __content = __event_response.message.content
                    elif hasattr(__event_response, "content"):
                        __content = __event_response.content
                    else:
                        __content = str(__event_response)
                    if __content:
                        __thought: str = str(__content).strip()
                        if __thought:
                            logger.info(
                                f"\n{Colors.YELLOW}[SIMM Agent Thought/Log]:\n{__thought}{Colors.RESET}\n"
                            )
                    if not _is_streaming_final:
                        _stream_buffer = ""
                elif __event_name == "ToolCallResult":
                    if hasattr(__event, "tool_call"):
                        __tool_name: str = getattr(
                            __event.tool_call, "tool_name", "Unknown"
                        )
                        __tool_kwargs: dict = getattr(
                            __event.tool_call, "tool_kwargs", {}
                        )
                        logger.info(f"{Colors.CYAN}Action: {__tool_name}{Colors.RESET}")
                        logger.info(
                            f"{Colors.CYAN}Action Input: {__tool_kwargs}{Colors.RESET}"
                        )
                    if hasattr(__event, "tool_output"):
                        __tool_output_text: str = str(__event.tool_output)
                        logger.info(
                            f"{Colors.CYAN}Observation: {__tool_output_text[:500]}{Colors.RESET}"
                        )
                        if _contains_simm_chart_markup(__tool_output_text):
                            logger.info(
                                f"\n{Colors.GREEN}[SIMM Agent Direct Tool Output]:\n{__tool_output_text[:500]}{Colors.RESET}\n"
                            )
                            logger.debug(f"[End]: ---------- {self.name} ----------")
                            return __tool_output_text

            _result: AgentOutput = await handler
            if _final_answer_buffer:
                logger.info(
                    f"\n{Colors.GREEN}[SIMM Agent Final Answer]:\n{_final_answer_buffer.strip()}{Colors.RESET}\n"
                )

            logger.debug(f"[End]: ---------- {self.name} ----------")
            if hasattr(_result, "response"):
                return str(_result.response)
            return str(_result)
        except Exception as e:
            logger.error(
                f"[Agent Error] SIMM 에이전트 처리 중 오류가 발생했습니다: {str(e)}"
            )
            return "모의운영 실행 요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
