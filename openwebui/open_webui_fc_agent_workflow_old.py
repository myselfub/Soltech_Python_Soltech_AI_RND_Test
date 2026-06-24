"""
title: PipeTest
author: Soltech
author_url:
version: 1.0.0
icon_url:
required_open_webui_version: 0.9.0
requirements: llama-index-core==0.12.52.post1, llama-index-embeddings-ollama==0.4.0, llama-index-llms-ollama==0.4.2, oracledb==3.4.2
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import oracledb
import requests
import sqlalchemy
import sys
from datetime import datetime
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

event_emitter_var: contextvars.ContextVar[Callable[[dict], Any] | None] = (
    contextvars.ContextVar("event_emitter_var", default=None)
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
        SIMM_API_URL: str = Field(default="http://[SIMM_APIIP]:8084", description="SIMM API URL")

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
            __event_emitter__: Callable[[dict], Any] = event_emitter_var.get()
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "데이터베이스 조회중...", "done": False},
                }
            )

            _response: str = await self.db_tools.db_agent(user_message=user_query)

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
            __event_emitter__: Callable[[dict], Any] = event_emitter_var.get()
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "API 조회중...", "done": False},
                }
            )

            _response: str = await self.api_tools.api_agent(user_message=user_query)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "API 조회완료...", "done": True},
                }
            )

            return _response

        return [
            FunctionTool.from_defaults(async_fn=db_agent_tools),
            FunctionTool.from_defaults(async_fn=api_agent_tools),
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
        __event_emitter__: Callable[[dict], Any],
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
                    logger.info(
                        f"{Colors.BLUE}Observation: {str(__event.tool_output)[:500]}{Colors.RESET}"
                    )

        await _handler

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "응답생성중...", "done": False},
            }
        )

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
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

        if __event_emitter__:
            event_emitter_var.set(__event_emitter__)

        _system_prompt: str = f"""
        당신은 엄격한 도구 사용 전문가다.
        1. 질문을 받으면, 절대로 스스로 추측하지 말고 반드시 필요한 도구를 호출해라.
        2. 사용자의 질문에 과제가 2개 이상 있다면, 모든 과제에 대해 각각 도구를 실행한 후에만 'Final Answer'를 작성할 수 있다.
        3. 필요한 파라메터가 부족한 경우에는 임의의 값을 스스로 만들어 도구를 호출하지 말고, 사용자에게 필요한 정보를 요청해라.
        4. 알맞는 정보나 도구가 없다면 "지원하지 않는 기능입니다."라고만 답해라.
        5. 반드시 plain text만 사용해라. 마크다운 코드블럭을 사용하지 마라.
        """

        try:
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
            )

            async for __chunk in _response:
                yield __chunk

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "", "done": True},
                }
            )
        except Exception as e:
            logger.error(f"[Agent Error] 오케스트레이션 에이전트 처리 중 오류가 발생했습니다: {str(e)}")
            yield f"오케스트레이션 에이전트 처리 중 오류가 발생했습니다: {str(e)}"


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
        logger.debug(f'----- get_table_comments_in_table -----')
        _schema_info: str = ''
        with self.engine.connect() as _conn:
            _query: TextClause = text('''
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
                          ''').bindparams(bindparam('tables', expanding=True))

            _tables_upper_list: list = self._get_upper_tables_list()

            _result: CursorResult = _conn.execute(_query, {
                'tables': _tables_upper_list
            })

            _current_table: str = ''
            for __table_name, __table_descr, __column_name, __data_type, __col_descrt in _result.fetchall():
                if __table_name != _current_table:
                    _current_table: str = __table_name
                    __table_desc: str = f' ({__table_descr})' if __table_descr else ''
                    _schema_info += f'\n### TABLE: {__table_name}{__table_desc}\n'

                __col_desc: str = f' - {__col_descrt}' if __col_descrt else ''
                _schema_info += f'  * {__column_name} [{__data_type}]{__col_desc}\n'

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

        _schema_info: str = ''
        with self.engine.connect() as _conn:
            _query = text('''
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
                              ''').bindparams(bindparam('tables', expanding=True))

            _tables_upper_list: list = self._get_upper_tables_list()

            _result: CursorResult = _conn.execute(_query, {
                'schema': schema,
                'tables': _tables_upper_list
            })

            _current_table: str = ''
            for __row in _result:
                if __row.table_name != _current_table:
                    _current_table = __row.table_name
                    table_desc: str = f' ({__row.table_comment})' if __row.table_comment else ''
                    _schema_info += f"\n### TABLE: {__row.table_name}{table_desc}\n"

                __column_desc = f' - {__row.column_comment}' if __row.column_comment else ''
                _schema_info += f'  * {__row.column_name} [{__row.data_type}]{__column_desc}\n'

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
            return f"프로시저(Cursor) 실행 중 오류가 발생했습니다: {str(e)}"
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

    async def db_agent(self, user_message: str) -> str:
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
            logger.error(f"[Agent Error] 데이터베이스 에이전트 처리 중 오류가 발생했습니다: {str(e)}")
            return f"데이터베이스 에이전트 처리 중 오류가 발생했습니다: {str(e)}"


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
        __connection_url: str = self.valves.API_URL + '/req-tag'
        __headers: dict = {"Content-Type": "application/json"}
        __response: requests.Response = requests.post(
            __connection_url, json=params, headers=__headers
        )
        if not (200 <= __response.status_code < 300):
            logger.error(f"[API Error] API 호출 중 오류가 발생했습니다: {__response.status_code}")
            raise Exception(f"API 호출 중 오류가 발생했습니다: {__response.text}({__response.status_code})")

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
            logger.error(f"[API Error] 태그 데이터 API 조회 중 오류가 발생했습니다: {str(e)}")
            return f"태그 데이터 API 조회 중 오류가 발생했습니다: {str(e)}"

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
            logger.error(f"[API Error] API 데이터 호출 중 오류가 발생했습니다: {str(e)}")
            return f"API 데이터 호출 중 오류가 발생했습니다: {str(e)}"

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

    async def api_agent(self, user_message: str) -> str:
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
            logger.error(f"[API Error] API 에이전트 처리 중 오류가 발생했습니다: {str(e)}")
            return f"API 에이전트 처리 중 오류가 발생했습니다: {str(e)}"


class SIMMAgent:
    valves: "Valves"
    engine: sqlalchemy.engine.base.Engine = None
    llm: Ollama = None

    class Valves(BaseModel):
        """
            밸브 설정
        """
        SIMM_DB_HOST: str = None
        SIMM_DB_PORT: str = None
        SIMM_DB_DATABASE: str = None
        SIMM_DB_USER: str = None
        SIMM_DB_PASSWORD: str = None
        SIMM_DB_SCHEMA: str = None
        SIMM_DB_TABLES: str = None
        SIMM_API_URL: str = None

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
        _db_tools: list = [
            FunctionTool.from_defaults(fn=self.get_table_comments_in_database),
            FunctionTool.from_defaults(fn=self.get_simm_api_data),
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
        return [t.strip().upper() for t in self.valves.SIMM_DB_TABLES.split(",")]

    def _get_lower_tables_list(self) -> list:
        """
        Lower 테이블명 목록
        """
        return [t.strip().lower() for t in self.valves.SIMM_DB_TABLES.split(",")]

    def _init_db_connection(self) -> sqlalchemy.engine.base.Engine:
        """
        Database 연결
        Oracle + oracledb 드라이버 사용 설정 (형식: oracle+oracledb://user:pass@host:port/?service_name=db)
        """
        _connection_url: str = (
            f"oracle+oracledb://{self.valves.SIMM_DB_USER}:{self.valves.SIMM_DB_PASSWORD}@"
            f"{self.valves.SIMM_DB_HOST}:{self.valves.SIMM_DB_PORT}/"
            f"?service_name={self.valves.SIMM_DB_DATABASE}"
        )
        _engine: sqlalchemy.engine.base.Engine = create_engine(
            _connection_url
        )  # , echo=True)

        return _engine

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
        schema: str = self.valves.SIMM_DB_SCHEMA.strip().upper()

        _schema_info: str = ''
        with self.engine.connect() as _conn:
            _query = text('''
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
                              ''').bindparams(bindparam('tables', expanding=True))

            _tables_upper_list: list = self._get_upper_tables_list()

            _result: CursorResult = _conn.execute(_query, {
                'schema': schema,
                'tables': _tables_upper_list
            })

            _current_table: str = ''
            for __row in _result:
                if __row.table_name != _current_table:
                    _current_table = __row.table_name
                    table_desc: str = f' ({__row.table_comment})' if __row.table_comment else ''
                    _schema_info += f"\n### TABLE: {__row.table_name}{table_desc}\n"

                __column_desc = f' - {__row.column_comment}' if __row.column_comment else ''
                _schema_info += f'  * {__row.column_name} [{__row.data_type}]{__column_desc}\n'

        return _schema_info

    def get_simm_api_data(
            self,
            start: str = Field(..., description="조회 시작 일시 (형식: 'YYYY-MM-DD HH:MM:SS')"),
            end: str = Field(..., description="조회 종료 일시 (형식: 'YYYY-MM-DD HH:MM:SS')"),
            tag_list: str = Field(..., description="조회할 태그 리스트 (쉼표로 구분된 문자열. 예: 'D1009710FCPCP112,D1009710FCPCP113')"),
            ip: str = Field(default="[CLIENTIP]", description="요청 IP 주소"),
            channel: str = Field(default="/queue/phdWeb", description="요청 채널")
    ) -> dict:
        """
        강우 및 트렌드 데이터를 조회하기 위해 API를 호출합니다.
        사용자가 특정 기간과 태그(센서)의 데이터를 요청할 때 사용합니다.
        """
        ### --------------------------------------------------
        ### 강우 데이터 조회
        ### --------------------------------------------------
        logger.debug(f"[Start]: ---------- get_simm_api_data ----------")

        _tags = [
            t.strip().strip("\"'")
            for t in tag_list.replace("[", "").replace("]", "").split(",")
            if t.strip()
        ]

        _json = {
            "cmd": "fetch",
            "param": {
                "start": start,
                "end": end,
                "function": "getTrend",
                "tagList": _tags
            }
        }

        _form_data = {
            "json": json.dumps(_json),
            "ip": ip,
            "channel": channel
        }

        _connection_url: str = self.valves.SIMM_API_URL + '/KHNP/EA/EaiRequest_get'
        _headers: dict = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
        }

        _response: requests.Response = requests.post(
            _connection_url,
            data=_form_data,
            headers=_headers
        )

        if not (200 <= _response.status_code < 300):
            logger.error(f"[API Error] API 호출 중 오류가 발생했습니다: {_response.status_code}")
            raise Exception(f"API 호출 중 오류가 발생했습니다: {_response.text}({_response.status_code})")

        _result = _response.json()
        logger.debug(f"[API Result]: {_result}")

        return _result

    def insert_sim_st_if(
            self,
            user_id: str = Field(..., max_length=100, description="사용자 ID"),
            dam_cd: str = Field(..., min_length=7, max_length=7, description="댐관리코드 (7자리 고정)"),
            std_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
                                  description="기준시간 (형식: YYYY-MM-DD HH:MM:SS)"),
            simm_id: str = Field(..., max_length=100, description="SIMM ID"),
            ins_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
                                  description="저장일시 (형식: YYYY-MM-DD HH:MM:SS)"),
            recv_date: Optional[str] = Field(default=None, description="수신일시 (형식: YYYY-MM-DD HH:MM:SS)"),
            start_date: Optional[str] = Field(default=None, description="모형시작 일시 (형식: YYYY-MM-DD HH:MM:SS)"),
            end_date: Optional[str] = Field(default=None, description="모형종료 일시 (형식: YYYY-MM-DD HH:MM:SS)"),
            proc_cd: Optional[str] = Field(default=None, description="분석코드"),
            proc_rslt: Optional[str] = Field(default=None, description="모형처리결과"),
            was_rslt: Optional[str] = Field(default=None, description="WAS처리결과"),
            start_date_diff: Optional[int] = Field(default=None, description="기준시간 대비 시뮬레이션 시작 시간 차이"),
            end_date_diff: Optional[int] = Field(default=None, description="기준시간 대비 시뮬레이션 종료 시간 차이"),
            batch_start: Optional[str] = Field(default=None, description="배치 시작"),
            batch_end: Optional[str] = Field(default=None, description="배치 종료"),
            mg_se_id: Optional[str] = Field(default=None, description="설정한 매개변수 CD_NO")
    ) -> str:
        """
        사용자가 [DBSCHEMA].TB_SIM_ST_IF 테이블에 시뮬레이션 상태 및 인터페이스 데이터를 추가(Insert)해 달라고 요청할 때 호출합니다.

        Args:
            user_id (str): 사용자 ID
            dam_cd (str): 댐관리코드
            std_date (str): 기준시간
            simm_id (str): SIMM ID
            ins_date (str): 저장일시
            recv_date (Optional[str]): 수신일시
            start_date (Optional[str]): 모형시작 일시
            end_date (Optional[str]): 모형종료 일시
            proc_cd (Optional[str]): 분석코드
            proc_rslt (Optional[str]): 모형처리결과
            was_rslt (Optional[str]): WAS처리결과
            start_date_diff (Optional[int]): 기준시간 대비 시뮬레이션 시작 시간
            end_date_diff (Optional[int]): 기준시간 대비 시뮬레이션 종료 시간
            batch_start (Optional[str]): 배치 시작
            batch_end (Optional[str]): 배치 종료
            mg_se_id (Optional[str]): 설정한 매개변수 CD_NO

        Returns:
            str: 데이터 추가 성공 여부를 나타내는 시스템 메시지
        """
        ### --------------------------------------------------
        ### TB_SIM_ST_IF Insert 쿼리 실행
        ### --------------------------------------------------
        logger.debug(f"[Call]: ---------- insert_sim_st_if ----------")

        def _validate_date_logic(date_str: str, field_name: str):
            if date_str:
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    raise ValueError(f"'{field_name}'에 달력에 존재하지 않는 잘못된 날짜가 입력되었습니다: {date_str}")
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
            logger.error(f"[Value Error] 데이터 추가 중 검증 오류가 발생했습니다: {str(ve)}")
            return f"[시스템 알림] 데이터 검증에 실패하였습니다. 더 이상 도구를 찾지 말고 즉시 사용자에게 '검증에 실패하였습니다. 확인 후 다시 입력해주세요.'라고 답변하세요."

        _query = text("""
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
                T.USER_ID = S.USER_ID AND 
                T.DAM_CD = S.DAM_CD AND 
                T.STD_DATE = S.STD_DATE AND 
                T.INS_DATE = S.INS_DATE
            )
            WHEN MATCHED THEN
                UPDATE SET 
                    T.SIMM_ID = S.SIMM_ID,
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
        logger.info(type(_query))

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
                            "mg_se_id": mg_se_id
                        }
                    )

            _success_msg: str = f"[시스템 알림] 사용자 '{user_id}'의 시뮬레이션(SIMM ID: {simm_id}) 인터페이스 데이터가 성공적으로 추가되었습니다. 다음 단계를 진행하세요."
            logger.debug(_success_msg)
            return _success_msg
        except Exception as e:
            logger.error(f"[Insert Error] 데이터 추가 중 오류가 발생했습니다: {str(e)}")
            return f"데이터 추가 중 오류가 발생했습니다: {str(e)}"

    async def simm_agent(self, user_message: str) -> str:
        """
        SIMM Tools Agent
        """
        logger.debug(f"[Start]: ---------- {self.name} ----------")
        _db_tools_retriever: ObjectRetriever = self.obj_index.as_retriever(
            similarity_top_k=self.valves.TOP_K_TOOLS
        )

        _system_prompt: str = f"""
        당신은 수력/강우 데이터 및 시뮬레이션(모의운영)을 관리하는 엄격한 전문가입니다.
        질문을 받으면 다음의 핵심 행동 강령과 단계를 엄격하게 지키세요.
        [기본 규칙]
        1. 절대로 스스로 추측하지 말고 반드시 필요한 도구를 호출해라.
        2. 필요한 파라미터가 부족한 경우 임의로 지어내지 말고 사용자에게 되물어라.
        
        [시뮬레이션(모의운영) 특별 절차]
        사용자가 '시뮬레이션' 또는 '모의운영' 실행을 요청할 경우 반드시 다음 순서를 따르십시오.
        Step 1: 데이터 증감/설정 조건 확인
        - 사용자의 요청에 강우량 등을 "증가", "감소", "설정" 하라는 조건(예: "10% 증가해서 돌려줘")이 있는지 확인합니다.
        Step 2: 조건에 따른 분기 처리
        - 조건이 없을 경우: 절대 임의로 도구를 실행하지 말고, 즉시 실행을 멈춘 뒤 사용자에게 이렇게 질문하십시오. 
          "기존 강우량 데이터를 그대로 사용하여 모의운영을 실행할까요? 아니면 강우량 증가/감소 조건을 추가하시겠습니까?"
        - 조건이 있을 경우 (예: 10% 증가): 
          ① `get_simm_api_data` 도구를 호출하여 모의운영에 필요한 기존 강우 데이터를 먼저 불러옵니다.
          ② 데이터를 성공적으로 불러왔다면, 사용자가 요청한 조건(10% 증가 등)을 적용할 준비가 된 것입니다.
          ③ 곧바로 `insert_sim_st_if` 도구를 호출하여 시뮬레이션 인터페이스 데이터를 [DBSCHEMA].TB_SIM_ST_IF 테이블에 등록하십시오.
        Step 3: 최종 답변
        - 데이터베이스 Insert가 성공적으로 완료되면, 사용자에게 "강우 데이터 조회를 완료하고, 요청하신 조건(예: 10% 증가)을 적용하여 시뮬레이션 대기열에 등록을 완료했습니다."라고 답변하십시오.
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
            logger.error(f"[Agent Error] 데이터베이스 에이전트 처리 중 오류가 발생했습니다: {str(e)}")
            return f"데이터베이스 에이전트 처리 중 오류가 발생했습니다: {str(e)}"
