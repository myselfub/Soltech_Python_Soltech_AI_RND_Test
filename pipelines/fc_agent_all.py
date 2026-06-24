"""
title: FC_AGENT
author: Soltech
version: 1.0
requirements: llama-index-embeddings-ollama==0.4.0
"""
from __future__ import annotations

import contextvars
import logging
from typing import List, Union, Generator, Iterator

import json
from datetime import datetime

import oracledb
import requests
import sqlalchemy
from llama_index.core.objects.base_node_mapping import BaseObjectNodeMapping
from oracledb import Var
from pydantic import BaseModel
from sqlalchemy import create_engine, text, bindparam, CursorResult, PoolProxiedConnection, TextClause

from llama_index.core import SQLDatabase, PromptTemplate, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse, AgentChatResponse
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.objects import SimpleToolNodeMapping, ObjectIndex, ObjectRetriever
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from sqlalchemy.engine.interfaces import DBAPICursor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
current_chat_memory_context: contextvars.ContextVar[ChatMemoryBuffer | None] = contextvars.ContextVar('current_chat_memory', default=None)

class Pipeline:
    name: str
    description: str
    valves: 'Valves'

    llm: Ollama = None
    db_tools: DBAgent = None
    api_tools: APIAgent = None
    obj_index: ObjectIndex = None
    embed_model: OllamaEmbedding = None

    """
        밸브 설정
    """
    class Valves(BaseModel):
        LLM_HOST: str = 'http://[LLMIP]:11434'
        LLM_MODEL_ID: str = 'qwen3:30b-instruct'
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
        self.name: str = 'Agent All Pipeline'
        self.description: str = (
            'Agent All Pipeline'
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
        self.db_tools: Pipeline.DBAgent = self._init_db_agent_tools()
        self.api_tools: Pipeline.APIAgent = self._init_api_agent_tools()
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
        return Ollama(model=self.valves.LLM_MODEL_ID, base_url=self.valves.LLM_HOST,
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
    def _init_db_agent_tools(self) -> DBAgent:
        return Pipeline.DBAgent(pipeline_valves=self.valves, llm=self.llm, embed_model=self.embed_model)

    """
        API Agent 초기화
    """
    def _init_api_agent_tools(self) -> APIAgent:
        return Pipeline.APIAgent(pipeline_valves=self.valves, llm=self.llm, embed_model=self.embed_model)

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

            _full_text: str = ''
            for __token in _response.response_gen:
                _full_text += __token
                yield __token
            logger.debug(f'[End]: ---------- {self.name} ----------')

        except Exception as e:
            logger.error(f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
            yield f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'


    class DBAgent:
        valves: 'Valves'
        engine: sqlalchemy.engine.base.Engine = None
        llm: Ollama = None

        """
            밸브 설정
        """
        class Valves(BaseModel):
            DB_HOST: str = None
            DB_PORT: str = None
            DB_USER: str = None
            DB_SCHEMA: str = None
            DB_PASSWORD: str = None
            DB_DATABASE: str = None
            DB_TABLES: str = None

            TOP_K_TOOLS: int = 0

        """
            초기화
        """
        def __init__(self, pipeline_valves: BaseModel, llm: Ollama, embed_model: OllamaEmbedding) -> None:
            self.name: str = 'Oracle Database Agent'

            _valves_dict: dict = pipeline_valves.model_dump()
            self.valves: Pipeline.DBAgent.Valves = self.Valves(**_valves_dict)
            self.engine: sqlalchemy.engine.base.Engine = self._init_db_connection()
            logger.debug(f'[DataBase Connected]: {self.valves.DB_HOST}')

            self.llm: Ollama = llm
            self.embed_model: OllamaEmbedding = embed_model
            self.obj_index: ObjectIndex = self._init_embed_tools()

        """
            서버 종료
        """
        def on_shutdown(self):
            if hasattr(self, 'engine') and self.engine:
                self.engine.dispose()
            pass

        """
           Agent Tool 임베딩
       """
        def _init_embed_tools(self) -> ObjectIndex:
            _db_tools: list = [
                FunctionTool.from_defaults(fn=self._get_table_comments_in_table),
                FunctionTool.from_defaults(fn=self._call_query),
                FunctionTool.from_defaults(fn=self._get_data_procedure),
            ]
            _tool_mapping: BaseObjectNodeMapping = SimpleToolNodeMapping.from_objects(_db_tools)

            return ObjectIndex.from_objects(
                objects=_db_tools,
                object_mapping=_tool_mapping,
                index_cls=VectorStoreIndex,
                embed_model=self.embed_model
            )

        """
            Upper 테이블명 목록
        """
        def _get_upper_tables_list(self) -> list:
            return [t.strip().upper() for t in self.valves.DB_TABLES.split(',')]

        """
            Lower 테이블명 목록
        """
        def _get_lower_tables_list(self) -> list:
            return [t.strip().lower() for t in self.valves.DB_TABLES.split(',')]

        """
            Database 연결
            Oracle + oracledb 드라이버 사용 설정 (형식: oracle+oracledb://user:pass@host:port/?service_name=db)
        """
        def _init_db_connection(self) -> sqlalchemy.engine.base.Engine:
            _connection_url: str = (
                f'oracle+oracledb://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}@'
                f'{self.valves.DB_HOST}:{self.valves.DB_PORT}/'
                f'?service_name={self.valves.DB_DATABASE}'
            )
            _engine: sqlalchemy.engine.base.Engine = create_engine(_connection_url)#, echo=True)

            return _engine

        """
            Table에 정의된 테이블 설명 조회
        """
        def _get_table_comments_in_table(self) -> str:
            """
            사용자가 데이터베이스 테이블 구조, 컬럼 정보, 데이터 타입 또는 테이블의 의미(코멘트)에 대해 질문할 때 호출합니다.

            현재 설정된 스키마와 대상 테이블 목록을 기준으로, DB 딕셔너리를 조회하여 테이블명, 컬럼명,
            데이터 타입 및 코멘트(설명)를 마크다운(Markdown) 형식의 텍스트로 반환합니다.
            반환된 결과를 바탕으로 사용자에게 데이터베이스 구조를 설명하거나 SQL 쿼리를 작성할 수 있습니다.
            Returns:
                str: 테이블 및 컬럼의 메타데이터가 정리된 마크다운 문자열
            """
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

        """
            Query 실행
        """
        def _call_query(self, message: str) -> str:
            """
            사용자가 데이터베이스의 실제 데이터(예: 특정 카테고리의 상품 목록, 재고 수량 등)나 집계를 조회해 달라고 요청할 때 호출합니다.
            자연어 질문(message)을 입력받아, 내부적으로 Oracle SQL 쿼리를 생성하고 실행한 뒤 그 결과값을 문자열로 반환합니다.

            Args:
                message (str): 사용자의 자연어 데이터 조회 요청 (예: '카테고리가 전자인 상품들을 보여줘')
            """
            logger.debug(f'----- call_query -----')
            _tables_lower_list: list = self._get_lower_tables_list()
            _sql_database: SQLDatabase = SQLDatabase(self.engine, schema=self.valves.DB_USER.lower().strip(),
                                                     include_tables=_tables_lower_list)
            _schema_info: str = self._get_table_comments_in_table()
            logger.debug(f'[Schema Info]: {_schema_info}')

            _sql_prompt: str = f'''
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
            '''
            logger.debug(f'[Prompt]: {_sql_prompt}')
            _sql_template: PromptTemplate = PromptTemplate(_sql_prompt)

            _query_engine: NLSQLTableQueryEngine = NLSQLTableQueryEngine(
                sql_database=_sql_database,
                tables=_tables_lower_list,
                llm=self.llm,
                embed_model='local',
                text_to_sql_prompt=_sql_template,
                streaming=True
            )

            _response: StreamingResponse = _query_engine.query(message)
            logger.debug(_response.metadata)
            '''
            _full_text = ''
            for _token in _response.response_gen:
                _full_text += _token
            '''

            return str(_response)

        """
         get_data 프로시저 실행
        """
        def _get_data_procedure(self, category_code: str) -> str:
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
            if category_code == 'ALL':
                return f'[시스템 알림] 구체적인 카테고리 코드가 없습니다. 더 이상 도구를 찾지 말고 즉시 Final Answer에 "어떤 카테고리를 조회할까요?"라고 작성하고 종료하세요.'
            _result_text: str = ''

            _conn: PoolProxiedConnection | None = None
            _cursor: DBAPICursor | None = None
            _ref_cursor: oracledb.Cursor | None = None

            try:
                _conn: PoolProxiedConnection = self.engine.raw_connection()
                _cursor: DBAPICursor = _conn.cursor()

                # 1. Cursor를 반환받을 OUT 파라미터 생성
                _out_cursor: Var = _cursor.var(oracledb.CURSOR)

                # 2. 프로시저 호출
                _procedure_name: str = f'{self.valves.DB_SCHEMA.upper()}.GET_DATA_CURSOR'
                _cursor.callproc(_procedure_name, [category_code, _out_cursor])

                # 3. 반환된 커서에서 데이터 가져오기
                _ref_cursor: oracledb.Cursor = _out_cursor.getvalue()

                if _ref_cursor:
                    # 컬럼명 가져오기 (LLM이 데이터를 이해하기 쉽도록 헤더 추가)
                    _columns: list = [col[0] for col in _ref_cursor.description]
                    _result_text += ' | '.join(_columns) + '\n'
                    _result_text += '-' * 50 + '\n'

                    # 데이터 행 가져오기 (최대 100건 등 제한을 두는 것이 좋습니다)
                    _rows = _ref_cursor.fetchmany(100)
                    if not _rows:
                        return '[시스템 알림] 조회된 데이터가 0건입니다. 절대로 임의의 데이터를 지어내지 마세요. 사용자에게 \'요청하신 조건에 맞는 데이터가 없습니다. 카테고리를 다시 확인해 주세요.\'라고 답변하세요.'

                    for row in _rows:
                        _result_text += ' | '.join(str(item) for item in row) + '\n'
                else:
                    _result_text = '조회된 데이터가 없습니다.'

                logger.debug(f'----- {_result_text} -----')

                return _result_text
            except Exception as e:
                logger.error(f'[Procedure Error]: {str(e)}')
                return f'프로시저(Cursor) 실행 중 오류가 발생했습니다: {str(e)}'
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

        """
            DB Tools Agent
        """
        def db_agent(
                self, user_message: str, memory: ChatMemoryBuffer
        ) -> str:
            logger.debug(f'[Start]: ---------- {self.name} ----------')
            _db_tools_retriever: ObjectRetriever = self.obj_index.as_retriever(similarity_top_k=self.valves.TOP_K_TOOLS)

            try:
                _system_prompt: str = f'''
                당신은 엄격한 도구 사용 전문가다.
                1. 질문을 받으면, 절대로 스스로 추측하지 말고 반드시 필요한 도구를 호출해라.
                2. 사용자의 질문에 과제가 2개 이상 있다면, 모든 과제에 대해 각각 도구를 실행한 후에만 'Final Answer'를 작성할 수 있다.
                3. 필요한 파라메터가 부족한 경우에는 임의의 값을 스스로 지어내서 도구를 호출하지 말고, 사용자에게 "~~ 정보가 필요한데 알려주시겠어요?" 라고 질문만 해라.
                4. 알맞는 정보나 도구가 없다면 "지원하지 않는 기능입니다."라고만 답해라.
                5. [매우 중요] 반드시 정해진 포맷(Thought, Action, Action Input)만 사용하고, 절대로 응답 텍스트 전체를 마크다운 코드 블록(```)으로 감싸지 마라. 오직 평문으로만 출력해라.
                '''

                _agent: ReActAgent = ReActAgent(tools=[], tool_retriever=_db_tools_retriever, llm=self.llm, memory=memory, context=_system_prompt, verbose=True, max_iterations=8)

                _response: AgentChatResponse = _agent.chat(message=user_message)

                logger.debug(_response)
                logger.debug(f'[End]: ---------- {self.name} ----------')

                return str(_response)

            except Exception as e:
                logger.error(f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
                return f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'

    class APIAgent:
        valves: 'Valves'
        llm: Ollama = None

        """
            밸브 설정
        """
        class Valves(BaseModel):
            API_URL: str = None

            TOP_K_TOOLS: int = None
            DIGITS: int = 0
            ARRAY_MAX_LENGTH: int = 0

        """
            초기화
        """
        def __init__(self, pipeline_valves: BaseModel, llm: Ollama, embed_model: OllamaEmbedding) -> None:
            self.name: str = 'API Agent'

            _valves_dict: dict = pipeline_valves.model_dump()
            self.valves: Pipeline.APIAgent.Valves = self.Valves(**_valves_dict)

            self.llm: Ollama = llm
            self.embed_model: OllamaEmbedding = embed_model
            self.obj_index: ObjectIndex = self._init_embed_tools()

        """
            서버 종료
        """
        def on_shutdown(self):
            if hasattr(self, 'engine') and self.engine:
                self.engine.dispose()
            pass

        """
           Agent Tool 임베딩
        """
        def _init_embed_tools(self) -> ObjectIndex:
            _api_tools: list = [
                # FunctionTool.from_defaults(fn=self._call_api),
                FunctionTool.from_defaults(fn=self.get_current_time),
                # FunctionTool.from_defaults(fn=self.get_tag_list),
                FunctionTool.from_defaults(fn=self.find_tag_list),
                FunctionTool.from_defaults(fn=self.get_factory_data),
                FunctionTool.from_defaults(fn=self.slice_list),
            ]
            _tool_mapping: BaseObjectNodeMapping = SimpleToolNodeMapping.from_objects(_api_tools)

            return ObjectIndex.from_objects(
                objects=_api_tools,
                object_mapping=_tool_mapping,
                index_cls=VectorStoreIndex,
                embed_model=self.embed_model
            )

        """
            API 통신
        """
        def _call_api(self, params: dict) -> dict:
            """
            BizNexus API 통신을 위한 도구
            Args:
                params (str): API 호출에 필요한 body 데이터 (예: '카테고리가 전자인 상품들을 보여줘')
            """
            logger.debug('----- _call_api -----')
            __connection_url: str = self.valves.API_URL
            __headers: dict = {
                'Content-Type': 'application/json'
            }
            __response: requests.Response = requests.post(__connection_url, json=params, headers=__headers)
            if not (200 <= __response.status_code < 300):
                logger.error(f'API 호출 실패: {__response.status_code}')
                raise Exception(f'[Error] {__response.text}({__response.status_code})')

            __result = __response.json()
            logger.debug(f'[API Result]: {__result}')
            return __result

        """
            현재 시간
        """
        def get_current_time(self) -> str:
            """
            현재 시스템의 날짜와 시간을 'YYYY-MM-DD HH:mm:ss' 형식으로 반환합니다.
            LLM의 내부 지식으로 시간을 말하지 마세요. 반드시 이 도구를 호출해서 얻은 결과만 답변에 사용하세요.
            Returns:
                string: '2026-01-01 00:00:00'
            """
            logger.debug('----- get_current_time -----')
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        """
            태그 목록 전체 조회
        """
        def get_tag_list(self) -> str:
            """
            공장에서 관리하는 모든 태그(온도, 전압 등)의 이름과 설명 목록을 반환합니다.
            """
            logger.debug('----- get_tag_list -----')
            __params: dict = {
                'cmd': 'selectTags',
                'param': {
                    'category': 'byName',
                    'val': ''
                }
            }
            try:
                __response: dict = self._call_api(__params)
                __tag_list = [
                    {'name': __item.get('name', ''), 'desc': __item.get('desc', '')}
                    for __item in __response.get('param', [])
                    if not __item.get('name', '$').startswith('$')
                ]
                return json.dumps(__tag_list, ensure_ascii=False)
            except Exception as e:
                return f'태그 목록 조회 실패: {str(e)}'

        """
            태그 목록 키워드 조회
        """
        def find_tag_list(self, keyword: str = '') -> list:
            """
            공장에서 관리하는 모든 태그(온도, 전압 등) 목록에서 특정 키워드가 포함된 태그만 검색하여 이름과 설명 목록을 반환합니다.
            (예: '1층 온도' => 키워드에 '1층' 혹은 '온도' 중 하나)
            Args:
                keyword: 검색할 키워드 (예: '1층', '온도', 'ROOM1')
            """
            logger.debug('----- find_tag_list -----')
            _all_tags: list = json.loads(self.get_tag_list())
            _filtered = [_item for _item in _all_tags if keyword in _item['name'] or keyword in _item['desc']]

            return _filtered[:self.valves.ARRAY_MAX_LENGTH]

        """
            API 조회
        """
        def get_factory_data(self, cmd: str = 'reqValues',
                             tag_list: list[str] = None, start: str = '', end: str = '') -> str:
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
            logger.debug('----- get_factory_data -----')
            __param: dict = {
                'cmd': cmd,
                'param': {
                    'tagList': tag_list
                }
            }
            if start and end:
                __param['param'].update({
                    'start': start,
                    'end': end,
                    'span': 10
                })

            try:
                logger.debug(f'param: {__param}')
                __response: dict = self._call_api(__param)
                __process: dict = self._process_data(__response)
                logger.debug(f'process: {__process}')

                return json.dumps(__process, ensure_ascii=False)
            except Exception as e:
                return f'Values 데이터 조회 실패: {str(e)}'

        """
            통계 계산
        """
        def _process_data(self, response: dict) -> dict:
            __results: dict = {}
            for __item in response.get('param', []):
                __name: str = __item.get('name', '')
                __vals: list = []
                if __item.get('val', ''):
                    # fetchValues
                    __vals: list = [float(__item.get('val'))]
                elif __item.get('values', []):
                    __vals: list = [self.safe_convert_float(__v.get('val', '')) for __v in __item.get('values', []) if
                                    self.safe_convert_float(__v.get('val', '')) is not None]

                if __vals:
                    __digits: int = self.valves.DIGITS
                    __sum: float = sum(__vals)
                    __count: int = len(__vals)
                    __results[__name] = {
                        'count': __count,
                        'sum': __sum,
                        'min': round(min(__vals, default=0), __digits),
                        'max': round(max(__vals, default=0), __digits),
                        'avg': round(__sum / __count, __digits) if __count > 0 else 0
                    }
                    if __item.get('val', ''):
                        __results[__name].update({'raw_data': __vals})
                    elif __item.get('values', []):
                        __results[__name].update( {'raw_data': self.slice_list(__item.get('values', []), self.valves.ARRAY_MAX_LENGTH)})

            return __results

        """
            float 형변환
        """
        def safe_convert_float(self, num: str) -> float | None:
            try:
                return float(num)
            except Exception as e:
                logger.error(f'[Error] {e}')
                return None

        """
            list 자르기
        """
        def slice_list(self, target_list: list | str = None, num: int = 20) -> list:
            """
            List를 num의 개수만큼 잘라 반환
            Args:
                target_list: 자를 리스트
                num: 자를 리스트 개수
            """
            logger.debug('----- slice_list -----')
            if target_list is None:
                return []
            elif type(target_list) is str:
                target_list: list = json.loads(target_list)

            return target_list[-num:]

        """
            API Tools Agent
        """
        def api_agent(
                self, user_message: str, memory: ChatMemoryBuffer
        ) -> str:
            logger.debug(f'[Start]: ---------- {self.name} ----------')
            _api_tools_retriever: ObjectRetriever = self.obj_index.as_retriever(similarity_top_k=self.valves.TOP_K_TOOLS)

            try:
                __max_length: int = self.valves.ARRAY_MAX_LENGTH
                _system_prompt: str = f'''
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
                '''

                _agent: ReActAgent = ReActAgent(tools=[], tool_retriever=_api_tools_retriever, llm=self.llm, memory=memory,
                                                context=_system_prompt, verbose=True, max_iterations=8)

                _response: AgentChatResponse = _agent.chat(message=user_message)

                logger.debug(_response)
                logger.debug(f'[End]: ---------- {self.name} ----------')

                return str(_response)

            except Exception as e:
                logger.error(f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
                return f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'
