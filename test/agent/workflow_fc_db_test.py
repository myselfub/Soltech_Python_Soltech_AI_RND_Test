from __future__ import annotations

import logging
from pathlib import Path

import oracledb
import sqlalchemy
from llama_index.core import SQLDatabase, PromptTemplate, VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent, AgentOutput, AgentStream
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping, ObjectRetriever
from llama_index.core.objects.base_node_mapping import BaseObjectNodeMapping
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from oracledb import Var
from pydantic import BaseModel
from sqlalchemy import create_engine, text, bindparam, CursorResult, TextClause, PoolProxiedConnection
from sqlalchemy.engine.interfaces import DBAPICursor
from workflows.handler import WorkflowHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Colors:
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
        self.valves: DBAgent.Valves = self.Valves(**_valves_dict)
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
                _result_text: str = '조회된 데이터가 없습니다.'

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
    async def db_agent(
            self, user_message: str
    ) -> str:
        logger.info(f'[Start]: ---------- {self.name} ----------')
        _db_tools_retriever: ObjectRetriever = self.obj_index.as_retriever(similarity_top_k=self.valves.TOP_K_TOOLS)

        _system_prompt: str = f'''
        당신은 엄격한 도구 사용 전문가다.
        1. 질문을 받으면, 절대로 스스로 추측하지 말고 반드시 필요한 도구를 호출해라.
        2. 사용자의 질문에 과제가 2개 이상 있다면, 모든 과제에 대해 각각 도구를 실행한 후에만 'Final Answer'를 작성할 수 있다.
        3. 필요한 파라메터가 부족한 경우에는 임의의 값을 스스로 지어내서 도구를 호출하지 말고, 사용자에게 "~~ 정보가 필요한데 알려주시겠어요?" 라고 질문만 해라.
        4. 알맞는 정보나 도구가 없다면 "지원하지 않는 기능입니다."라고만 답해라.
        5. [매우 중요] 반드시 정해진 포맷(Thought, Action, Action Input)만 사용하고, 절대로 응답 텍스트 전체를 마크다운 코드 블록(```)으로 감싸지 마라. 오직 평문으로만 출력해라.
        '''
        try:
            _agent: ReActAgent = ReActAgent(llm=self.llm, tools=[], tool_retriever=_db_tools_retriever,
                                            system_prompt=_system_prompt, verbose=False)
            _handler: WorkflowHandler = _agent.run(user_msg=user_message, max_iterations=8)

            _is_streaming_final: bool = False
            _stream_buffer: str = ''
            _final_answer_buffer: str = ''

            async for __event in _handler.stream_events():
                __event_name: str = type(__event).__name__
                if isinstance(__event, AgentStream):
                    if __event.delta:
                        if _is_streaming_final:
                            _final_answer_buffer += __event.delta
                        else:
                            _stream_buffer += __event.delta
                            __keyword: str = ''
                            if 'Final Answer:' in _stream_buffer:
                                __keyword = 'Final Answer:'
                            elif 'Answer:' in _stream_buffer:
                                __keyword: str = 'Answer:'
                            if __keyword:
                                _is_streaming_final: bool = True
                                __split_text: str = _stream_buffer.split(__keyword, 1)[1]
                                if __split_text:
                                    _final_answer_buffer += __split_text
                    continue
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
                            logger.info(f'\n{Colors.YELLOW}[Agent Thought/Log]:\n{__thought}{Colors.RESET}\n')
                    if not _is_streaming_final:
                        _stream_buffer: str = ''
                elif __event_name == 'ToolCallResult':
                    if hasattr(__event, 'tool_call'):
                        __tool_name: str = getattr(__event.tool_call, 'tool_name', 'Unknown')
                        __tool_kwargs: dict = getattr(__event.tool_call, 'tool_kwargs', {})
                        logger.info(f'{Colors.CYAN}Action: {__tool_name}{Colors.RESET}')
                        logger.info(f'{Colors.CYAN}Action Input: {__tool_kwargs}{Colors.RESET}')
                    if hasattr(__event, 'tool_output'):
                        logger.info(f'{Colors.CYAN}Observation: {str(__event.tool_output)[:500]}{Colors.RESET}')

            _result: AgentOutput = await _handler
            if _final_answer_buffer:
                logger.info(f'\n{Colors.GREEN}[API Agent Final Answer]:\n{_final_answer_buffer.strip()}{Colors.RESET}\n')

            logger.debug(f'[End]: ---------- {self.name} ----------')
            if hasattr(_result, 'response'):
                return str(_result.response)
            return str(_result)
        except Exception as e:
            logger.error(f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
            return f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'