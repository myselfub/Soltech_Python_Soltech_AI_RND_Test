from __future__ import annotations
from typing import List, Union, Generator, Iterator, Iterable

from llama_index.core.base.response.schema import StreamingResponse
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase, PromptTemplate
from sqlalchemy import create_engine, text, bindparam, CursorResult
import sqlalchemy
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Pipeline:
    name: str
    description: str
    engine: sqlalchemy.engine.base.Engine
    valves: 'Valves'

    """
        밸브 설정
    """
    class Valves(BaseModel):
        DB_HOST: str = '[DBIP]'
        DB_PORT: str = '1521'
        DB_USER: str = '[DBUSER]'
        DB_SCHEMA: str = '[DBSCHEMA]'
        DB_PASSWORD: str = '[DBPASSWORD]'
        DB_DATABASE: str = '[DBNAME]'
        DB_TABLES: str = '[DBTABLES]'
        OLLAMA_HOST: str = 'http://[LLMIP]:11434'
        OLLAMA_MODEL: str = 'qwen3:30b-instruct'

    """
        초기화
    """
    def __init__(self):
        self.name: str = 'Oracle Database Pipeline'
        self.description: str = (
            'Oracle Database Pipeline'
        )

        self.valves: Pipeline.Valves = self.Valves(
            **{
                'pipelines': ['*'],
            }
        )

    """
        서버 시작
    """
    def on_startup(self):
        self.engine: sqlalchemy.engine.base.Engine = self._init_db_connection()
        logger.info(f'[DataBase Connected]: {self.valves.DB_HOST}')

    """
        서버 종료
    """
    def on_shutdown(self):
        if self.engine:
            self.engine.dispose()
        pass

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
        _engine: sqlalchemy.engine.base.Engine = create_engine(_connection_url)

        return _engine

    """
        DB에 정의된 테이블 설명 조회
    """
    def _get_table_comments_in_database(self) -> str:
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

    """
        Table에 정의된 테이블 설명 조회
    """

    def _get_table_comments_in_table(self) -> str:
        _schema_info: str = ''
        with self.engine.connect() as _conn:
            _query = text('''
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
            for __table_name, __table_descr, __column_name, __data_type, __column_descrt in _result.fetchall():
                if __table_name != _current_table:
                    _current_table = __table_name
                    __table_desc: str = f' ({__table_descr})' if __table_descr else ''
                    _schema_info += f"\n### TABLE: {__table_name}{__table_desc}\n"

                __column_desc = f' - {__column_descrt}' if __column_descrt else ''
                _schema_info += f'  * {__column_name} [{__data_type}]{__column_desc}\n'

        return _schema_info

    """
        SQLDatabase에서 모든 테이블 리스트 조회
    """
    def get_all_tables(self) -> Iterable:
        _sql_database: SQLDatabase = SQLDatabase(self.engine, schema=self.valves.DB_SCHEMA.lower().strip())
        logger.info(f'[Table loads]: {_sql_database.get_usable_table_names()}')

        return _sql_database.get_usable_table_names()

    """
        SQL 안전 검사
    def validate_sql(self, sql):
        sql_lower = sql.lower()

        if not sql_lower.startswith('select'):
            raise Exception('SELECT 쿼리만 허용됩니다.')

        forbidden = ['insert', 'update', 'delete', 'drop', 'truncate', 'alter']
        for word in forbidden:
            if word in sql_lower:
                raise Exception(f"금지된 키워드 포함: {word}")

        return True
    """

    """
        Pipeline 실행
    """
    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.info(f'[Start]: ---------- {self.name} ----------')

        _tables_lower_list: list = self._get_lower_tables_list()
        _sql_database: SQLDatabase = SQLDatabase(self.engine, schema=self.valves.DB_USER.lower().strip(),
                                   include_tables=_tables_lower_list)
        _schema_info: str = self._get_table_comments_in_table()
        logger.info(f'[Schema Info]: {_schema_info}')

        _llm: Ollama = Ollama(model=self.valves.OLLAMA_MODEL, base_url=self.valves.OLLAMA_HOST, request_timeout=300.0,
                              context_window=16384)

        _sql_prompt: str = f'''
        당신은 Oracle SQL 전문가입니다. 
        제공된 테이블 스키마를 참고하여 사용자의 질문에 최적화된 SQL 쿼리를 생성하고 결과에 기반해 답변하십시오.

        [스키마 정보]
        {_schema_info}

        [작성 규칙]
        1. **문법**:
            - 반드시 Oracle SQL(Oracle 12c 이상) 문법을 사용하십시오. (LIMIT 대신 'FETCH FIRST n ROWS ONLY' 사용)
            - SQL(또는 관련 언어) 쿼리를 작성할 때, 모든 AS(Alias) 키워드 뒤의 별칭은 반드시 쌍따옴표("")로 감싸야 합니다. 예: SELECT column_name AS "alias_name". 이 규칙을 예외 없이 적용하세요.
        2. **제한**: 
            - SELECT 쿼리만 허용합니다. 
            - DELETE, UPDATE, DROP, ALTER 등 데이터 변경 시도는 절대 금지합니다.
            - 사용자가 명시하지 않는 한 최대 100건만 조회하도록 하십시오. (FETCH FIRST 100 ROWS ONLY)
        3. **정렬**: 최신 데이터를 찾을 때는 날짜와 관련된 컬럼을 'DESC' 기준으로 정렬하십시오.
        4. **효율성**: 
            - `SELECT *`를 사용하지 마십시오. 질문에 필요한 핵심 컬럼만 명시하십시오.
            - 중복 제거가 필요한 경우 `DISTINCT`를 적극적으로 활용하십시오.
            - 스키마에 정의된 컬럼명만 사용하고, 필요한 경우 테이블명을 접두어로 사용하십시오(Table.Column).
        5. **금지**: SQL 쿼리 생성 시 서술형 설명이나 주석을 붙이지 말고 오직 실행 가능한 SQL만 출력하십시오.

        [출력 형식]
        모든 대답은 한글로 답변하며, 반드시 아래 형식을 유지하며 각 항목은 한 줄씩 작성하십시오:

        Question: 사용자의 질문 내용
        SQLQuery: 실행할 Oracle SQL 쿼리
        SQLResult: SQL 실행 결과 (이 단계는 시스템에 의해 채워짐)
        Answer: 결과에 기반한 최종 답변

        질문: {user_message}
        SQLQuery: 
        '''
        logger.info(f'[Prompt]: {_sql_prompt}')
        _sql_template: PromptTemplate = PromptTemplate(_sql_prompt)

        _query_engine: NLSQLTableQueryEngine = NLSQLTableQueryEngine(
            sql_database=_sql_database,
            tables=_tables_lower_list,
            llm=_llm,
            embed_model='local',
            text_to_sql_prompt=_sql_template,
            streaming=True
        )

        _response: StreamingResponse = _query_engine.query(user_message)
        logger.info(f'[End]: ---------- {self.name} ----------')
        '''
        if hasattr(_response, "response_gen"):
            for token in _response.response_gen:
                yield token
        else:
            yield str(_response)
        '''

        _full_text = ""
        for _token in _response.response_gen:
            _full_text += _token
        logger.info(_full_text)
        return _response.response_gen

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

pipeline = Pipeline()
pipeline.on_startup()
pipeline.pipe('카테고리가 의류인 항목의 합계 알려줘.', '', [], {})
