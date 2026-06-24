from __future__ import annotations
from typing import List, Union, Generator, Iterator
import logging
import sys

from pydantic import BaseModel
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Pipeline:
    name: str
    description: str
    llm: Ollama
    valves: 'Valves'

    class Valves(BaseModel):
        OLLAMA_HOST: str = 'http://[LLMIP]:11434'
        OLLAMA_MODEL: str = 'qwen3:30b-instruct'

    def __init__(self):
        self.name: str = 'Multi-Agent Router Pipeline'
        self.description: str = '질문의 의도에 따라 전문 서브 에이전트(SQL/Procedure)로 라우팅하는 구조'
        self.valves: Pipeline.Valves = self.Valves(**{'pipelines': ['*']})

    def on_startup(self):
        logger.info('[Startup] 파이프라인 초기화 및 공유 LLM 생성...')
        # 에이전트들이 공통으로 사용할 LLM 인스턴스 1개만 생성 (VRAM 절약)
        self.llm = Ollama(
            model=self.valves.OLLAMA_MODEL,
            base_url=self.valves.OLLAMA_HOST,
            request_timeout=300.0,
            temperature=0.1,
            streaming=True,
            additional_kwargs={"stop": ["Observation:"]}
        )

    def on_shutdown(self):
        pass

    # =========================================================================
    # [1단계] 가장 하단에 위치한 실제 도구들 (기능 구현부)
    # =========================================================================
    def _call_query(self, query_instruction: str) -> str:
        """[도구] 실제 DB에 붙어서 SQL을 실행하는 함수 (생략)"""
        return "SQL 실행 결과: 총합 150,000"

    def _proc_inventory(self, category: str) -> str:
        """[도구] 실제 재고 프로시저를 실행하는 함수 (생략)"""
        return f"{category} 재고 내역 표 데이터..."

    # =========================================================================
    # [2단계] 서브 에이전트(전문 부서)를 호출하는 래퍼 도구 (Agent-as-a-Tool)
    # =========================================================================
    def _route_to_sql_agent(self, user_query: str) -> str:
        """
        [라우팅 도구] 사용자가 '합계', '평균', '조건부 검색' 등 능동적인 데이터베이스 계산과
        SQL 쿼리 조회가 필요할 때 이 에이전트에게 질문(user_query)을 그대로 넘기세요.
        """
        logger.info(f'[Router] SQL 전담 에이전트로 작업을 이관합니다. (요청: {user_query})')

        # SQL 전용 툴 세팅
        sql_tools = [FunctionTool.from_defaults(fn=self._call_query)]

        # SQL 전용 시스템 프롬프트 (강력하게 통제 가능)
        sql_context = (
            "당신은 사내 DB를 다루는 최고 수준의 오라클 SQL 데이터 분석가입니다. "
            "주어진 도구를 사용해 SQL을 실행하고, 반드시 숫자가 포함된 정확한 분석 결과를 반환하세요."
        )

        sql_agent = ReActAgent.from_tools(tools=sql_tools, llm=self.llm, context=sql_context, verbose=True)

        # 서브 에이전트는 스트리밍하지 않고 결과만 문자열로 뱉어서 라우터에게 줍니다.
        response = sql_agent.chat(user_query)
        return str(response)

    def _route_to_procedure_agent(self, user_query: str) -> str:
        """
        [라우팅 도구] 사용자가 사내에 정해진 '목록 조회', '재고 확인', '상태 변경' 등
        특정 비즈니스 프로시저를 실행해 달라고 할 때 이 에이전트에게 질문(user_query)을 넘기세요.
        """
        logger.info(f'[Router] Procedure 전담 에이전트로 작업을 이관합니다. (요청: {user_query})')

        # 프로시저 전용 툴 세팅 (여기에 수십 개의 프로시저를 넣어도 SQL 툴과 섞이지 않음)
        proc_tools = [FunctionTool.from_defaults(fn=self._proc_inventory)]

        # 프로시저 전용 시스템 프롬프트
        proc_context = (
            "당신은 사내 비즈니스 프로시저 실행 담당자입니다. "
            "사용자의 요구에 맞는 적절한 프로시저 도구를 찾아 파라미터를 정확히 넣어 실행하세요. "
            "절대 임의로 데이터를 계산하거나 SQL을 짜려고 시도하지 마세요."
        )

        proc_agent = ReActAgent.from_tools(tools=proc_tools, llm=self.llm, context=proc_context, verbose=True)
        response = proc_agent.chat(user_query)
        return str(response)

    # =========================================================================
    # [3단계] 메인 라우터 에이전트 (안내 데스크)
    # =========================================================================
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[
        str, Generator, Iterator]:
        logger.info(f'[Start]: ---------- {self.name} ----------')

        _chat_history: list = []
        for m in messages[:-1]:
            if m['content'] == user_message: continue
            role = MessageRole.USER if m['role'] == 'user' else MessageRole.ASSISTANT
            _chat_history.append(ChatMessage(role=role, content=m['content']))
        _memory = ChatMemoryBuffer.from_defaults(chat_history=_chat_history)

        try:
            # 라우터 에이전트는 오직 2개의 '부서 연결 도구'만 가집니다.
            router_tools = [
                FunctionTool.from_defaults(fn=self._route_to_sql_agent),
                FunctionTool.from_defaults(fn=self._route_to_procedure_agent)
            ]

            router_context: str = '''
            당신은 사용자의 요청을 분석하여 적절한 전문 부서(서브 에이전트)로 안내하는 라우터(Router)입니다.
            직접 질문에 답하려고 하지 말고, 사용자의 원래 질문을 그대로 복사하여 적절한 도구(route_to_...)의 파라미터로 넘기세요.
            전문 에이전트가 답변을 가져오면, 그 내용을 바탕으로 사용자에게 친절하게 최종 답변을 작성해 주세요.
            '''

            router_agent = ReActAgent.from_tools(
                tools=router_tools,
                llm=self.llm,
                memory=_memory,
                context=router_context,
                verbose=True,
                max_iterations=5
            )

            # 라우터 에이전트만 UI로 스트리밍을 수행합니다.
            _response = router_agent.stream_chat(message=user_message)

            def streaming_generator():
                _full_text = ""
                for _token in _response.response_gen:
                    _full_text += _token
                    yield _token
                logger.info(f'[Final Output]: {_full_text}')
                logger.info(f'[End]: ---------- {self.name} ----------')

            return streaming_generator()

        except Exception as e:
            logger.error(f'시스템 오류가 발생했습니다: {str(e)}')
            return f'시스템 오류가 발생했습니다: {str(e)}'
