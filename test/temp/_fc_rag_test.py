from __future__ import annotations
from typing import List, Union, Generator, Iterator
import logging
import sys

from pydantic import BaseModel
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool

# [🔥 Tool Retrieval을 위한 핵심 모듈]
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

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
    embed_model: OllamaEmbedding
    tool_retriever: any  # 도구 검색기 인스턴스 저장용
    valves: 'Valves'

    class Valves(BaseModel):
        OLLAMA_HOST: str = 'http://[LLMIP]:11434'
        OLLAMA_MODEL: str = 'qwen3:30b-instruct'
        # 임베딩 전용 모델 (예: mxbai-embed-large, nomic-embed-text 등 Ollama에 설치된 가벼운 임베딩 모델 추천)
        EMBED_MODEL: str = 'nomic-embed-text'
        # 한 번에 에이전트에게 쥐어줄 최대 도구 개수
        TOP_K_TOOLS: int = 3

    def __init__(self):
        self.name: str = 'Scalable Tool Retrieval Pipeline'
        self.description: str = '수십 개의 프로시저 도구를 벡터 기반으로 검색하여 동적으로 할당하는 에이전트'
        self.valves: Pipeline.Valves = self.Valves(**{'pipelines': ['*']})

    def on_startup(self):
        logger.info('[Startup] 파이프라인 초기화 시작...')

        # 1. LLM 및 임베딩 모델 초기화
        self.llm = Ollama(
            model=self.valves.OLLAMA_MODEL,
            base_url=self.valves.OLLAMA_HOST,
            request_timeout=300.0,
            temperature=0.1,
            streaming=True,
            additional_kwargs={"stop": ["Observation:"]}
        )

        self.embed_model = OllamaEmbedding(
            model_name=self.valves.EMBED_MODEL,
            base_url=self.valves.OLLAMA_HOST
        )

        # 2. 시스템에 존재하는 모든 도구(프로시저)를 리스트업 합니다. (50개가 있어도 문제없음)
        all_tools = self._get_all_registered_tools()

        # 3. [🔥 핵심] 도구들을 벡터 인덱스(Vector Store)에 저장하여 검색기(Retriever) 생성
        tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)

        obj_index = ObjectIndex.from_objects(
            all_tools,
            tool_mapping,
            VectorStoreIndex,
            embed_model=self.embed_model  # 도구의 설명(Docstring)을 벡터화
        )

        # 질문과 유사도가 가장 높은 TOP_K개의 도구만 꺼내오는 검색기 세팅
        self.tool_retriever = obj_index.as_retriever(similarity_top_k=self.valves.TOP_K_TOOLS)

        logger.info(f'[Startup] 총 {len(all_tools)}개의 도구가 벡터 스토어에 인덱싱 되었습니다.')

    def on_shutdown(self):
        pass

    # =========================================================================
    # [가상의 프로시저 도구들] - 실제로는 각각 다른 쿼리나 프로시저를 실행하는 함수들
    # =========================================================================
    def _proc_sales_summary(self, target_month: str) -> str:
        """사용자가 특정 월의 '매출 합계', '영업 이익' 등 월간 재무/매출 요약 통계를 물어볼 때 호출합니다."""
        return f"[{target_month}] 매출 총합: 1억 5천만원"

    def _proc_inventory_list(self, category_code: str) -> str:
        """사용자가 특정 카테고리의 '재고 목록', '상품 리스트', '남은 수량'을 나열해 달라고 할 때 호출합니다."""
        return f"[{category_code}] 재고 리스트: 냉장고(5), TV(2)"

    def _proc_hr_employee_info(self, emp_name: str) -> str:
        """사용자가 '직원 정보', '인사 기록', '부서 조회' 등 사람(직원)에 대한 정보를 찾을 때 호출합니다."""
        return f"[{emp_name}] 인사팀 소속, 직급: 대리"

    def _proc_delivery_status(self, order_id: str) -> str:
        """사용자가 '배송 상태', '택배 위치', '송장 번호'로 주문의 현재 위치를 추적할 때 호출합니다."""
        return f"[{order_id}] 현재 상태: 배송 완료"

    def _proc_customer_voc(self, date: str) -> str:
        """사용자가 특정 일자의 '고객 불만', 'VOC', 'CS 접수 내역'을 확인할 때 호출합니다."""
        return f"[{date}] 접수된 불만 3건 처리 중"

    # =========================================================================

    def _get_all_registered_tools(self) -> List[FunctionTool]:
        """모든 함수를 LlamaIndex FunctionTool 객체로 감싸서 반환합니다."""
        return [
            FunctionTool.from_defaults(fn=self._proc_sales_summary),
            FunctionTool.from_defaults(fn=self._proc_inventory_list),
            FunctionTool.from_defaults(fn=self._proc_hr_employee_info),
            FunctionTool.from_defaults(fn=self._proc_delivery_status),
            FunctionTool.from_defaults(fn=self._proc_customer_voc),
            # ... 50개가 넘는 도구들을 이곳에 계속 추가 ...
        ]

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[
        str, Generator, Iterator]:
        logger.info(f'[Start]: ---------- {self.name} ----------')

        _chat_history: list = []
        for m in messages[:-1]:
            if m['content'] == user_message:
                continue
            role = MessageRole.USER if m['role'] == 'user' else MessageRole.ASSISTANT
            _chat_history.append(ChatMessage(role=role, content=m['content']))

        _memory = ChatMemoryBuffer.from_defaults(chat_history=_chat_history)

        try:
            # 💡 기존의 _system_prompt에서 어떤 도구를 써라 마라 지시할 필요가 완전히 사라집니다.
            _system_prompt: str = '''
            당신은 데이터베이스 조회 전문가입니다.
            제공된 도구(Tools)를 사용하여 사용자의 질문에 정확한 답변을 도출하세요.
            '''

            # [🔥 핵심] tools 파라미터 대신 tool_retriever 파라미터를 넘깁니다.
            # 사용자가 질문을 던지면, 에이전트가 내부적으로 retriever를 먼저 돌려서
            # 50개 중 가장 쓸모 있어 보이는 3개(TOP_K)만 추려낸 뒤 문맥(Context)에 주입합니다.
            _agent = ReActAgent.from_tools(
                tool_retriever=self.tool_retriever,
                llm=self.llm,
                memory=_memory,
                context=_system_prompt,
                verbose=True,
                max_iterations=8
            )

            _response = _agent.stream_chat(message=user_message)

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
