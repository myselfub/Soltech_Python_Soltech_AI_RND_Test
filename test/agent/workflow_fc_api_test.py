from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import requests
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent, AgentOutput, AgentStream
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping, ObjectRetriever
from llama_index.core.objects.base_node_mapping import BaseObjectNodeMapping
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel
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
        self.valves: APIAgent.Valves = self.Valves(**_valves_dict)

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
            #FunctionTool.from_defaults(fn=self._call_api),
            FunctionTool.from_defaults(fn=self.get_current_time),
            #FunctionTool.from_defaults(fn=self.get_tag_list),
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
                    __results[__name].update({'raw_data': self.slice_list(__item.get('values', []), self.valves.ARRAY_MAX_LENGTH)})

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
    async def api_agent(
            self, user_message: str
    ) -> str:
        logger.info(f'[Start]: ---------- {self.name} ----------')
        _api_tools_retriever: ObjectRetriever = self.obj_index.as_retriever(similarity_top_k=self.valves.TOP_K_TOOLS)

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
        try:
            _agent: ReActAgent = ReActAgent(llm=self.llm, tools=[], tool_retriever=_api_tools_retriever,
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