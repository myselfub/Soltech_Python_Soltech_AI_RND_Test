from __future__ import annotations

import sys
from typing import List, Union, Generator, Iterator

from llama_index.core.agent import ReActAgent
from llama_index.core.chat_engine.types import StreamingAgentChatResponse, AgentChatResponse
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from llama_index.llms.ollama import Ollama
import logging
import requests
import json
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Pipeline:
    name: str
    description: str
    valves: 'Valves'
    digits: int = 2
    max_length: int = 60

    """
        밸브 설정
    """
    class Valves(BaseModel):
        API_URL: str = 'http://[APIIP]:8080/bizmanager/req-tag'
        OLLAMA_HOST: str = 'http://[LLMIP]:11434'
        OLLAMA_MODEL: str = 'qwen3:30b-instruct'
        # 'qwen3:30b-a3b-instruct-2507-q4_K_M'
        # 'danielsheep/Qwen3-Coder-30B-A3B-Instruct-1M-Unsloth:UD-Q5_K_XL'
        # 'mdq100/Qwen3-Coder-30B-A3B-Instruct:30b'

    """
        초기화
    """
    def __init__(self) -> None:
        self.name: str = 'API FC Pipeline'
        self.description: str = (
            'API FC Pipeline'
        )

        self.valves: Pipeline.Valves = self.Valves(
            **{
                'pipelines': ['*'],
            }
        )

    """
        서버 시작
    """
    def on_startup(self) -> None:
        pass

    """
        서버 종료
    """
    def on_shutdown(self) -> None:
        pass

    """
        API 통신
    """
    def _call_api(self, params: dict = Field(None, description='API 호출에 필요한 body 데이터')) -> dict:
        """
        BizNexus API 통신을 위한 도구
        """
        logger.info('----- _call_api -----')
        __connection_url: str = self.valves.API_URL
        __headers: dict = {
            'Content-Type': 'application/json'
        }
        __response: requests.Response = requests.post(__connection_url, json=params, headers=__headers)
        if 200 > __response.status_code <= 300:
            logger.error(f'API 호출 실패: {__response.status_code}')
            raise Exception(f'[Error] {__response.text}({__response.status_code})')

        __result = __response.json()
        #logger.info(f'[API Result]: {__result}')
        return __result

    """
        현재 시간
    """
    @staticmethod
    def get_current_time() -> str:
        """
        현재 시스템의 날짜와 시간을 'YYYY-MM-DD HH:mm:ss' 형식으로 반환합니다.
        LLM의 내부 지식으로 시간을 말하지 마세요. 반드시 이 도구를 호출해서 얻은 결과만 답변에 사용하세요.
        Returns:
            string: '2026-01-01 00:00:00'
        """
        logger.info('----- get_current_time -----')
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def get_tag_list(self) -> str:
        """
        공장에서 관리하는 모든 태그(온도, 전압 등)의 이름과 설명 목록을 반환합니다.
        """
        logger.info('----- get_tag_list -----')
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

    def find_tag_list(self, keyword: str = '') -> list:
        """
        공장에서 관리하는 모든 태그(온도, 전압 등) 목록에서 특정 키워드가 포함된 태그만 검색하여 이름과 설명 목록을 반환합니다.
        (예: '1층 온도' => 키워드에 '1층' 혹은 '온도' 중 하나)
        Args:
            keyword: 검색할 키워드 (예: '1층', '온도', 'ROOM1')
        """
        logger.info('----- find_tag_list -----')
        _all_tags: list = json.loads(self.get_tag_list())
        _filtered = [_item for _item in _all_tags if keyword in _item['name'] or keyword in _item['desc']]

        return _filtered[:self.max_length]

    """
        API 조회
    """
    def get_factory_data(self, cmd: str = 'reqValues',
                         tag_list: List[str] = None, start: str = '', end: str = '') -> str:
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
        logger.info('----- get_factory_data -----')
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
            logger.info(f'param: {__param}')
            __response: dict = self._call_api(__param)
            __process: dict = self._process_data(__response)
            logger.info(f'process: {__process}')

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
                __sum: float = sum(__vals)
                __count: int = len(__vals)
                __results[__name] = {
                    'count': __count,
                    'sum': __sum,
                    'min': round(min(__vals, default=0), self.digits),
                    'max': round(max(__vals, default=0), self.digits),
                    'avg': round(__sum / __count, self.digits) if __count > 0 else 0
                }
                if __item.get('val', ''):
                    __results[__name].update({'raw_data': __vals})
                elif __item.get('values', []):
                    __results[__name].update({'raw_data': self.slice_list(__item.get('values', []), self.max_length)})

        return __results

    """
        float 형변환
    """
    @staticmethod
    def safe_convert_float(num: str) -> float | None:
        try:
            return float(num)
        except Exception as e:
            logger.error(f'[Error] {e}')
            return None

    """
        list 자르기
    """
    @staticmethod
    def slice_list(target_list: list | str = None, num: int = 20) -> list:
        """
        List를 num의 개수만큼 잘라 반환
        Args:
            target_list: 자를 리스트
            num: 자를 리스트 개수
        """
        logger.info('----- slice_list -----')
        if target_list is None:
            return []
        elif type(target_list) is str:
            target_list: list = json.loads(target_list)

        return target_list[-num:]

    """
        Pipeline 실행
    """
    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        _tool_list: list = [
            #FunctionTool.from_defaults(fn=self._call_api),
            FunctionTool.from_defaults(fn=self.get_current_time),
            #FunctionTool.from_defaults(fn=self.get_tag_list),
            FunctionTool.from_defaults(fn=self.find_tag_list),
            FunctionTool.from_defaults(fn=self.get_factory_data),
            FunctionTool.from_defaults(fn=self.slice_list),
        ]
        _memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults()

        _llm: Ollama = Ollama(model=self.valves.OLLAMA_MODEL, base_url=self.valves.OLLAMA_HOST,
                              request_timeout=300.0, temperature=0.05, keep_alive=1, streaming=True,
                              context_window=16384,
                              additional_kwargs={'stop': ['Observation:']})

        try:
            _system_prompt: str = f'''
            당신은 공장 데이터 관리 전문가입니다.
            사용자의 질문에 답하기 위해 반드시 다음 단계를 지키세요.
            1. 현재 시간을 모른다면 'get_current_time'을 먼저 호출하여 기준을 잡으세요.
            2. 태그 조회를 위한 키워드 추출 진행:
                - 질문에서 핵심 명사(예: '1층', '온도', '전압')를 하나만 추출하세요.
                - 예: '1층 온도' => '1층' 혹은 '온도' 중 하나만 선택
            3. 태그 확인 및 검색 규칙:
                - 추출한 키워드를 'find_tag_list'의 keyword 파라미터에 넣고 호출하여 관련 키워드로 검색하세요. (키워드는 한 단어(예: '1층')로 넣는 것이 가장 검색 효율이 좋습니다.)
                - 'find_tag_list'의 검색 결과가 없거나, 더 넓은 범위의 태그 혹은 전체 태그 확인이 필요할 때만 'get_tag_list'를 호출하세요.
                - **중요:** 사용자가 "태그 목록을 보여달라"고 명시적으로 요청하지 않는 한, 도구에서 반환된 태그 리스트를 답변에 나열하거나 정리해서 보여주지 마세요. 
                - 태그 리스트는 내부적으로 적절한 태그명(name)을 결정하기 위한 참조용으로만 사용하고, 확인 즉시 다음 단계(데이터 조회)를 수행하세요.
            4. 실제 수치(온도, 전압 등)를 조회할 때는 'get_factory_data'를 사용하세요.
                - 오늘/어제 등 상대적인 날짜는 현재 시간을 기준으로 계산해서 YYYY-MM-DD HH:mm:ss 형식으로 넣으세요.
                - 'get_factory_data' 도구 호출 결과 활용 규칙은 아래와 같습니다.
                    -- 통계적인 질문(평균, 최댓값 등)에는 'summary' 데이터를 바탕으로 문장으로 답하세요.
                    -- 상세 내역 조회 요청(데이터 보여줘, 기록 알려줘 등)에는 'raw_data'의 내용을 사용하여 답변하세요.
                    -- 만약 'raw_data'가 너무 많다면({self.max_length}개 초과), "최근 데이터 {self.max_length}개만 표시합니다"라는 안내와 함께 'slice_list'를 사용하여 상위 {self.max_length}개만 표(Markdown Table) 형태로 깔끔하게 보여주세요.
            5. 답변 규칙:
                - 어떤 상황에서도 최종 답변은 한국어로 하세요.
                - 분석 결과가 영어로 추론되더라도 한국어로 번역하여 설명하세요.
                - 업무와 무관한 질문은 거절하세요.
            '''

            _agent: ReActAgent = ReActAgent(tools=_tool_list, llm=_llm, memory=_memory,
                                            context=_system_prompt, verbose=True, max_iterations=10)

            _response: AgentChatResponse = _agent.chat(message=user_message)
            return f'{_response}'

        except Exception as e:
            logger.error(f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
            return f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'


handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

pipeline = Pipeline()
pipeline.on_startup()
pipeline.pipe('3층 온도 알려줘.', '', [], {})
