from __future__ import annotations
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
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
        self.name: str = 'API Pipeline'
        self.description: str = (
            'API Pipeline'
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
        pass

    """
        서버 종료
    """
    async def on_shutdown(self) -> None:
        pass

    """
        API 통신
    """
    def _call_api(self, _params: dict) -> dict:
        _connection_url: str = self.valves.API_URL
        _headers: dict = {
            'Content-Type': 'application/json'
        }
        logger.debug(f'[Calling API]: URL= {_connection_url} , Param= {_params}')
        _response: requests.Response = requests.post(_connection_url, json=_params, headers=_headers)
        if 200 > _response.status_code <= 300:
            logger.error(f'에러 발생: {_response.status_code}')
            raise Exception(f'[Error] {_response.text}({_response.status_code})')

        _result = _response.json()
        logger.debug(f'[API Result]: {_result}')
        return _result

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
        Pipeline 실행
    """
    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.debug(f'[Start]: ---------- {self.name} ----------')
        _tags_params: dict = {
            'cmd': 'selectTags',
            'param': {
                'category': 'byName',
                'val': ''
            }
        }

        _llm: Ollama = Ollama(model=self.valves.OLLAMA_MODEL, base_url=self.valves.OLLAMA_HOST,
                              request_timeout=300.0, temperature=0.35, keep_alive=1, streaming=True, context_window=16384)

        try:
            _tags_api_response: dict = self._call_api(_tags_params)
            _tag_api_response_list: list = [
                {'name': __item.get('name', ''), 'desc': __item.get('desc', '')}
                for __item in _tags_api_response.get('param', [])
                if not __item.get('name', '$').startswith('$')
            ]

            _now: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            _data_param_prompt: str = f'''
                # 역할
                사용자의 질문에서 데이터 조회를 위한 파라미터를 추출하여 정확한 JSON 형식으로 변환하는 전문가
                
                # 데이터 정보
                - **[기준시간]**: {_now}
                - **[태그 목록]**: {_tag_api_response_list}
                
                # 출력 가이드라인
                1. **질문 유형 판단**:
                    - 태그 목록 질문: 사용자가 태그의 종류, 목록, 이름, 명칭 등을 묻는 경우 (예: "태그 목록 알려줘", "리스트 보여줘", "어떤 항목이 있어?", "태그명 목록")
                        '**[태그 목록]**'에 있는 데이터를 바탕으로 "{{'msg': '태그 목록은 다음과 같습니다.\nROOM1_TEMP: [휴게실] 실내온도\nROOM2_TEMP2: 사무실(3F) 내부온도\nROOM1_L1_AMP: 휴게실 전류(L1)\n...'}}" 형태의 JSON으로 출력.
                        단, **[태그 목록]** List의 길이가 100 이상 이라면, 앞의 20개 항목만 잘라서 출력
                    - 시스템 역할 질문: 사용자가 "무엇을 할 수 있냐", "기능이 뭐냐", "뭐하는 거냐"와 같이 시스템의 역할을 질문할 경우
                        "{{'msg': '데이터 조회와 태그 조회를 수행할 수 있습니다.'}}" 라고만 답변.
                    - 데이터 분석 질문: 그 외 수치 조회, 평균값 계산, 현황 파악 등 일반적인 질문의 경우
                        **명령어(cmd) 결정**부터 **태그 리스트**까지 순서에 맞게 진행.
                2. **명령어(cmd) 결정**:
                    - 실시간/현재 시점 조회("지금 어때?", "현재 온도"): 'reqValues'
                    - 단순 태그명 조회: 'reqValues'
                    - 특정 기간/과거 기록 조회("어제", "지난달", "3월 데이터"): 'fetchValues'
                3. **날짜 형식 (ISO 8601 변형)**:
                    - 모든 날짜는 'YYYY-MM-DD HH:mm:ss' 형식을 엄격히 준수.
                    - 반드시 **[기준시간]**: "{_now}"을 바탕으로 계산할 것.
                    - "오늘"/"금일": 오늘 00:00:00 ~ **[기준시간]**
                    - "어제"/"전일": **[기준시간]**에서 -1일의 00:00:00 ~ **[기준시간]**에서 -1일의 23:59:59
                    - "2026년 3월": 2026-03-01 00:00:00 ~ 2026-03-31 23:59:59 (해당 월의 마지막 날짜 자동 계산)
                4. **태그 리스트(tagList)**:
                    - 반드시 **[태그 목록]**에 존재하는 태그명만 선택.
                    - 질문에서 언급된 대상이 태그 목록에 없으면 태그를 임의로 생성하거나 추론하지 말고 예외 처리 규칙을 따라라.
                    - **[태그 목록]** 데이터에 기반하여 질문의 맥락에 맞는 시스템 태그를 선택.
                        (예: 휴게실 온도 -> 'ROOM1_TEMP', 휴게실 전압 -> 'ROOM1_L1_VOLT', 휴게실 전류 -> 'ROOM1_L1_AMP')
                    - 사용자가 장소에 대한 언급이 없다면 모든 태그를 찾지 말고 "휴게실(ROOM1)"을 기본으로 하나의 태그만 선택.
                        (예: 온도 -> 'ROOM1_TEMP, 전압 -> 'ROOM_L1_VOLT', 전류 -> 'ROOM1_...')
                5. **예외 처리**:
                    - 사용자의 질문이 데이터 조회(온도, 전압, 전류, 태그, 기간 조회, 평균, 추세 등)와 관련이 없는 경우: "{{'msg': '업무와 관련된 질문이 아닙니다.'}}"
                    - 질문에서 언급된 대상이 **[태그 목록]**에 존재하지 않는 경우: "{{'msg': '관련 데이터는 없습니다.'}}"
                6. **최종 출력**
                    - 서론, 결론, 설명 또는 마크다운 코드 블록(```json)을 생략하고 **반드시 순수 JSON 형식 문자열만 출력** 출력.
                
                # 입출력 예시
                - 질문: '지금 실내 온도 알려줘'
                    출력: {{'cmd': 'reqValues', 'param': {{'tagList': ['ROOM1_TEMP']}}}}
                - 질문: '현재 온도 알려줘'
                    출력: {{'cmd': 'reqValues', 'param': {{'tagList': ['ROOM1_TEMP']}}}}
                - 질문: '휴게실 태그명이 뭐야'
                    출력: {{'cmd': 'reqValues', 'param': {{'tagList': ['ROOM1_TEMP']}}}}
                - 질문: '어제 휴게실 온도 기록 보여줘' (**[기준시간]**이 '2026-04-03 12:34:56'인 경우)
                    출력: {{'cmd': 'fetchValues', 'param': {{'start': '2026-04-02 00:00:00', 'end': '2026-04-02 23:59:59', 'span': 10, 'tagList': ['ROOM1_TEMP']}}}}
                - 질문: '3월 휴게실 온도평균' (**[기준시간]**이 '2026-04-03 12:34:56'인 경우)
                    출력: {{'cmd': 'fetchValues', 'param': {{'start': '2026-02-01 00:00:00', 'end': '2026-02-28 23:59:59', 'span': 10, 'tagList': ['ROOM1_TEMP']}}}}
                - 질문: '태그명 목록'
                    출력: {{'msg': '태그 목록은 다음과 같습니다.\nROOM1_TEMP: [휴게실] 실내온도\nROOM2_TEMP2: 사무실(3F) 내부온도\nROOM1_L1_AMP: 휴게실 전류(L1)\n...'}}
                
                질문: {user_message}
            '''
            logger.debug(f'[Prompt]: {_data_param_prompt}')
            _data_param_template: PromptTemplate = PromptTemplate(_data_param_prompt)
            _data_param_llm_response: str = _llm.predict(_data_param_template)

            _data_params: dict = json.loads(_data_param_llm_response)
            logger.debug(f'[Prompt Result]: {_data_params}')

            _cmd: str = _data_params.get('cmd', '')
            if not _cmd:
                _msg: str = _data_params.get('msg', '')
                if not _msg:
                    _msg: str = str(_data_params)
                logger.debug(_msg)
                yield _msg


            _start_date: str = _data_params.get('param', {}).get('start', '')
            _end_date: str = _data_params.get('param', {}).get('end', '')

            _data_api_response: dict = self._call_api(_data_params)
            _data_response_param: list = _data_api_response.get('param', [])
            _data_response_dict: dict = {}
            for _param_item in _data_response_param:
                _param_item_name: str = _param_item.get('name')
                _data_response_dict[_param_item_name] = {}
                _name: str = next(
                    (__item['desc'] for __item in _tag_api_response_list if __item['name'] == _param_item_name), '')

                _param_item_val: str = _param_item.get('val', None)
                if _param_item_val:
                    _data_response_dict[_param_item_name]['명칭'] = _name
                    _data_response_dict[_param_item_name]['개수'] = 1
                    _data_response_dict[_param_item_name]['최솟값'] = None
                    _data_response_dict[_param_item_name]['최댓값'] = None
                    _data_response_dict[_param_item_name]['합계'] = None
                    _data_response_dict[_param_item_name]['평균'] = None
                else:
                    _param_item_values: list = _param_item.get('values', [])
                    _param_item_values_val_list: list = [self.safe_convert_float(_values_item.get('val', '')) for
                                                         _values_item in _param_item_values if self.safe_convert_float(
                            _values_item.get('val', '')) is not None]

                    _count: int = len(_param_item_values_val_list)
                    _sum: float = sum(_param_item_values_val_list)
                    _data_response_dict[_param_item_name]['명칭'] = _name
                    _data_response_dict[_param_item_name]['개수'] = _count
                    # _data_response_dict[_param_item_name]['개수'] = len(_param_item.get('values', []))
                    _data_response_dict[_param_item_name]['최솟값'] = round(min(_param_item_values_val_list, default=0),
                                                                         self.digits)
                    _data_response_dict[_param_item_name]['최댓값'] = round(max(_param_item_values_val_list, default=0),
                                                                         self.digits)
                    _data_response_dict[_param_item_name]['합계'] = round(_sum, self.digits)
                    _data_response_dict[_param_item_name]['평균'] = round(_sum / _count, self.digits) if _count > 0 else 0

            logger.debug(f'[Reference Data]: {_data_response_dict}')

            _api_prompt = f'''
                # 역할
                제공된 데이터를 분석하여 사용자의 질문에 답변하는 분석 전문가.
                
                # 데이터 정보
                - **[기준시간]**: {_now}
                - **[데이터 시작일]**: {_start_date}
                - **[데이터 종료일]**: {_end_date}
                - **[원본 데이터]**: {json.dumps(_data_api_response)}
                - **[참고 데이터]**: {_data_response_dict}
                - 제공된 **[원본 데이터]**의 "param" 리스트 안의 "name"은 태그명(key)이고, "values" 리스트의 내부의 값들은 time이 시간 정보(년도-월-일 시간:분:초.밀리세컨드)이고 val이 데이터의 값.
                - 제공된 **[참고 데이터]**의 key는 태그명.
                
                # 데이터 처리 원칙 (중요)
                1. **전체 기간 준수**: 시작일(start)부터 종료일(end)까지의 모든 데이터를 분석 범위에 포함하고 특정 일자에 치우치지 마라.
                2. **데이터 밀도 확인**: 데이터 포인트가 많을 경우, 전체적인 추세(Trend)를 파악하기 위해 시계열 전체를 고르게 샘플링하여 분석.
                3. **수치 정확도**: 평균값 계산 시 단순히 마지막 값들을 참조하지 말고, 전체 데이터의 총합과 개수를 고려하여 산출하는 논리적 단계를 거쳐라.
                4. **단위**: 데이터의 단위가 부정확하다면 단위를 붙이지 말고, 숫자는 소수점 {self.digits}자리까지 표현.
                5. **참고 데이터 우선 원칙**: 계산된 통계값이 존재할 경우, 원본 데이터를 직접 계산하는 것보다 이 값을 최우선 신뢰 지표로 삼아라.
                6. **모순 검증**: 본인이 분석한 수치와 참고 데이터의 수치가 다를 경우, 참고 데이터를 기준으로 교정.
                7. **예외 처리**:
                   - 사용자의 질문이 데이터 조회(온도, 전압, 전류, 태그, 기간 조회, 평균, 추세 등)와 관련이 없는 경우 다음과 같이 응답: '업무와 관련된 질문이 아닙니다.'
                   - 태그 목록에 없는 태그를 질문한 경우 다음과 같이 응답: '관련 데이터는 없습니다.'
                
                # 출력 가이드라인
                - 사용자의 질문 의도를 파악하고, "참고 데이터"를 활용하거나 "조회된 원본 데이터"를 사용해 적절한 계산 수행하여 질문에 대한 기간을 포함한 답변만 문장 형태로 정확하게 기술.
                1. 질문 유형 판단:
                    - 태그 목록 요청: 사용자가 태그의 종류, 목록, 이름, 명칭 등을 묻는 경우 (예: "태그 알려줘", "리스트 보여줘", "어떤 항목이 있어?", "태그명 알려줘", "태그 목록")
                      데이터 정보의 **[태그 목록]** 항목들을 반드시 "태그명(key): 명칭(desc)" 형식으로 한 줄에 하나씩 목록 형태로 나열.
                      (예: ROOM1_TEMP: 휴게실 온도)
                    - 데이터 분석 응답: 그 외 수치 조회, 평균값 계산, 현황 파악 등 일반적인 질문의 경우
                      사용자가 질의하지 않는 이상 기본적으로 태그명(key)을 노출하지 말고, "참고 데이터"의 "명칭"을 사용하여 문장형으로 답변.
                      (예: '휴게실 온도는 25도입니다.' (O) / 'ROOM1_TEMP는 25도입니다.' (X))
                2. 데이터 구성 및 수치 표현:
                  - 답변 시작 부분에 분석 대상 데이터의 전체 "시작일"과 "종료일"이나 "기준시간"을 반드시 명시.
                  - 태그/명칭이 여러 개일 경우, 각 항목별 수치를 개별적으로 모두 언급.
                  - 숫자는 소수점 {self.digits}자리까지 정확히 표현.
                  - 만약 원본 데이터의 "values" 리스트가 비어 있다면, 해당 기간에 데이터가 존재하지 않음을 정중히 알려라.
                3. [금기 사항]
                  - "참고 데이터에 따르면", "조회된 원본 데이터에 의하면" 등 데이터 출처에 대한 메타 발언을 절대 하지 마라.
                  - 서론, 결론, 부연 설명 없이 사용자의 질문에 대한 핵심 답변만 문장 형태로 기술.
                
                질문: {user_message}
    
                답변:
            '''
            logger.debug(f'[Prompt]: {_api_prompt}')
            _api_template: PromptTemplate = PromptTemplate(_api_prompt)
            logger.debug(f'[End]: ---------- {self.name} ----------')
            for _chunk in _llm.stream(_api_template):
                yield _chunk

        except Exception as e:
            logger.error(f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
            yield f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'
