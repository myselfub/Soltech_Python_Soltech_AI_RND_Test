from __future__ import annotations
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
import logging
import sys
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
        API_URL: str = 'http://[APIIP]:8080/'
        SITE_ID: str = '[SITEUSER]'
        SITE_PW: str = '[SITEPASSWORD]'
        OLLAMA_HOST: str = 'http://[LLMIP]:11434'
        OLLAMA_MODEL: str = 'qwen3:30b-instruct'
        # OLLAMA_MODEL: str = 'qwen3:30b-a3b-instruct-2507-q4_K_M'
        # OLLAMA_MODEL: str = 'danielsheep/Qwen3-Coder-30B-A3B-Instruct-1M-Unsloth:UD-Q5_K_XL'
        # OLLAMA_MODEL: str = 'mdq100/Qwen3-Coder-30B-A3B-Instruct:30b'

    """
        초기화
    """
    def __init__(self):
        self.name: str = 'Light API Pipeline'
        self.description: str = (
            'Light API Pipeline'
        )

        self.valves: Pipeline.Valves = self.Valves(
            **{
                'pipelines': ['*'],
            }
        )

    """
        서버 시작
    """
    async def on_startup(self):
        pass

    """
        서버 종료
    """
    async def on_shutdown(self):
        pass

    """
        API 통신
    """
    def _call_api(self, _url: str, _params: dict) -> dict:
        if not self.session:
            raise Exception(f'[Error] 로그인 후 이용해주세요.,')
        _connection_url: str = self.valves.API_URL + _url
        _headers: dict = {
            'Content-Type': 'application/json;charset=utf8;'
        }
        logger.info(f'[Calling API]: URL= {_connection_url} , Param= {_params}')
        _response: requests.Response = self.session.post(_connection_url, json=_params, headers=_headers)
        if 200 > _response.status_code <= 300:
            logger.error(f'에러 발생: {_response.status_code}')
            raise Exception(f'[Error] {_response.text}({_response.status_code})')

        _result = _response.json()
        logger.info(f'[API Result]: {_result}')
        return _result

    """
        로그인
    """
    def _login(self) -> None:
        _session: requests.Session = requests.Session()
        _login_url: str = self.valves.API_URL + 'login'

        _headers: dict = {
            'Content-Type': 'application/json;charset=utf8;'
        }
        _params: dict = {
            'userId': self.valves.SITE_ID,
            'userPassword': self.valves.SITE_PW,
        }
        _response: requests.Response = _session.post(_login_url, json=_params)
        if 200 > _response.status_code <= 300:
            logger.error(f'에러 발생: {_response.status_code}')
            raise Exception(f'[Error] {_response.text}({_response.status_code})')
        logger.info(f'[Login Success]: URL= {_login_url}')
        self.session = _session

    """
        float 형변환
    """
    @staticmethod
    def safe_convert_float(num: str):
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
        logger.info(f'[Start]: ---------- {self.name} ----------')

        _llm: Ollama = Ollama(model=self.valves.OLLAMA_MODEL, base_url=self.valves.OLLAMA_HOST,
                              request_timeout=300.0,
                              temperature=0.35,
                              keep_alive=1,
                              streaming=True,
                              context_window=16384)

        try:
            self._login()

            _tags_params: dict = {
            }
            _tags_api_response: dict = self._call_api('selectMgmtIndCirc', _tags_params)
            _tags_api_response_list: list = [
                {'tag': __item.get('point', ''), 'desc': __item.get('alias', '')}
                for __item in _tags_api_response.get('response', {}).get('data', [])
                if __item.get('alias', '') != ''
            ]

            _now: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            _data_param_prompt: str = f'''
                # 역할
                사용자의 질문에서 조명제어를 위한 파라미터를 추출하여 정확한 JSON 형식으로 변환하는 전문가
                
                # 데이터 정보
                - **[태그 목록]**: {_tags_api_response_list}
                
                # 출력 가이드라인
                1. **질문 유형 판단**:
                    - 태그 목록 질문: 사용자가 태그의 종류, 목록, 이름, 명칭 등을 묻는 경우 (예: "태그 목록 알려줘", "리스트 보여줘", "어떤 항목이 있어?", "태그명 목록")
                        '**[태그 목록]**'에 있는 데이터를 바탕으로 {{'msg': '태그 목록은 다음과 같습니다.\n01-IO-005: 1층 라운지\n01-IO-006: 1층 주방/화장실\n01-IO-009: '2층 회의실\n...'}} 형태의 JSON으로 출력.
                        단, **[태그 목록]** List의 길이가 100 이상 이라면, 앞의 20개 항목만 출력
                    - 시스템 역할 질문: 사용자가 "무엇을 할 수 있냐", "기능이 뭐냐", "뭐하는 거냐"와 같이 시스템의 역할을 질문할 경우
                        {{'msg': '조명제어를 수행할 수 있습니다.'}}라고만 답변.
                    - 조명제어 지시: 그 외 조명제어 지시의 경우
                        **명령어(cmd) 결정**부터 **값 선택(val)**까지 순서에 맞게 진행.
                2. **명령어(cmd) 결정**:
                    - 항상 {{'cmd': 'outValues'}}로 고정한다. 
                3. **태그 리스트(param)**:
                    - 반드시 **[태그 목록]**에 존재하는 태그명만 선택.
                    - **[태그 목록]** 데이터에 기반하여 질문의 맥락에 맞는 태그를 선택하여 {{'param': [{{'name': '01-IO-005', 'val': '1'}}]}} 형태
                        (예: 1층 라운지 -> '01-IO-005', 2층 회의실 -> '01-IO-009', 3층 우측 -> '01-IO-015')
                    - 질문에서 언급된 대상이 태그 목록에 없으면 태그를 임의로 생성하거나 추론하지 말고 반드시 예외 처리 규칙을 따라라.
                    - 사용자가 장소에 대한 언급이 없다면 태그를 임의로 생성하거나 추론하지 말고 반드시 예외 처리 규칙을 따라라.
                4. **값 선택(val)**:
                    - 불을 키는 행위(Switch On/Turn On)는 {{'val': '1'}}
                    - 불을 끄는 행위(Switch Off/Turn Off)는 {{'val': '0'}}
                5. **예외 처리**:
                    - 사용자의 질문이 조명제어와 관련이 없는 경우: "{{'msg': '업무와 관련된 질문이 아닙니다.'}}"
                    - 질문에서 언급된 대상이 **[태그 목록]**에 존재하지 않는 경우: "{{'msg': '관련 데이터는 없습니다.'}}"
                6. **최종 출력**
                    - 서론, 결론, 설명 또는 마크다운 코드 블록(```json)을 생략하고 **반드시 순수 JSON 형식 문자열만 출력** 출력.
                
                # 입출력 예시
                - 질문: '1층 라운지 불꺼줘'
                    출력: {{'cmd': 'outValues', 'param': [{{'name': '01-IO-005', 'val': '0'}}]}}
                - 질문: '1층 라운지 불켜줘'
                    출력: {{'cmd': 'outValues', 'param': [{{'name': '01-IO-005', 'val': '1'}}]}}
                - 질문: '1층 라운지 태그명이 뭐야'
                    출력: {{'msg': '01-IO-005: 1층 라운지'}}
                - 질문: '사무실 전체 불켜줘'
                    출력: {{'cmd': 'outValues', 'param': [{{'name': '01-P-001', 'val': '1'}}, {{'name': '01-P-003', 'val': '1'}}, {{'name': '01-P-005', 'val': '1'}}, {{'name': '01-P-007', 'val': '1'}}]}}
                - 질문: '태그명 목록'
                    출력: {{'msg': '태그 목록은 다음과 같습니다.\n01-IO-005: 1층 라운지\n01-IO-006: 1층 주방/화장실\n01-IO-009: '2층 회의실\n...'}}
                
                질문: {user_message}
            '''
            logger.info(f'[Prompt]: {_data_param_prompt}')
            _data_param_template: PromptTemplate = PromptTemplate(_data_param_prompt)
            _data_param_llm_response: str = _llm.predict(_data_param_template)

            _data_params: dict = json.loads(_data_param_llm_response)
            logger.info(f'[Prompt Result]: {_data_params}')

            _cmd: str = _data_params.get('cmd', '')
            _msg: str = ''
            if not _cmd:
                _msg: str = _data_params.get('msg', '')
                if not _msg:
                    _msg: str = str(_data_params)
            else:
                _param_list: list = _data_params.get('param', [])

                for _param in _param_list:
                    _data_params['param'] = [_param]
                    _call_api_response: dict = self._call_api('req-tag', _data_params)
                _msg: str = '완료했습니다.'
                '''
                if _call_api_response.get('param', {}).get('msg', '') == 'Success':
                    _msg: str = '성공했습니다.'
                else:
                    _msg: str = '실패했습니다.'
                '''

            logger.info(f'[End]: ---------- {self.name} ----------')
            yield _msg

        except Exception as e:
            logger.error(f'처리중 오류가 발생했습니다: {str(e)}')
            yield f'처리중 오류가 발생했습니다: {str(e)}'
