from __future__ import annotations
from typing import List, Union, Generator, Iterator

import asyncio
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
import logging
import sys
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
        OLLAMA_HOST: str = 'http://[LLMIP]:11434'
        OLLAMA_MODEL: str = 'qwen3:30b-instruct'

    """
        초기화
    """
    def __init__(self) -> None:
        self.name: str = 'FC Pipeline'
        self.description: str = (
            'FC Pipeline'
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
        :param target_list: 리스트
        :param num: 자를 리스트 개수
        :return: 리스트
    """
    @staticmethod
    def slice_list(target_list: list | str = Field(None, description='리스트'),
                   num: int = Field(20, description='자를 리스트 개수')) -> list:
        """
            List를 num의 개수만큼 잘라 반환
        """
        if target_list is None:
            return []
        elif type(target_list) is str:
            target_list: list = json.loads(target_list)

        logger.info('----- slice_list -----')
        return target_list[:num]

    """
        계산
    """
    @staticmethod
    def calculator(equation: str = Field(..., description="계산할 수학 방정식")) -> str:
        """
        수식을 계산한다
        """
        logger.info("-----calculator-----")
        try:
            result = eval(equation)
            return f"{equation} = {result}"
        except Exception as e:
            print(e)
            return "Invalid equation"

    """
        현재시간
    """
    @staticmethod
    def get_current_time() -> str:
        """
        현재 시간을 가져온다.
        LLM의 내부 지식으로 시간을 말하지 마세요. 반드시 이 도구를 호출해서 얻은 결과만 답변에 사용하세요.
        """
        logger.info("-----current_time-----")
        now = datetime.now()
        current_time = now.strftime("%I:%M:%S %p")
        current_date = now.strftime(
            "%A, %B %d, %Y"
        )

        return f"현재 시간 : {current_date}, {current_time}"

    """
        Pipeline 실행
    """
    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        #_message: str = '현재시간을 알려주고, 문자열 맨뒤에 1+2+3+4 결과를 붙여줘'
        _message: str = '현재시간을 알려주고 1+2+3+4 결과도 알려줘'

        _tool_list: list = [
            FunctionTool.from_defaults(fn=Pipeline.slice_list),
            FunctionTool.from_defaults(fn=Pipeline.get_current_time),
            FunctionTool.from_defaults(fn=Pipeline.calculator)
        ]
        _memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults()

        _llm: Ollama = Ollama(model=self.valves.OLLAMA_MODEL, base_url=self.valves.OLLAMA_HOST,
                              request_timeout=300.0, temperature=0.35, keep_alive=1, streaming=True, context_window=16384,
                              additional_kwargs={"stop": ["Observation:"]})

        try:
            _data_param_prompt: str = f'''
            당신은 엄격한 도구 사용 전문가입니다.
            1. 질문을 받으면, 절대로 스스로 추측하지 말고 반드시 필요한 도구를 호출하세요.
            2. 사용자의 질문에 과제가 2개 이상 있다면(예: 시간 확인 + 계산), 모든 과제에 대해 각각 도구를 실행한 후에만 'Final Answer'를 작성할 수 있습니다.
            '''
            _data_param_template: PromptTemplate = PromptTemplate(_data_param_prompt)
            #_data_param_llm_response: str = _llm.predict(_data_param_template)

            _agent: ReActAgent = ReActAgent(tools=_tool_list, llm=_llm, memory=_memory, system_prompt=_data_param_prompt, verbose=True)
            #_agent.update_prompts({"agent_worker:system_prompt": _data_param_template})

            async def get_agent_response():
                # Workflow 기반은 run() 메서드를 사용하며 인자는 user_msg
                result = await _agent.run(user_msg=_message, max_iterations=10)
                return result

            # 현재 실행 중인 루프를 가져오거나 새로 생성
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            _response = loop.run_until_complete(get_agent_response())

            logger.info(type(_response))
            # 5. 결과 반환 (객체이므로 문자열로 변환)
            final_text = str(_response)
            logger.info(f"최종 응답: {final_text}")

            #_response = _agent.run(user_msg=_data_param_prompt)

            return _response

        except Exception as e:
            logger.error(f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
            return f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'
            # return f'데이터를 가져오는 중 오류가 발생했습니다: {str(e)}'

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

pipeline = Pipeline()
pipeline.on_startup()
pipeline.pipe('', '', [], {})
