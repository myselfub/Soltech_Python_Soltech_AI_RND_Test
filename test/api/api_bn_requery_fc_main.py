from __future__ import annotations
from typing import List, Union, Generator, Iterator

from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
import logging
import sys
import json

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
        self.name: str = 'Test'
        self.description: str = (
            'Test'
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

    @staticmethod
    def get_tag_list() -> str:
        """
        시스템에 등록된 사용 가능한 모든 태그(장소 및 항목) 목록을 반환합니다.
        사용자가 장소를 말하지 않았을 때 이 목록을 보고 사용자에게 선택지를 제시하세요.
        """
        return json.dumps([
            {"KEY": "ROOM_01_TEMP", "NAME": "거실 온도"},
            {"KEY": "ROOM_02_TEMP", "NAME": "안방 온도"},
            {"KEY": "KITCHEN_HUMID", "NAME": "주방 습도"}
        ], ensure_ascii=False)

    @staticmethod
    def fetch_tag_data(tag_key: str) -> str:
        """
        특정 태그 키(KEY)를 입력받아 현재 센서 값을 조회합니다.
        반드시 get_tag_list에서 확인된 KEY만 인자로 사용해야 합니다.
        [필수] 사용자가 특정 장소와 항목을 확정하면 이 도구를 즉시 호출하여 값을 제공하세요.
        [경고] 사용자가 장소와 항목을 명확히 선택했을 때만 호출하며 추측해서 넣지 마세요.
        tag_key는 반드시 'ROOM_01_TEMP'와 같은 형식이어야 합니다.
        :param tag_key: 검색할 태그키 (예: 'ROOM_01_TEMP', 'KITCHEN_HUMID')
        :return:
        """
        return json.dumps({
            "ROOM_01_TEMP": 24.5,
            "ROOM_02_TEMP": 22.0,
            "KITCHEN_HUMID": 45.0
        }, ensure_ascii=False)

    """
        Pipeline 실행
    """
    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        _tool_list: list = [
            FunctionTool.from_defaults(fn=Pipeline.get_tag_list),
            FunctionTool.from_defaults(fn=Pipeline.fetch_tag_data)
        ]

        _chat_history = []
        # 마지막 메시지는 ReActAgent.chat()의 인자로 들어갈 것이므로 제외하고 메모리에 넣음
        for m in messages[:-1]:
            if m['content'] == user_message:  # 현재 질문은 제외
                continue
            role = MessageRole.USER if m['role'] == 'user' else MessageRole.ASSISTANT
            _chat_history.append(ChatMessage(role=role, content=m['content']))

        _memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(chat_history=_chat_history)

        _llm: Ollama = Ollama(model=self.valves.OLLAMA_MODEL, base_url=self.valves.OLLAMA_HOST,
                              request_timeout=300.0, temperature=0.1, keep_alive=1, streaming=True, context_window=16384,
                              additional_kwargs={"stop": ["Observation:"]})

        try:
            _system_prompt: str = f'''
            당신은 스마트홈 데이터 조회 전문가입니다. 절대로 사용자의 의도를 추측하지 마세요.

            [핵심 행동 강령]
            1. 사용자가 질문하면 'get_tag_list'를 통해 어떤 장소와 항목이 있는지 먼저 확인한다.
            2. 사용자의 질문에 장소가 언급되지 않았다면(예: 그냥 "온도 알려줘"), 절대로 'fetch_tag_data'를 호출하지 않는다. 
            3. 장소를 모를 때는 반드시 "어느 장소의 온도를 알려드릴까요? 거실, 안방 등이 있습니다."라고 질문만 하고 답변을 기다려라.
            4. 사용자가 명확하게 장소(예: "거실")를 말한 시점에만 'fetch_tag_data'를 호출한다.
            5. '거실'이 리스트의 첫 번째라고 해서 그것을 기본값으로 사용하지 마라.
            6. 목록에 없는 장소면 없는 장소라고 답하라.
            '''

            _agent: ReActAgent = ReActAgent(tools=_tool_list, llm=_llm, memory=_memory, context=_system_prompt, verbose=True, max_iterations=8)

            _response = _agent.chat(message=user_message)

            logger.info(type(_response))
            # 5. 결과 반환 (객체이므로 문자열로 변환)
            final_text = str(_response)
            logger.info(f"최종 응답: {final_text}")

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

msg_01 = "온도가 몇도야"
history = [{"role": "user", "content": msg_01}]
res_01 = pipeline.pipe(msg_01, '', history, {})

msg_02 = "사무실"
history.append({"role": "assistant", "content": str(res_01)})
history.append({"role": "user", "content": msg_02})
res_02 = pipeline.pipe(msg_02, '', history, {})

msg_03 = "안방"
history.append({"role": "assistant", "content": str(res_02)})
history.append({"role": "user", "content": msg_03})
res_03 = pipeline.pipe(msg_03, '', history, {})
