"""
title: PipeTest
author: Soltech
author_url:
version: 1.0.0
icon_url:
required_open_webui_version: 0.9.0
requirements: llama-index-core==0.12.52.post1, llama-index-embeddings-ollama==0.4.0
"""

from pydantic import BaseModel, Field
from typing import Optional
from llama_index.core.base.llms.types import MessageRole, ChatMessage


class Pipe:
    class Valves(BaseModel):
        LLM_HOST: str = Field(
            default="http://[LLMIP]:11434", description="LLM HOST"
        )
        HISTORY_COUNT: int = Field(default=10, description="참고할 채팅 기록 수")
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    """
        User Message 파싱
    """

    def _parse_user_message(self, messages: list[dict]) -> str:
        _user_message: str = ""
        for __msg in reversed(messages):
            if __msg.get("role") == "user":
                __content: str | list = __msg.get("content", "")
                if isinstance(__content, str):
                    _user_message: str = __content
                elif isinstance(__content, list):
                    __texts: list = []
                    for __item in __content:
                        if __item.get("type") == "text":
                            __texts.append(__item.get("text", ""))
                    _user_message: str = "\n".join(__texts)
                break

        return _user_message

    """
        Chat History 파싱
    """

    def _parse_chat_history(self, user_message: str, messages: list[dict]) -> list:
        _chat_history: list = []
        _history_count: int = -(abs(self.valves.HISTORY_COUNT) + 1)
        for m in messages[_history_count:-1]:
            if m["content"] == user_message:
                continue
            __role: MessageRole = (
                MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
            )
            _chat_history.append(ChatMessage(role=__role, content=m["content"]))

        return _chat_history

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: dict = None,
        __event_emitter__=None,
    ) -> dict:
        _message: list[dict] = body.get("messages", [])
        print(_message)
        _user_message: str = self._parse_user_message(_message)
        print(_user_message)
        import asyncio

        print(f"inlet:{__name__}")
        print("-------------")
        print(f"inlet:body:{body}")
        print("-------------")
        print(f"inlet:user:{__user__}")
        print("-------------")
        print(f"inlet:__metadata__:{__metadata__}")
        print("-------------")
        print(f"inlet:__event_emitter__:{__event_emitter__}")

        yield "테스트시작\n"

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "처리 중입니다...", "done": False},
            }
        )
        await asyncio.sleep(2.5)
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "처리 중 깜박임 없어짐", "done": True},
            }
        )
        await asyncio.sleep(2.5)
        await __event_emitter__(
            {
                "type": "message",
                "data": {
                    "content": f"\n## 메세지 추가\n**메세지 추가**가 완료되었습니다...\n"
                },
            }
        )
        await asyncio.sleep(2.5)
        await __event_emitter__(
            {
                "type": "replace",
                "data": {"content": f"메세지 전체 변경"},
            }
        )
        await asyncio.sleep(2.5)
        yield "테스트 끝"
