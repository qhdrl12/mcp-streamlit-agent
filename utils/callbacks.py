from typing import Any, Dict, List
import json
import streamlit as st
from langchain_core.callbacks import AsyncCallbackHandler


class StreamlitCallbackHandler(AsyncCallbackHandler):
    """
    기본 Streamlit 콜백 핸들러
    """
    def __init__(self, text_placeholder, tool_placeholder):
        self.text_placeholder = text_placeholder
        self.tool_placeholder = tool_placeholder
        self.accumulated_text = []
        self.accumulated_tool = []

    async def streamlit_log_tokens(self, text: str):
        self.text_placeholder.markdown("".join(self.accumulated_text))

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        print(f"on_tool_start serialized: {serialized}, input_str: {input_str}, kwargs: {kwargs}")
        self.accumulated_tool.append("\n```json\n" + json.dumps(serialized, indent=2, ensure_ascii=False) + "\n```\n")
        self.accumulated_tool.append("\n```json\n" + input_str + "\n```\n")
        print(f"on_tool_start accumulated_tool: {self.accumulated_tool}")
        with self.tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
            st.markdown("".join(self.accumulated_tool))

    async def on_tool_end(self, output: str, **kwargs) -> None:
        print(f"on_tool_end output: {output}, kwargs: {kwargs}")
        
        # output에서 content 영역을 추출하여 accumulated_tool에 추가
        if hasattr(output, 'content'): # claude
            # 한글 문자가 유니코드 이스케이프 시퀀스로 표시되는 경우를 처리
            try:
                # 문자열이 JSON 형식인지 확인하고 파싱
                json_obj = json.loads(output.content)
                # 다시 한글이 제대로 표시되도록 JSON으로 변환
                formatted_content = json.dumps(json_obj, indent=2, ensure_ascii=False)
                self.accumulated_tool.append("\n```json\n" + formatted_content + "\n```\n")
            except (json.JSONDecodeError, TypeError):
                # JSON 파싱에 실패하면 원본 내용 그대로 사용
                self.accumulated_tool.append("\n```json\n" + output.content + "\n```\n")
        else:
            self.accumulated_tool.append("\n```json\n" + output + "\n```\n")
        with self.tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
            st.markdown("".join(self.accumulated_tool))

    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        with self.tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
            st.markdown("".join(self.accumulated_tool))


class ClaudeCallbackHandler(StreamlitCallbackHandler):
    """
    Claude/Anthropic 모델용 콜백 핸들러
    """
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"[ClaudeCallbackHandler] token: {token} , kwargs : {kwargs}")
        if isinstance(token, list):
            content = token[0]
            if isinstance(content, dict) and "text" in content:
                self.accumulated_text.append(content["text"])
                await self.streamlit_log_tokens(token)


class GPTCallbackHandler(StreamlitCallbackHandler):
    """
    GPT/OpenAI 모델용 콜백 핸들러
    """
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"[GPTCallbackHandler] token: {token} , kwargs : {kwargs}")
        self.accumulated_text.append(token)
        await self.streamlit_log_tokens(token)


class GeminiCallbackHandler(StreamlitCallbackHandler):
    """
    Gemini/Google 모델용 콜백 핸들러
    """
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"[GeminiCallbackHandler] token: {token} , kwargs : {kwargs}")
        self.accumulated_text.append(token)
        await self.streamlit_log_tokens(token)


def get_model_callback_handler(model_type: str, text_placeholder, tool_placeholder) -> AsyncCallbackHandler:
    """
    모델 타입에 따른 적절한 콜백 핸들러를 반환합니다.
    """
    handlers = {
        "claude": ClaudeCallbackHandler,
        "openai": GPTCallbackHandler,
        "gemini": GeminiCallbackHandler
    }
    handler_class = handlers.get(model_type, StreamlitCallbackHandler)
    return handler_class(text_placeholder, tool_placeholder) 