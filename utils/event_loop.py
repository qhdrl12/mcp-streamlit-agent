import asyncio
import nest_asyncio
import streamlit as st

# nest_asyncio 적용: 이미 실행 중인 이벤트 루프 내에서 중첩 호출 허용
nest_asyncio.apply()

def initialize_event_loop():
    """
    이벤트 루프를 초기화하고 세션 상태에 저장합니다.
    이미 초기화되어 있다면 기존 루프를 반환합니다.
    """
    if "event_loop" not in st.session_state:
        loop = asyncio.new_event_loop()
        st.session_state.event_loop = loop
        asyncio.set_event_loop(loop)
        return loop
    return st.session_state.event_loop

def ensure_event_loop():
    """
    현재 이벤트 루프가 설정되어 있는지 확인하고, 
    없으면 새로 생성하거나 세션 상태에서 가져옵니다.
    
    Returns:
        asyncio.AbstractEventLoop: 현재 활성화된 이벤트 루프
    """
    try:
        loop = asyncio.get_event_loop()
        
        # 이벤트 루프가 닫혀있는 경우 새로 생성
        if loop.is_closed():
            return initialize_event_loop()
        
        return loop
    except RuntimeError:
        # 이벤트 루프가 없거나 접근할 수 없는 경우 새로 생성
        return initialize_event_loop()