from typing import Any, Callable, Dict, List
import uuid
import streamlit as st
import asyncio
import nest_asyncio
import json
from langchain_core.callbacks import AsyncCallbackHandler

# nest_asyncio 적용: 이미 실행 중인 이벤트 루프 내에서 중첩 호출 허용
nest_asyncio.apply()

# 전역 이벤트 루프 생성 및 재사용 (한번 생성한 후 계속 사용)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

# 환경 변수 로드 (.env 파일에서 API 키 등의 설정을 가져옴)
load_dotenv(override=True)


async def astream_graph(
    graph: CompiledStateGraph,
    inputs: Dict[str, Any],
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable[[Dict[str, str]], None] = None,
):
    """
    Asynchronously streams the execution results of a LangGraph.

    Parameters:
    - graph (CompiledStateGraph): The compiled LangGraph to be executed.
    - inputs (dict): The input data dictionary to be passed to the graph.
    - config (RunnableConfig): Execution configuration.
    - node_names (List[str], optional): List of node names to filter output (empty list means all nodes).
    - callback (Callable[[Dict[str, str]], None], optional): A callback function for processing each chunk.
      The callback receives a dictionary with keys "node" (str) and "content" (str).

    Returns:
    - None: This function prints the streaming output but does not return any value.
    """    
    # final_result = {}

    prev_node = ""

    async for chunk_msg, metadata in graph.astream(
        inputs, config, stream_mode="messages"
    ):
        curr_node = metadata["langgraph_node"]
        # final_result = {"node": curr_node, "content": chunk_msg, "metadata": metadata}
        # Process only the specified nodes if node_names is not empty
        if not node_names or curr_node in node_names:
            if callback:
                callback({"node": curr_node, "content": chunk_msg})
            else:
                if curr_node != prev_node:
                    print("\n" + "=" * 50)
                    print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                    print("- " * 25)
                                        
            prev_node = curr_node            

    # return final_result
# 페이지 설정: 제목, 아이콘, 레이아웃 구성
st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")

# 사이드바 최상단에 저자 정보 추가 (다른 사이드바 요소보다 먼저 배치)
st.sidebar.markdown("### ✍️ Made by [테디노트](https://youtube.com/c/teddynote) 🚀")
st.sidebar.divider()  # 구분선 추가

# 기존 페이지 타이틀 및 설명
st.title("🤖 Agent with MCP Tools")
st.markdown("✨ MCP 도구를 활용한 ReAct 에이전트에게 질문해보세요.")

# 세션 상태 초기화
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  # 세션 초기화 상태 플래그
    st.session_state.agent = None  # ReAct 에이전트 객체 저장 공간
    st.session_state.history = []  # 대화 기록 저장 리스트
    st.session_state.mcp_client = None  # MCP 클라이언트 객체 저장 공간
    st.session_state.selected_model = "claude"  # 기본 모델로 Claude 설정

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4()


# --- 함수 정의 부분 ---
def print_message():
    """
    채팅 기록을 화면에 출력합니다.
    """
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"], unsafe_allow_html=True)
        elif message["role"] == "assistant_tool":
            with st.expander("🔧 도구 호출 정보", expanded=False):
                st.markdown(message["content"])

# 새로운 콜백 핸들러 클래스들
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
        # self.text_placeholder.markdown("".join(self.accumulated_text))

class GeminiCallbackHandler(StreamlitCallbackHandler):
    """
    Gemini/Google 모델용 콜백 핸들러
    """
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"[GeminiCallbackHandler] token: {token} , kwargs : {kwargs}")
        # if isinstance(token, dict) and "text" in token:
        #     text = token["text"]
        # else:
        #     text = token
        self.accumulated_text.append(token)
        await self.streamlit_log_tokens(token)
        # self.text_placeholder.markdown("".join(self.accumulated_text))

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

# 기존 process_query 함수를 주석 처리하고 새로운 버전 추가
async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    사용자 질문을 처리하고 응답을 생성합니다.
    """
    try:
        if st.session_state.agent:
            # 새로운 콜백 핸들러 사용
            callback_handler = get_model_callback_handler(
                st.session_state.selected_model,
                text_placeholder,
                tool_placeholder
            )
            
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        config=RunnableConfig(
                            callbacks=[callback_handler],
                            recursion_limit=100,
                            thread_id=st.session_state.thread_id
                        ),
                    ),
                    timeout=timeout_seconds,
                )
                
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                return {"error": error_msg}, error_msg, ""
            
            final_text = "".join(callback_handler.accumulated_text)
            final_tool = "".join(callback_handler.accumulated_tool)

            return response, final_text, final_tool

        else:
            return (
                {"error": "🚫 에이전트가 초기화되지 않았습니다."},
                "🚫 에이전트가 초기화되지 않았습니다.",
                "",
            )
    except Exception as e:
        import traceback
        error_msg = f"❌ 쿼리 처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""


async def initialize_session(mcp_config=None):
    """
    MCP 세션과 에이전트를 초기화합니다.

    매개변수:
        mcp_config: MCP 도구 설정 정보(JSON). None인 경우 기본 설정 사용

    반환값:
        bool: 초기화 성공 여부
    """
    try:
        with st.spinner("🔄 MCP 서버에 연결 중..."):
            print("inner MCP 서버에 연결 중...")
            # 이전 MCP 클라이언트가 존재하면 먼저 정리
            if st.session_state.mcp_client:
                try:
                    await st.session_state.mcp_client.__aexit__(None, None, None)
                except:
                    pass

            client = MultiServerMCPClient(mcp_config)
            await client.__aenter__()
            tools = client.get_tools()
            st.session_state.tool_count = len(tools)
            st.session_state.mcp_client = client

            # 선택된 모델에 따라 다른 LLM 초기화
            selected_model = st.session_state.selected_model
            
            prompt = "Use your tools to answer the question. Answer in Korean."
            if selected_model == "claude":
                model = ChatAnthropic(
                    model="claude-3-5-haiku-20241022", 
                    temperature=0.1, 
                    max_tokens=8192
                )
            elif selected_model == "openai":
                model = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.1,
                    max_tokens=4096
                )
            elif selected_model == "gemini":
                model = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.1,
                    max_tokens=8192
                )
            else:
                # 기본값은 Claude로 설정
                model = ChatAnthropic(
                    model="claude-3-5-haiku-20241022", 
                    temperature=0.1, 
                    max_tokens=8192
                )
            agent = create_react_agent(
                model,
                tools,
                checkpointer=MemorySaver(),
                prompt=prompt,
            )
            st.session_state.agent = agent
            st.session_state.session_initialized = True
            return True
    except Exception as e:
        st.error(f"❌ 초기화 중 오류 발생: {str(e)}")
        import traceback

        st.error(traceback.format_exc())
        return False


# --- 사이드바 UI: MCP 도구 추가 인터페이스로 변경 ---
with st.sidebar.expander("MCP 도구 추가", expanded=False):
    default_config = """{
  "weather": {
    "command": "python",
    "args": ["./adapters/weather_server.py"],
    "transport": "stdio"
  }
}"""
    # pending config가 없으면 기존 mcp_config_text 기반으로 생성
    if "pending_mcp_config" not in st.session_state:
        # configs/default.json 파일에서 값을 로드하여 사용
        with open("configs/default.json", "r") as f:
            default_json = json.load(f)
            
        # default.json의 mcpServers 내용을 pending_mcp_config에 할당
        if "mcpServers" in default_json:
            st.session_state.pending_mcp_config = default_json["mcpServers"]
        else:
            st.session_state.pending_mcp_config = default_json
            

    # 개별 도구 추가를 위한 UI
    st.subheader("개별 도구 추가")
    st.markdown(
        """
    **하나의 도구**를 JSON 형식으로 입력하세요:
    
    ```json
    {
      "도구이름": {
        "command": "실행 명령어",
        "args": ["인자1", "인자2", ...],
        "transport": "stdio"
      }
    }
    ```    
    ⚠️ **중요**: JSON을 반드시 중괄호(`{}`)로 감싸야 합니다.
    """
    )

    # 보다 명확한 예시 제공
    example_json = {
        "github": {
            "command": "npx",
            "args": [
                "-y",
                "@smithery/cli@latest",
                "run",
                "@smithery-ai/github",
                "--config",
                '{"githubPersonalAccessToken":"your_token_here"}',
            ],
            "transport": "stdio",
        }
    }

    default_text = json.dumps(example_json, indent=2, ensure_ascii=False)

    new_tool_json = st.text_area(
        "도구 JSON",
        default_text,
        height=250,
    )

    # 추가하기 버튼
    if st.button(
        "도구 추가",
        type="primary",
        key="add_tool_button",
        use_container_width=True,
    ):
        try:
            # 입력값 검증
            if not new_tool_json.strip().startswith(
                "{"
            ) or not new_tool_json.strip().endswith("}"):
                st.error("JSON은 중괄호({})로 시작하고 끝나야 합니다.")
                st.markdown('올바른 형식: `{ "도구이름": { ... } }`')
            else:
                # JSON 파싱
                parsed_tool = json.loads(new_tool_json)

                # mcpServers 형식인지 확인하고 처리
                if "mcpServers" in parsed_tool:
                    # mcpServers 안의 내용을 최상위로 이동
                    parsed_tool = parsed_tool["mcpServers"]
                    st.info("'mcpServers' 형식이 감지되었습니다. 자동으로 변환합니다.")

                # 입력된 도구 수 확인
                if len(parsed_tool) == 0:
                    st.error("최소 하나 이상의 도구를 입력해주세요.")
                else:
                    # 모든 도구에 대해 처리
                    success_tools = []
                    for tool_name, tool_config in parsed_tool.items():
                        # URL 필드 확인 및 transport 설정
                        if "url" in tool_config:
                            # URL이 있는 경우 transport를 "sse"로 설정
                            tool_config["transport"] = "sse"
                            st.info(
                                f"'{tool_name}' 도구에 URL이 감지되어 transport를 'sse'로 설정했습니다."
                            )
                        elif "transport" not in tool_config:
                            # URL이 없고 transport도 없는 경우 기본값 "stdio" 설정
                            tool_config["transport"] = "stdio"

                        # 필수 필드 확인
                        if "command" not in tool_config and "url" not in tool_config:
                            st.error(
                                f"'{tool_name}' 도구 설정에는 'command' 또는 'url' 필드가 필요합니다."
                            )
                        elif "command" in tool_config and "args" not in tool_config:
                            st.error(
                                f"'{tool_name}' 도구 설정에는 'args' 필드가 필요합니다."
                            )
                        elif "command" in tool_config and not isinstance(
                            tool_config["args"], list
                        ):
                            st.error(
                                f"'{tool_name}' 도구의 'args' 필드는 반드시 배열([]) 형식이어야 합니다."
                            )
                        else:
                            # pending_mcp_config에 도구 추가
                            st.session_state.pending_mcp_config[tool_name] = tool_config
                            success_tools.append(tool_name)

                    # 성공 메시지
                    if success_tools:
                        if len(success_tools) == 1:
                            st.success(
                                f"{success_tools[0]} 도구가 추가되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요."
                            )
                        else:
                            tool_names = ", ".join(success_tools)
                            st.success(
                                f"총 {len(success_tools)}개 도구({tool_names})가 추가되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요."
                            )
        except json.JSONDecodeError as e:
            st.error(f"JSON 파싱 에러: {e}")
            st.markdown(
                f"""
            **수정 방법**:
            1. JSON 형식이 올바른지 확인하세요.
            2. 모든 키는 큰따옴표(")로 감싸야 합니다.
            3. 문자열 값도 큰따옴표(")로 감싸야 합니다.
            4. 문자열 내에서 큰따옴표를 사용할 경우 이스케이프(\\")해야 합니다.
            """
            )
        except Exception as e:
            st.error(f"오류 발생: {e}")

    # 구분선 추가
    st.divider()

    # 현재 설정된 도구 설정 표시 (읽기 전용)
    st.subheader("현재 도구 설정 (읽기 전용)")
    st.code(
        json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False)
    )

# --- 등록된 도구 목록 표시 및 삭제 버튼 추가 ---
with st.sidebar.expander("등록된 도구 목록", expanded=True):
    try:
        pending_config = st.session_state.pending_mcp_config
    except Exception as e:
        st.error("유효한 MCP 도구 설정이 아닙니다.")
    else:
        # pending config의 키(도구 이름) 목록을 순회하며 표시
        for tool_name in list(pending_config.keys()):
            col1, col2 = st.columns([8, 2])
            col1.markdown(f"- **{tool_name}**")
            if col2.button("삭제", key=f"delete_{tool_name}"):
                # pending config에서 해당 도구 삭제 (즉시 적용되지는 않음)
                del st.session_state.pending_mcp_config[tool_name]
                st.success(
                    f"{tool_name} 도구가 삭제되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요."
                )

with st.sidebar:
    # 모델 선택 UI 추가
    st.subheader("🚀 모델 선택")
    model_options = {
        "claude": "Claude 3.5 Haiku (Anthropic)",
        "openai": "GPT-4o (OpenAI)",
        "gemini": "Gemini 1.5 Pro (Google)"
    }
    
    selected_model = st.selectbox(
        "LLM 모델을 선택하세요",
        list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.selected_model)
    )
    
    # 선택된 모델이 변경되면 세션 상태 업데이트 및 재초기화
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.session_initialized = False
        st.session_state.agent = None
        st.warning(f"모델이 변경되었습니다. 에이전트를 재초기화합니다.")
        
        # 초기화 실행
        try:
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )
            
            if success:
                st.success("✅ 새로운 모델 설정이 적용되었습니다.")
            else:
                st.error("❌ 새로운 모델 적용에 실패하였습니다.")
        except Exception as e:
            st.error(f"❌ 모델 변경 중 오류 발생: {str(e)}")
            
        st.rerun()  # 페이지 새로고침
    
    st.divider()
    
    # 적용하기 버튼: pending config를 실제 설정에 반영하고 세션 재초기화
    if st.button(
        "도구설정 적용하기",
        key="apply_button",
        type="primary",
        use_container_width=True,
    ):
        # 적용 중 메시지 표시
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 변경사항을 적용하고 있습니다. 잠시만 기다려주세요...")
            progress_bar = st.progress(0)

            # 진행 상태 업데이트
            progress_bar.progress(30)
            
            # 세션 초기화 준비
            st.session_state.session_initialized = False
            st.session_state.agent = None
            st.session_state.mcp_client = None

            print(f"st.session_state.pending_mcp_config: {st.session_state.pending_mcp_config}")

            # 초기화 실행
            try:
                success = st.session_state.event_loop.run_until_complete(
                    initialize_session(st.session_state.pending_mcp_config)
                )
                
                # 진행 상태 업데이트
                progress_bar.progress(100)

                if success:
                    st.success("✅ 새로운 MCP 도구 설정이 적용되었습니다.")
                else:
                    st.error("❌ 새로운 MCP 도구 설정 적용에 실패하였습니다.")
            except Exception as e:
                progress_bar.progress(100)
                st.error(f"❌ 도구 설정 적용 중 오류 발생: {str(e)}")

        # 페이지 새로고침
        st.rerun()


# --- 기본 세션 초기화 (초기화되지 않은 경우) ---
if not st.session_state.session_initialized:
    st.info("🔄 MCP 서버와 에이전트를 초기화합니다. 잠시만 기다려주세요...")
    success = st.session_state.event_loop.run_until_complete(
        initialize_session(st.session_state.pending_mcp_config)
    )
    if success:
        st.success(
            f"✅ 초기화 완료! {st.session_state.tool_count}개의 도구가 로드되었습니다."
        )
    else:
        st.error("❌ 초기화에 실패했습니다. 페이지를 새로고침해 주세요.")


# --- 대화 기록 출력 ---
print_message()

# --- 사용자 입력 및 처리 ---
user_query = st.chat_input("💬 질문을 입력하세요")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user").markdown(user_query)
        with st.chat_message("assistant"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(user_query, text_placeholder, tool_placeholder)
                )
            )

        print(f"chat resp: {resp}\nfinal_text: {final_text}\nfinal_tool: {final_tool}")
        if resp and "error" in resp:
            st.error(resp["error"])
        else:
            # 모델별 태그 스타일 정의
            model_tags = {
                "claude": {"name": "Claude", "color": "orange"},
                "openai": {"name": "GPT", "color": "black"},
                "gemini": {"name": "Gemini", "color": "blue"}
            }
            
            current_model = st.session_state.selected_model
            model_info = model_tags.get(current_model, {"name": "AI", "color": "gray"})
            
            # HTML 스타일의 태그 생성
            model_tag = f'<span style="background-color: {model_info["color"]}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 8px;">{model_info["name"]}</span>'
            
            # 태그를 답변 앞에 추가
            final_text_with_tag = f"{model_tag}{final_text}"

            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append(
                {"role": "assistant", "content": final_text_with_tag}
            )
            if final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            st.rerun()
    else:
        st.warning("⏳ 시스템이 아직 초기화 중입니다. 잠시 후 다시 시도해주세요.")

# --- 사이드바: 시스템 정보 표시 ---
with st.sidebar:
    st.subheader("🔧 시스템 정보")
    st.write(f"🛠️ MCP 도구 수: {st.session_state.get('tool_count', '초기화 중...')}")
    
    # 현재 사용 중인 모델 표시
    model_display_names = {
        "claude": "Claude 3.5 Haiku (Anthropic)",
        "openai": "GPT-4o (OpenAI)",
        "gemini": "Gemini 1.5 Pro (Google)"
    }
    st.write(f"🚀 현재 모델: {model_display_names.get(st.session_state.selected_model, '알 수 없음')}")

    # 구분선 추가 (시각적 분리)
    st.divider()

    # 사이드바 최하단에 대화 초기화 버튼 추가
    if st.button("🔄 대화 초기화", use_container_width=True, type="primary"):
        # thread_id 초기화
        st.session_state.thread_id = uuid.uuid4()

        # 대화 히스토리 초기화
        st.session_state.history = []

        # 알림 메시지
        st.success("✅ 대화가 초기화되었습니다.")

        # 페이지 새로고침
        st.rerun()

        