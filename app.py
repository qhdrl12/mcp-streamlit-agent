import json
import uuid
import streamlit as st
import asyncio
from typing import Any, Callable, Dict, List
from utils.event_loop import initialize_event_loop, ensure_event_loop
from utils.callbacks import get_model_callback_handler

initialize_event_loop()

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

    prev_node = ""

    async for chunk_msg, metadata in graph.astream(
        inputs, config, stream_mode="messages"
    ):
        curr_node = metadata["langgraph_node"]
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

# 페이지 설정: 제목, 아이콘, 레이아웃 구성
st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")

# 사이드바 최상단에 저자 정보 추가 (다른 사이드바 요소보다 먼저 배치)
st.sidebar.markdown("### ✍️ Forked from [테디노트](https://youtube.com/c/teddynote) | Developed by [qhdrl12](https://github.com/qhdrl12)")
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


# 기존 process_query 함수를 주석 처리하고 새로운 버전 추가
async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    사용자 질문을 처리하고 응답을 생성합니다.
    """
    try:
        ensure_event_loop()

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


def _create_agent_with_model(model_name: str, tools: List[Any]):
    """Creates the specific LLM and the ReAct agent."""
    prompt = "Use your tools to answer the question. Answer in Korean."
    if model_name == "claude":
        model = ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=0.1,
            max_tokens=8192
        )
    elif model_name == "openai":
        model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=4096
        )
    elif model_name == "gemini":
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            max_tokens=8192
        )
    else:
        # Default to Claude
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
    return agent


async def initialize_mcp_connection(mcp_config=None):
    """
    Initializes the MCP client connection and retrieves tools.
    Handles closing of the previous client.

    Returns:
        tuple(MultiServerMCPClient | None, List[Any] | None): 
            A tuple containing the client and tools, or (None, None) on failure.
    """
    # Close previous client if exists
    if st.session_state.mcp_client:
        try:
            print(f"기존 MCP 클라이언트 종료")
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None # Clear reference after closing
        except Exception as client_exit_e:
            print(f"Error closing previous MCP client: {client_exit_e}") # Log error but continue
            st.session_state.mcp_client = None # Attempt to clear reference even if close failed

    # Create and enter new client
    try:
        with st.spinner("🔄 MCP 서버에 연결 중..."):
            print("inner MCP 서버에 연결 중...")
            client = MultiServerMCPClient(mcp_config)
            await client.__aenter__()
            tools = client.get_tools()
            print(f"MCP Connection successful, {len(tools)} tools retrieved.")
            return client, tools # Return client and tools
    except Exception as e:
        st.error(f"❌ MCP 클라이언트 연결 또는 도구 로딩 실패: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None # Indicate failure


async def apply_and_reinitialize(config):
    """
    Applies the provided MCP configuration, re-initializes the MCP connection,
    and recreates the agent. Handles status updates and errors.
    """
    apply_status = st.empty()
    initialization_attempted = False # Flag to track if full init was attempted
    success = False
    try:
        with apply_status.container():
            st.warning("🔄 MCP 도구 설정을 적용하고 에이전트를 재구성합니다...")
            progress_bar = st.progress(0)
            progress_bar.progress(10)

            # Clear old state
            st.session_state.session_initialized = False
            st.session_state.agent = None
            st.session_state.tools = None
            # st.session_state.mcp_client is handled within initialize_mcp_connection

            print(f"Applying new MCP config: {config}")
            initialization_attempted = True
            
            # Initialize MCP Connection
            progress_bar.progress(30)
            new_client, new_tools = await initialize_mcp_connection(config)
            progress_bar.progress(60)

            if new_client and new_tools:
                st.session_state.mcp_client = new_client
                st.session_state.tools = new_tools # Store tools
                st.session_state.tool_count = len(new_tools)

                try:
                    # Create Agent using the current model selection
                    selected_model_name = st.session_state.selected_model
                    st.session_state.agent = _create_agent_with_model(selected_model_name, new_tools)
                    st.session_state.session_initialized = True
                    success = True
                    progress_bar.progress(100)
                    st.success("✅ MCP 도구 설정 적용 및 에이전트 재구성 완료.")
                    await asyncio.sleep(2)
                    apply_status.empty()
                except Exception as agent_e:
                    progress_bar.progress(100)
                    st.error(f"❌ 에이전트 생성 중 오류 발생: {agent_e}")
                    # Keep error message visible
            else:
                # initialize_mcp_connection already showed an error
                progress_bar.progress(100)
                st.error("❌ MCP 연결 또는 도구 로딩 실패로 에이전트를 생성할 수 없습니다.")
                # Keep error message visible

    except Exception as outer_e:
        # Catch errors in the status display logic itself
        st.error(f"❌ 재초기화 상태 업데이트 중 오류 발생: {str(outer_e)}")
        # Attempt to clear any lingering status message if possible
        try:
            apply_status.empty()
        except: pass # Ignore cleanup errors

    # Rerun only if initialization was actually attempted and successful
    if initialization_attempted and success:
        st.rerun()
    elif not success:
         # Clear spinner/warning if it failed but didn't rerun
         try: apply_status.empty()
         except: pass


async def recreate_agent_only():
    """
    Recreates the agent using the currently selected model and existing tools.
    Assumes the MCP client connection is already valid.
    """
    
    success = False
    try:        
        st.session_state.session_initialized = False # Mark as uninitialized during agent creation
        st.session_state.agent = None # Clear old agent

        selected_model_name = st.session_state.selected_model
        tools = st.session_state.tools
        new_agent = _create_agent_with_model(selected_model_name, tools)

        st.session_state.agent = new_agent
        st.session_state.session_initialized = True
        success = True

    except Exception as e:
         st.error(f"❌ 에이전트 재구성 중 오류 발생: {e}")
         import traceback
         st.error(traceback.format_exc())
         # Keep error message visible
         st.session_state.session_initialized = False # Ensure state reflects failure

    # Rerun only on success, show toast first
    if success:
        st.rerun()
    else:
        pass


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

                    # 성공 메시지 대신 즉시 적용
                    if success_tools:
                        st.info(f"추가된 도구: {', '.join(success_tools)}. 변경사항 적용 중...")
                        # Apply changes immediately
                        st.session_state.event_loop.run_until_complete(
                            apply_and_reinitialize(st.session_state.pending_mcp_config)
                        )
                        # Rerun happens inside apply_and_reinitialize on success

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
                tool_name_deleted = tool_name # Store name for message
                del st.session_state.pending_mcp_config[tool_name]
                st.info(f"{tool_name_deleted} 도구가 삭제되었습니다. 변경사항 적용 중...")
                # Apply changes immediately
                st.session_state.event_loop.run_until_complete(
                    apply_and_reinitialize(st.session_state.pending_mcp_config)
                )
                # Rerun happens inside apply_and_reinitialize on success


with st.sidebar:

    # 모델 선택 UI 추가
    st.subheader("🚀 모델 선택")
    model_options = {
        "claude": "Claude 3.5 Haiku (Anthropic)",
        "openai": "GPT-4o (OpenAI)",
        "gemini": "Gemini 1.5 Pro (Google)"
    }
    
    # Get index of currently selected model, default to 0 if not found
    current_model_key = st.session_state.selected_model
    default_index = 0
    model_keys = list(model_options.keys())
    if current_model_key in model_keys:
        default_index = model_keys.index(current_model_key)

    selected_model = st.selectbox(
        "LLM 모델을 선택하세요",
        model_keys,
        format_func=lambda x: model_options[x],
        index=default_index
    )
    
    # 선택된 모델이 변경되면 에이전트만 재구성
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Call the function to recreate only the agent
        st.session_state.event_loop.run_until_complete(
            recreate_agent_only()
        )
        # Rerun is handled within recreate_agent_only on success


# --- 기본 세션 초기화 (초기화되지 않은 경우) ---
if not st.session_state.session_initialized:
    st.info("🔄 MCP 서버 연결 및 에이전트 초기화 중...")
    # Use the full reinitialize function for the initial setup
    st.session_state.event_loop.run_until_complete(
        apply_and_reinitialize(st.session_state.pending_mcp_config)
    )
    # apply_and_reinitialize handles success/error messages and potential rerun


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

        