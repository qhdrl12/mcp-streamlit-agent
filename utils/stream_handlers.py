from typing import Any, Dict, List, Callable, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from .stream_types import MessageChunk, UpdateChunk, StreamResult

async def handle_message_stream(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig],
    node_names: List[str],
    callback: Optional[Callable[[StreamResult], Any]] = None
) -> StreamResult:
    """메시지 모드 스트리밍 처리 핸들러

    Args:
        graph: 컴파일된 LangGraph 객체
        inputs: 입력 데이터
        config: 실행 설정
        node_names: 처리할 노드 이름 목록
        callback: 콜백 함수

    Returns:
        StreamResult: 마지막으로 처리된 청크 정보
    """
    final_result: Optional[MessageChunk] = None

    prev_node = ""

    async for chunk_msg, metadata in graph.astream(
        inputs, config, stream_mode="messages"
    ):
        print(f"chunk_msg: {chunk_msg}, metadata: {metadata}")
        curr_node = metadata["langgraph_node"]
        
        # node_names 필터링
        if node_names and curr_node not in node_names:
            continue
            
        result: MessageChunk = {
            "node": curr_node,
            "content": chunk_msg.content,
            "response_metadata": chunk_msg.response_metadata,
            "metadata": metadata
        }

        if callback:
            callback_result = callback(result)
            if hasattr(callback_result, "__await__"):
                await callback_result
        else:
            # 노드가 변경된 경우에만 구분선 출력
            if curr_node != prev_node:
                print("\n" + "=" * 50)
                print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                print("- " * 25)


        final_result = result

    return final_result 

async def handle_update_stream(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig],
    node_names: List[str],
    include_subgraphs: bool,
    callback: Optional[Callable[[StreamResult], Any]] = None
) -> StreamResult:
    """업데이트 모드 스트리밍 처리 핸들러

    Args:
        graph: 컴파일된 LangGraph 객체
        inputs: 입력 데이터
        config: 실행 설정
        node_names: 처리할 노드 이름 목록
        include_subgraphs: 서브그래프 포함 여부
        callback: 콜백 함수

    Returns:
        StreamResult: 마지막으로 처리된 청크 정보
    """
    final_result: Optional[UpdateChunk] = None

    async for chunk in graph.astream(
        inputs, config, stream_mode="updates", subgraphs=include_subgraphs
    ):  
        print(f"  업데이트 청크: {chunk}")
        # 청크 형식 처리
        namespace: List[str] = []
        node_chunks: Dict[str, Any] = {}

        if isinstance(chunk, tuple) and len(chunk) == 2:
            namespace, node_chunks = chunk
        else:
            node_chunks = chunk

        
        if isinstance(node_chunks, dict):
            for node_name, node_chunk in node_chunks.items():
                if node_names and node_name not in node_names:
                    continue
                    
                result: UpdateChunk = {
                    "node": node_name,
                    "content": node_chunk,
                    "namespace": namespace
                }
                print(f"result: {result}")
                if callback:
                    callback_result = callback(result)
                    if hasattr(callback_result, "__await__"):
                        await callback_result

                final_result = result
        else:
            result: UpdateChunk = {
                "node": "raw_output",
                "content": node_chunks,
                "namespace": namespace if isinstance(chunk, tuple) else []
            }
            
            if callback:
                callback_result = callback(result)
                if hasattr(callback_result, "__await__"):
                    await callback_result

            final_result = result

    return final_result 

# 직접 실행을 위한 코드
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from dotenv import load_dotenv
    
    load_dotenv()


    async def main():
        print("handle_update_stream 실행 테스트")
        
        # model = ChatOpenAI(
        #     model="gpt-4o-mini",
        #     temperature=0.1,
        #     max_tokens=4096
        # )
        # model = ChatAnthropic(
        #     model="claude-3-5-haiku-20241022",
        #     temperature=0.1,
        #     max_tokens=8192
        # )

        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            max_tokens=8192
        )


        agent = create_react_agent(
            model,
            tools=[],
            prompt="인공지능이란 무엇인가요?"
        )

        # 테스트용 업데이트 청크 생성
        # Mock 그래프 설정
        # 콜백 함수 정의
        async def print_updates_callback(chunk):
            print("-[print_callback]-")
            print(f"노드: {chunk['node']}")
            print(f"내용: {chunk['content']}")
            print(f"네임스페이스: {chunk['namespace']}")
            print("-" * 40)
        
        async def print_messages_callback(chunk):
            print(chunk['content'], end="", flush=True)

        
        # handle_update_stream 실행
        # final_result = await handle_update_stream(
        #     graph=agent,
        #     inputs={"messages": [HumanMessage(content="인공지능이란 무엇인가요?")]},
        #     config=None,
        #     node_names=[],  # 모든 노드 처리
        #     include_subgraphs=True,
        #     callback=print_updates_callback
        # )
        final_result = await handle_message_stream(
            graph=agent,
            inputs={"messages": [HumanMessage(content="인공지능이란 무엇인가요?")]},
            config=None,
            node_names=[],  # 모든 노드 처리
            callback=print_messages_callback
        )
        
        print("\n최종 결과:")
        print(f"노드: {final_result['node']}")
        print(f"내용: {final_result['content']}")
        print(final_result)
        # print(f"내용: {final_result['content']['messages'][0].content}")
    
    # 비동기 함수 실행
    asyncio.run(main())



