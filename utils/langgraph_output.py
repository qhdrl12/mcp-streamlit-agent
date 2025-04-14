from typing import Any, Dict, List, Callable, Optional
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from .stream_types import StreamResult
from .stream_handlers import handle_message_stream, handle_update_stream

async def astream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable[[StreamResult], Any]] = None,
    stream_mode: str = "messages",
    include_subgraphs: bool = False,
) -> StreamResult:
    """
    LangGraph의 실행 결과를 비동기적으로 스트리밍하는 범용 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (Optional[RunnableConfig]): 실행 설정 (선택적)
        node_names (List[str], optional): 처리할 노드 이름 목록. 빈 리스트인 경우 모든 노드 처리
        callback (Optional[Callable], optional): 각 청크 처리를 위한 콜백 함수.
            콜백 함수는 StreamResult 타입의 데이터를 인자로 받습니다.
        stream_mode (str, optional): 스트리밍 모드 ("messages" 또는 "updates"). 기본값은 "messages"
        include_subgraphs (bool, optional): updates 모드에서 서브그래프 포함 여부. 기본값은 False

    Returns:
        StreamResult: 마지막으로 처리된 청크의 정보

    Examples:
        # 1. 기본 출력 콜백 사용
        async def print_callback(data: StreamResult):
            print(f"Node: {data['node']}")
            print(f"Content: {data['content']}")
        
        await astream_graph(
            graph=my_graph,
            inputs={"query": "What is AI?"},
            callback=print_callback
        )

        # 2. 특정 노드만 처리
        await astream_graph(
            graph=my_graph,
            inputs={"query": "Hello"},
            node_names=["agent", "output"],
            callback=my_callback
        )

        # 3. 서브그래프 포함 업데이트 모드
        await astream_graph(
            graph=my_graph,
            inputs={"query": "Test"},
            stream_mode="updates",
            include_subgraphs=True,
            callback=handle_updates
        )
    """
    config = config or {}

    if stream_mode == "messages":
        return await handle_message_stream(
            graph=graph,
            inputs=inputs,
            config=config,
            node_names=node_names,
            callback=callback
        )
    elif stream_mode == "updates":
        return await handle_update_stream(
            graph=graph,
            inputs=inputs,
            config=config,
            node_names=node_names,
            include_subgraphs=include_subgraphs,
            callback=callback
        )
    else:
        raise ValueError(
            f"Invalid stream_mode: {stream_mode}. Must be 'messages' or 'updates'."
        )


async def ainvoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
    include_subgraphs: bool = True,
) -> Dict[str, Any]:
    """
    LangGraph 앱의 실행 결과를 비동기적으로 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (Optional[RunnableConfig]): 실행 설정 (선택적)
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Optional[Callable], optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": Any} 형태의 딕셔너리를 인자로 받습니다.
        include_subgraphs (bool, optional): 서브그래프 포함 여부. 기본값은 True

    Returns:
        Dict[str, Any]: 최종 결과 (마지막 노드의 출력)
    """
    config = config or {}
    final_result = {}

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    # subgraphs 매개변수를 통해 서브그래프의 출력도 포함
    async for chunk in graph.astream(
        inputs, config, stream_mode="updates", subgraphs=include_subgraphs
    ):
        # 반환 형식에 따라 처리 방법 분기
        if isinstance(chunk, tuple) and len(chunk) == 2:
            # 기존 예상 형식: (namespace, chunk_dict)
            namespace, node_chunks = chunk
        else:
            # 단일 딕셔너리만 반환하는 경우 (REACT 에이전트 등)
            namespace = []  # 빈 네임스페이스 (루트 그래프)
            node_chunks = chunk  # chunk 자체가 노드 청크 딕셔너리

        # 딕셔너리인지 확인하고 항목 처리
        if isinstance(node_chunks, dict):
            for node_name, node_chunk in node_chunks.items():
                final_result = {
                    "node": node_name,
                    "content": node_chunk,
                    "namespace": namespace,
                }

                # node_names가 비어있지 않은 경우에만 필터링
                if node_names and node_name not in node_names:
                    continue

                # 콜백 함수가 있는 경우 실행
                if callback is not None:
                    result = callback({"node": node_name, "content": node_chunk})
                    # 코루틴인 경우 await
                    if hasattr(result, "__await__"):
                        await result
                # 콜백이 없는 경우 기본 출력
                else:
                    print("\n" + "=" * 50)
                    formatted_namespace = format_namespace(namespace)
                    if formatted_namespace == "root graph":
                        print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                    else:
                        print(
                            f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                        )
                    print("- " * 25)

                    # 노드의 청크 데이터 출력
                    if isinstance(node_chunk, dict):
                        for k, v in node_chunk.items():
                            if isinstance(v, BaseMessage):
                                v.pretty_print()
                            elif isinstance(v, list):
                                for list_item in v:
                                    if isinstance(list_item, BaseMessage):
                                        list_item.pretty_print()
                                    else:
                                        print(list_item)
                            elif isinstance(v, dict):
                                for node_chunk_key, node_chunk_value in v.items():
                                    print(f"{node_chunk_key}:\n{node_chunk_value}")
                            else:
                                print(f"\033[1;32m{k}\033[0m:\n{v}")
                    elif node_chunk is not None:
                        if hasattr(node_chunk, "__iter__") and not isinstance(
                            node_chunk, str
                        ):
                            for item in node_chunk:
                                print(item)
                        else:
                            print(node_chunk)
                    print("=" * 50)
        else:
            # 딕셔너리가 아닌 경우 전체 청크 출력
            print("\n" + "=" * 50)
            print(f"🔄 Raw output 🔄")
            print("- " * 25)
            print(node_chunks)
            print("=" * 50)
            final_result = {"content": node_chunks}

    # 최종 결과 반환
    return final_result