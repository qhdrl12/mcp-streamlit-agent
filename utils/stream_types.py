from typing import TypedDict, Any, Dict, List, Optional, Union
from langchain_core.messages import BaseMessage

class StreamMetadata(TypedDict):
    """스트리밍 메타데이터 타입 정의"""
    langgraph_node: str
    additional_info: Optional[Dict[str, Any]]

class MessageChunk(TypedDict):
    """메시지 모드의 청크 타입 정의"""
    node: str
    content: Union[str, BaseMessage, Dict[str, Any], List[Any]]
    metadata: StreamMetadata
    response_metadata: Optional[Dict[str, Any]]


class UpdateChunk(TypedDict):
    """업데이트 모드의 청크 타입 정의"""
    node: str
    content: Any
    namespace: List[str]

StreamResult = Union[MessageChunk, UpdateChunk]

# 스트리밍 설정 타입
class StreamConfig(TypedDict, total=False):
    """스트리밍 설정 타입 정의"""
    stream_mode: str  # "messages" 또는 "updates"
    include_subgraphs: bool
    node_names: List[str] 