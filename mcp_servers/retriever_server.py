from mcp.server.fastmcp import FastMCP
from typing import List
from langchain.schema import Document

mcp = FastMCP("RetrieverService")

DOCUMENTS = [
    Document(page_content="LangChain은 LLM 애플리케이션 개발을 쉽게 해줍니다.", metadata={"source": "langchain_docs"}),
    Document(page_content="MCP는 모델 간 컨텍스트를 관리하기 위한 프로토콜입니다.", metadata={"source": "mcp_docs"}),
    Document(page_content="FastAPI는 빠른 Python 웹 프레임워크입니다.", metadata={"source": "fastapi_docs"}),
]


@mcp.tool()
def retrieve_documents(query: str, top_k: int = 3) -> List[Document]:
    """Retrieve documents from the knowledge base.
    
    Args:
        query (str): 검색할 쿼리 문자열
        top_k (int, optional): 반환할 최대 문서 수. 기본값은 3입니다.
        
    Returns:
        List[Document]: 쿼리와 관련된 문서 목록
    """

    query = query.lower()
    scored_docs = []
    for doc in DOCUMENTS:
        score = sum(1 for word in query.split() if word in doc.page_content.lower())
        scored_docs.append((score, doc))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    documents = [doc for score, doc in scored_docs if score > 0][:top_k]

    return {
        "documents": documents
    }


def main():
    """서버 실행을 위한 메인 함수"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Retriever MCP Server')
    parser.add_argument(
        '--transport',
        choices=['stdio', 'sse'],
        default='stdio',
        help='Transport method to use (stdio or sse)'
    )
    
    args = parser.parse_args()
    print(f"Starting MCP Server with transport: {args.transport}", file=sys.stderr)
    mcp.run(transport=args.transport) 

if __name__ == "__main__":
    main()