# MCP Servers

MCP(Model Context Protocol) 서버들의 모음입니다.

## 서버 종류

### Retriever Server
문서 검색을 위한 MCP 서버입니다. 주어진 쿼리와 관련된 문서를 검색하여 반환합니다.

실행 예시:
```bash
# 기본 실행 (stdio transport 사용)
python -m mcp_servers.retriever_server

# SSE transport 사용
python -m mcp_servers.retriever_server --transport sse
```

### Weather Server
날씨 정보를 제공하는 MCP 서버입니다. 지역 이름을 입력받아 해당 지역의 날씨 정보를 반환합니다.

실행 예시:
```bash
# 기본 실행 (stdio transport 사용)
python -m mcp_servers.weather_server

# SSE transport 사용
python -m mcp_servers.weather_server --transport sse
```

## 공통 옵션

모든 서버는 다음과 같은 공통 옵션을 지원합니다:

- `--transport`: 통신 방식 선택
  - `stdio`: 표준 입출력 사용 (기본값)
  - `sse`: Server-Sent Events 사용

## 도움말 보기

각 서버의 자세한 사용법을 확인하려면 `--help` 옵션을 사용하세요:

```bash
python -m mcp_servers.retriever_server --help
python -m mcp_servers.weather_server --help
```
