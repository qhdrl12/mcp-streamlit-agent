#!/usr/bin/env python
"""
간단한 Weather MCP 서버 구현.
지역에 대한 날씨 질문이 오면 항상 '맑음'으로 응답합니다.
"""
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any

# FastMCP 인스턴스 생성
mcp = FastMCP("WeatherService")

@mcp.tool()
def get_weather(location: str) -> Dict[str, Any]:
    """Get weather forecast for a location.
    
    Args:
        location (str): 날씨를 조회할 지역 이름
        
    Returns:
        Dict[str, Any]: 날씨 정보
    """
    return {
        "location": location,
        "weather": "맑음",
        "temperature": "23°C",
        "humidity": "45%",
        "wind": "5m/s"
    }

if __name__ == "__main__":
    import sys
    
    # 명령행 인수에 따라 transport 설정
    transport = "stdio"  # 기본값
    
    if len(sys.argv) > 1:
        transport = sys.argv[1]
    
    print(f"Starting Weather MCP Server with transport: {transport}", file=sys.stderr)
    
    if transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio") 
