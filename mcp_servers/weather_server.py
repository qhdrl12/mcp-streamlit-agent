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

def main():
    """서버 실행을 위한 메인 함수"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Weather MCP Server')
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
