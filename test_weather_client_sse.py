#!/usr/bin/env python
"""
Weather MCP 서버를 테스트하기 위한 간단한 클라이언트 스크립트
"""
import asyncio
import os
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    """
    메인 함수
    """
    server_config = {
        "weather": {
            "url": "http://localhost:8001/sse",
            "transport": "sse",
        }
    }
    
    # 연결 및 도구 사용
    print("MCP Weather 서버에 연결 중...")
    async with MultiServerMCPClient(server_config) as client:
        print("연결 성공! 사용 가능한 도구:")
        tools = client.get_tools()
        
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        
        print("\n테스트 실행:")
        
        # get_weather 도구 찾기
        weather_tool = None
        for tool in tools:
            if tool.name == "get_weather":
                weather_tool = tool
                break
        
        if not weather_tool:
            print("Error: get_weather 도구를 찾을 수 없습니다!")
            return
        
        # 도구 실행
        locations = ["서울", "부산", "뉴욕", "런던", "파리"]
        for location in locations:
            print(f"\n{location}의 날씨 조회:")
            result = await weather_tool.ainvoke({"location": location})
            print(f"결과: {result}")
    
if __name__ == "__main__":
    asyncio.run(main()) 