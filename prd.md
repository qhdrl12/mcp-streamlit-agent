프로젝트 PRD (v1.1): Streamlit MCP Config & Chat Interface
1. 개요 (Overview)

본 프로젝트는 Streamlit을 사용하여 웹 기반 인터페이스를 구축하는 것을 목표로 합니다. 사용자는 이 인터페이스를 통해 MCP(Multi-Channel Platform 등) 시스템 연동에 필요한 JSON 설정을 관리할 수 있습니다. 설정 관리는 텍스트 영역에 직접 JSON을 입력/수정하고 '적용' 버튼을 통해 저장하는 방식으로 이루어집니다. 또한, 내장된 Streamlit Chat Interface를 통해 사용자는 선택한 LLM(Gemini, ChatGPT, Claude 등)과 상호작용할 수 있으며, 이 과정에서 '적용'된 MCP 설정이 LLM의 동작이나 MCP 어댑터 연동에 활용될 수 있습니다.

2. 목표 (Goals)

간편한 MCP 설정 관리: Streamlit UI 내 텍스트 영역을 통해 MCP JSON 설정을 쉽게 입력, 수정하고 '적용'할 수 있는 기능을 제공합니다. (필요시 여러 설정 파일 관리 기능 포함)
MCP 시스템 연동: '적용'된 JSON 설정을 기반으로, 정의된 'MCP Adapter'를 통해 실제 MCP 시스템과 연동하는 기능을 구현합니다.
LLM 기반 Chat Interface 제공: Streamlit Chat Interface를 통해 사용자가 선택한 LLM(Gemini, ChatGPT, Claude 등)과 자연스럽게 상호작용할 수 있는 환경을 제공합니다.
유연한 LLM 선택: 사용자가 Chat Interface에서 사용할 LLM을 UI 상에서 손쉽게 선택하고 변경할 수 있도록 지원합니다.
설정-챗봇 연계: (선택적) '적용'된 MCP 설정 정보를 Chat Interface의 LLM 컨텍스트나 MCP 어댑터 동작에 반영합니다.
3. 대상 사용자 (Target Audience)

MCP 시스템 연동 설정을 관리하고 테스트해야 하는 개발자.
MCP 설정 기반으로 LLM 상호작용 또는 시스템 테스트를 수행해야 하는 테스터/QA 엔지니어.
4. 주요 기능 (Key Features)

MCP 설정 관리:
(선택적) 여러 개의 명명된 MCP 설정 관리 기능 (목록 조회, 선택, 생성, 삭제).
선택된 (또는 단일) MCP 설정을 st.text_area에 표시 및 편집 기능.
JSON 형식 유효성 검사 기능 (입력/수정 시).
'적용' 버튼:
현재 텍스트 영역의 JSON 내용을 저장 (예: 파일 또는 세션).
(선택적) 저장된 설정을 기반으로 MCP 어댑터 로직 트리거.
MCP 어댑터 연동:
'적용' 버튼 클릭 시 또는 특정 Chat 명령어를 통해 트리거.
'적용'된 최신 MCP JSON 설정을 기반으로 동작.
어댑터 실행 상태 및 결과를 사용자에게 피드백 (예: st.status 또는 Chat 메시지).
LLM 통합 Chat Interface:
Streamlit Chat Interface (st.chat_input, st.chat_message, st.status 등) 활용.
UI 요소(예: st.selectbox)를 통해 Chat에 사용할 LLM(Gemini, ChatGPT, Claude 등) 선택 기능.
Langchain을 백엔드에서 활용하여 선택된 LLM 기반의 대화 로직 처리.
LLM API 키 또는 인증 정보 관리 (Streamlit secrets 또는 환경변수 활용).
사용자 입력(Chat)을 받아 처리 후, LLM 응답 및 어댑터 결과 등을 Chat 메시지로 표시.
5. 사용자 인터페이스 (UI/UX) 고려사항

단순성: 핵심 기능(JSON 편집, 적용, LLM 선택, 채팅)에 쉽게 접근할 수 있도록 직관적인 레이아웃 구성.
JSON 편집: st.text_area를 사용하여 JSON 입력 및 수정 편의성 제공.
Chat: 표준 Streamlit Chat Interface 패턴을 따라 사용자 경험 일관성 유지.
피드백: JSON 유효성, '적용' 결과, 어댑터 상태, LLM 처리 상태 등을 명확하게 사용자에게 전달.
6. 기술 요구사항 및 아키텍처 (Technical Requirements & Architecture)

UI 프레임워크: Streamlit (Chat Interface 컴포넌트 포함)
LLM 오케스트레이션/연동: Langchain
LLM 연동 라이브러리: google-generativeai, openai, anthropic 등
MCP 어댑터: 별도의 Python 모듈/클래스 (JSON 설정을 받아 특정 동작 수행)
설정 데이터 저장소: 로컬 파일 시스템 (JSON 파일), Streamlit Session State 등 (관리할 설정 개수에 따라 결정)
API 키 관리: Streamlit Secrets 또는 환경 변수
개발 언어: Python 3.x
7. 데이터 관리 (Data Management)

MCP JSON 설정: 사용자가 입력한 JSON 텍스트의 저장 및 로드. 형식 유효성 검증.
LLM API 키: 안전한 방식(Secrets, Env Vars)으로 관리.
Chat History: Streamlit Session State 등을 활용하여 대화 기록 관리.
8. 비기능적 요구사항 (Non-Functional Requirements)

사용성: 대상 사용자가 쉽게 설정하고 채팅할 수 있어야 함.
모듈성: MCP 어댑터, LLM 핸들러(Langchain 부분) 등 분리 고려.
확장성: 새로운 LLM 종류 추가 용이성.
보안: API 키 보호.
9. 향후 고려사항 (Future Considerations)

더 많은 MCP 설정 파일 관리 기능 (Import/Export 등).
MCP 설정 템플릿 제공.
고도화된 MCP 어댑터 기능 및 상태 시각화.
Chat 명령어 기반의 MCP 어댑터 제어.
10. 성공 지표 (Success Metrics)

사용자가 텍스트 영역을 통해 MCP JSON을 입력/수정하고 '적용'할 수 있음.
'적용'된 설정을 기반으로 MCP 어댑터가 의도대로 동작하고 결과 피드백을 받을 수 있음.
사용자가 UI에서 LLM을 선택하고 Chat Interface를 통해 원활하게 상호작용할 수 있음.
대상 사용자로부터 설정 관리 및 LLM 연동 테스트가 간편해졌다는 피드백 확보.
11. 가정 및 질문 사항 (Assumptions & Open Questions)

가정: 사용자는 관리할 MCP JSON의 기본 구조를 알고 있다.
가정: 'MCP Adapter'는 '적용'된 JSON을 인자로 받아 호출 가능한 Python 함수/메소드 형태이다.
질문: '적용' 버튼 클릭 시, MCP 어댑터는 항상 실행되어야 하는가, 아니면 특정 조건/명령 하에 실행되어야 하는가?
질문: 여러 개의 MCP 설정을 이름 등으로 구분하여 관리할 필요가 있는가? (아니면 항상 하나의 활성 설정만 관리하는가?)
질문: '적용'된 MCP JSON 설정은 LLM의 응답 생성(예: 시스템 프롬프트, 컨텍스트 주입)에 어떤 영향을 주어야 하는가? 아니면 MCP 어댑터 실행에만 사용되는가?
