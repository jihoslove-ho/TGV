================================================================
  TGV PSL Inspector v4.0 — EXE 빌드 방법
  SEIL TECHNO CORP.
================================================================

[준비물]
  - Python 3.9 이상 (현재 PC에 설치되어 있음 확인)
  - 인터넷 연결 (PyInstaller 자동 설치)

[빌드 순서]

  1. 아래 4개 파일을 같은 폴더에 넣으세요:
       tgv_v4.py                  ← 메인 소스 코드
       tgv_psl_inspector.spec     ← 빌드 설정
       version_info.txt           ← 버전 정보
       build.bat                  ← 빌드 실행 파일

  2. build.bat 를 더블클릭하여 실행

  3. 완료 후 생성되는 파일:
       dist\TGV_PSL_Inspector\
           TGV_PSL_Inspector.exe  ← 실행 파일
           (기타 DLL 파일들)

  4. dist\TGV_PSL_Inspector\ 폴더 전체를 현장 PC에 복사

[현장 PC 요구사항]
  - Windows 10 / 11 (64비트)
  - Python 설치 불필요
  - 별도 라이브러리 설치 불필요

[설정 파일 저장 위치]
  - Calibration 및 레시피 데이터:
    C:\Users\사용자명\AppData\Roaming\TGV_PSL_Inspector\settings.json
  - 프로그램 재설치 후에도 설정이 유지됩니다

[주의사항]
  - 빌드는 Windows PC에서만 가능합니다
  - 빌드 시간: 약 2~5분 소요
  - dist 폴더 크기: 약 300~500 MB (정상)
================================================================
