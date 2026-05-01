# PyMu PDF 변환기

PyMuPDF + PaddleOCR 기반의 PDF 변환·병합·OCR 웹 도구.

## 기능
- 이미지 / PDF / 오피스 문서(DOCX, XLSX, PPTX 등) → 단일 PDF 병합
- 드래그로 파일 순서 편집
- 파일별 90° / 180° / 270° 회전
- OCR 텍스트 레이어 추가 옵션 (기본 ON, 이미지 영역만 OCR)
- 기존 PDF에 대한 OCR 별도 실행

## 실행
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```
브라우저에서 http://127.0.0.1:8000 열기.

### 외부 의존: LibreOffice
DOCX/XLSX/PPTX 등 오피스 문서 변환은 LibreOffice (`soffice`)에 의존합니다.
- Windows: `winget install --id TheDocumentFoundation.LibreOffice`
- macOS: `brew install --cask libreoffice`
- Linux: 패키지 매니저로 `libreoffice` 설치
- 설치 후 자동 탐지됨. 비표준 경로면 `SOFFICE_PATH` 환경변수로 지정.

### 외부 의존: 한컴오피스 (HWP) 변환
HWP/HWPX 변환은 LibreOffice 의 H2Orestart 확장에 의존합니다 (선택 — HWP 안 쓰면 불필요).

1. 최신 release 의 `.oxt` 다운로드: https://github.com/ebandal/H2Orestart/releases
2. 시스템 단위 설치 (모든 사용자, 윈도우 서비스 계정 포함):
   ```powershell
   & "C:\Program Files\LibreOffice\program\unopkg.exe" add --shared --suppress-license H2Orestart.oxt
   ```
   관리자 권한 PowerShell 필요. 설치 확인:
   ```powershell
   & "C:\Program Files\LibreOffice\program\unopkg.exe" list --shared
   ```

### 외부 의존: AI 오탈자 교정 (선택)
PaddleOCR 결과의 한글 띄어쓰기·오탈자를 LLM 으로 교정하려면 ollama 와 한국어 모델이 필요합니다 (선택 — UI 의 "AI 오탈자교정" 토글로 켜고 끔).

1. ollama 설치: https://ollama.com/download
2. 한국어 모델 풀:
   ```bash
   ollama pull gemma4:e2b-it-q4_K_M
   ```
   (정확도 우선. 더 빠른 EXAONE 3.5 2.4B 도 옵션이지만 오탈자 교정은 약함)
3. ollama 데몬이 `localhost:11434` 에 떠있으면 자동 사용.
4. 모델 위치를 다른 디스크로 두려면 사용자 환경변수 `OLLAMA_MODELS` 설정 후 ollama 재시작.

ollama 가 없거나 모델이 없어도 OCR 자체는 정상 동작 (LLM 교정만 자동 fallback).

### 윈도우 서비스로 운영 (선택)
WinSW 를 이용해 부팅 시 자동 시작하는 윈도우 서비스로 운영 가능합니다.

1. WinSW 다운로드 (MIT, 단일 .exe): https://github.com/winsw/winsw/releases
   - `WinSW-x64.exe` 또는 `WinSW-net461.exe` 받아서 프로젝트 루트에 `pymu-svc.exe` 로 이름 변경
2. `pymu-svc.xml.example` 을 `pymu-svc.xml` 으로 복사 후 도메인을 본인 것으로 변경.
3. 관리자 권한 PowerShell 에서:
   ```powershell
   .\pymu-svc.exe install
   .\pymu-svc.exe start
   ```

### SSL 인증서 위치
프로덕션 모드 (`pymu-svc.xml`, `run_pymu.ps1`) 는 SSL 인증서를 **프로젝트 외부 형제 디렉토리** `../ssl/` 에서 읽습니다. 이는 인증서를 다른 서비스와 공유하기 쉽고, 실수로 git 에 커밋되는 것을 막기 위함입니다.

```
project-root/
├── ssl/                                    ← 외부 (git 관리 X)
│   ├── <your-domain>-key.pem
│   └── <your-domain>-chain.pem
└── pymu/                                   ← 본 프로젝트
    ├── pymu-svc.xml                        ← .gitignore (사용자별)
    └── pymu-svc.xml.example                ← commit 대상
```

Let's Encrypt 인증서 사용 시 `fullchain.pem` 을 `<도메인>-chain.pem` 으로, `privkey.pem` 을 `<도메인>-key.pem` 으로 이름 변경해 두세요.

### 환경변수
- `PYMU_SOURCE_URL` — 공개 배포 시 GitHub 저장소 URL. 푸터 / 라이선스 페이지의 "Source code" 링크에 사용.
- `SOFFICE_PATH` — LibreOffice CLI 경로 명시.
- `PYMU_OCR_THREADS` — OCR/BLAS 스레드 수 캡. 기본 `os.cpu_count() * 0.7` (소수점 버림). `0`으로 두면 캡 해제.

## 구성
- `backend/main.py` — FastAPI 엔트리, 정적 파일 서빙, `/api/info`
- `backend/services/converter.py` — 이미지/PDF/오피스 → PDF 병합 (PyMuPDF)
- `backend/services/office.py` — LibreOffice headless로 오피스 문서 → PDF
- `backend/services/ocr.py` — 이미지 영역에 한해 invisible OCR 텍스트 레이어 (PaddleOCR + PyMuPDF)
- `frontend/` — 정적 HTML/CSS/JS (SortableJS 사용)

## 메모
- 첫 OCR 호출 시 PaddleOCR이 모델 가중치(수백 MB)를 자동 다운로드.
- 한국어 모델은 영어도 함께 인식.
- 업로드 한도: 요청당 200 MB.

## 라이선스

이 프로젝트는 **GNU Affero General Public License v3.0 (AGPL-3.0)** 으로 배포됩니다.
이는 의존하는 PyMuPDF가 AGPL-3.0이기 때문이며, AGPL의 §13에 따라 네트워크
사용자에게 소스코드를 제공할 의무가 있습니다.

- 라이선스 전문: [LICENSE](LICENSE)
- 서드파티 고지: [NOTICE.md](NOTICE.md)
- 사용자용 라이선스 페이지: `/licenses.html`

광고 게재나 무료/유료 서비스 운영은 AGPL과 호환되지만, 소스 비공개는 불가합니다.
공개 배포 시 `PYMU_SOURCE_URL` 환경변수에 저장소 URL을 지정해 푸터에 표시되도록
하세요.
