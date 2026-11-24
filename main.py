import json
import random
import os
import re
import time
from datetime import datetime
import requests
from google import genai
from google.genai import types

# ============================================================
# 설정값 (Session-Album : Natural Easy-Listening All-Genre)
# ============================================================
SYSTEM_INSTRUCTION_FILE = 'data/system_instruction.txt'
TIMES_FILE = 'data/times.json'
PLACES_FILE = 'data/places.json'

MAX_OUTPUT_TOKENS = 16384   # 사고(thinking) 토큰 포함 모델 대비 넉넉하게 설정
TEMPERATURE = 1.0           # Gemini 3.x 이상 계열 권장 기본값
MAX_RETRIES = 3
RETRY_WAIT_SEC = 10

# 글자 수 및 보컬 호흡(음절 밀도) 검증 기준
# - Suno Style 한도: 1000자 미만 (권장 280~500자)
# - Suno Lyrics 한도: 5000자 미만 (억지 늘리기 방지를 위해 하한 450자)
SUNO_WARN_MIN, SUNO_MAX = 200, 999
LYRICS_MIN, LYRICS_MAX = 450, 4500

# 한국어·영어 혼용 가사의 실제 발성 음절 기준 (박자 급발진 방지 핵심)
SYLLABLES_MIN = 120          # 곡 전체 최소 발성 음절
SYLLABLES_MAX = 520          # 곡 전체 최대 발성 음절 (초과 시 Suno 급발진 위험)
LINE_SYLLABLE_SOFT_MAX = 13  # 한 줄당 권장 상한 음절 (초과 시 과밀 줄로 집계)
OVERCROWDED_LINES_FATAL = 4  # 과밀 줄이 4줄 이상이면 치명적 오류로 재생성
AVG_LINE_SYLLABLE_FATAL = 11.5

# 프로젝트 API 할당량(텍스트 출력 모델, RPD > 0) 기준 우선순위
PREFERRED_MODELS = [
    # 1군: 고품질 메인 (각 RPD 20, 총 100회)
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-4-hydrogen",
    "gemini-3-flash",
    "gemini-2.5-flash",
    # 2군: 대용량 백업 (각 RPD 500, 총 1,000회)
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    # 3군: 보조 백업 (RPD 20) 및 자동 별칭
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

# CLEAN_LYRICS는 파이썬 후처리로 최종 생성 (프롬프트에서 제외해 토큰 절약)
MARKERS_BASE = [
    "DETAIL", "PURPOSE", "SUNO", "EXCLUDE_STYLES", "VOCAL",
    "IMAGE", "LYRICS", "CLEAN_LYRICS", "TAG", "UPLOAD",
]

REQUIRED_SECTIONS = [
    "Intro", "Verse 1", "Pre-Chorus 1", "Chorus 1", "Verse 2", "Pre-Chorus 2",
    "Chorus 2", "Bridge", "Instrumental Solo", "Chorus 3", "Outro",
]
REQUIRED_FIELDS = ["detail", "purpose", "suno", "vocal", "lyrics", "tag", "upload"]

# 빠른 비트 위에서도 보컬이 급해지지 않도록 하프타임 프레이징을 권장할 고BPM/고밀도 장르
HIGH_BPM_GENRES = {
    "Liquid Drum & Bass", "UK Garage", "Jersey Club", "Baltimore Club",
    "Favela Funk", "Miami Bass", "EDM", "House", "Hyperpop", "Dubstep", "Trap",
}

# 랩-싱잉 예외가 자연스럽게 어울리는 장르
RAP_FRIENDLY_GENRES = {
    "Hip Hop", "Old-school Hip Hop", "Trap", "Baltimore Club", "Jersey Club",
}

# 서로 정체성이 과하게 충돌해 부자연스러워지는 조합 (순서 무관, 전 장르 호환)
UNWANTED_PAIRS = {
    # Session-Album 16종 풀 충돌 조합
    frozenset(("Favela Funk", "Liquid Drum & Bass")),
    frozenset(("Baltimore Club", "City Pop")),
    frozenset(("Jersey Club", "City Pop")),
    frozenset(("Jersey Club", "Old-school Hip Hop")),
    frozenset(("Miami Bass", "Contemporary R&B")),
    frozenset(("Moombahton", "New Jack Swing")),
    frozenset(("Favela Funk", "Old-school Hip Hop")),
    frozenset(("Favela Funk", "City Pop")),
    # 확장 31종 풀 충돌 조합
    frozenset(("Folk", "Dubstep")),
    frozenset(("Country", "Hyperpop")),
    frozenset(("Bossa Nova", "Industrial")),
    frozenset(("Operatic Pop", "Jersey Club")),
    frozenset(("Blues", "EDM")),
    frozenset(("Ballad", "Afrobeat")),
    frozenset(("Jazz", "Hyperpop")),
    frozenset(("Reggae", "Industrial")),
    frozenset(("Lo-fi", "Trap")),
    frozenset(("Operatic Pop", "Trap")),
    frozenset(("Folk", "Hyperpop")),
    frozenset(("Country", "Industrial")),
    frozenset(("Bossa Nova", "Dubstep")),
    frozenset(("Ballad", "Jersey Club")),
    frozenset(("Blues", "Hyperpop")),
    frozenset(("Lo-fi", "Dubstep")),
    frozenset(("Soul", "Industrial")),
    frozenset(("Jazz", "Dubstep")),
    frozenset(("Reggae", "Hyperpop")),
    frozenset(("Operatic Pop", "UK Garage")),
    frozenset(("Country", "Deep House")),
    frozenset(("Folk", "Jersey Club")),
    frozenset(("Bossa Nova", "Trap")),
    frozenset(("Ballad", "Industrial")),
    frozenset(("Indie", "EDM")),
}

_MODEL_CACHE = None  # list_models() 결과 재사용


# ============================================================
# 데이터 로드 / 랜덤 선택
# ============================================================
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def clean_seed(text):
    return text.replace("_", " ").strip()


def get_random_item(data):
    if isinstance(data, dict):
        all_items = []
        for items_in_category in data.values():
            all_items.extend(items_in_category)
        return clean_seed(random.choice(all_items))
    elif isinstance(data, list):
        return clean_seed(random.choice(data))
    return ""


def get_distinct_emotions(emotions1_data, emotions2_data):
    """emotions1과 emotions2에서 서로 다른 감정 키워드를 추출해요."""
    emo1 = get_random_item(emotions1_data)
    emo2 = get_random_item(emotions2_data)
    for _ in range(20):
        if emo1.lower() != emo2.lower():
            break
        emo2 = get_random_item(emotions2_data)
    return emo1, emo2


def is_unwanted_combination(genre1, genre2):
    if genre1 == genre2:
        return True
    return frozenset((genre1, genre2)) in UNWANTED_PAIRS


# ============================================================
# 시스템 프롬프트 조립
# ============================================================
def build_system_instruction(raw_text, genre1, genre2, rule1, rule2):
    """str.format 대신 replace를 사용해 본문에 { } 가 있어도 깨지지 않게 해요."""
    replacements = {
        "{selected_genre1}": genre1,
        "{selected_genre2}": genre2,
        "{selected_genre}": f"{genre1}, {genre2}",
        "{rule_for_genre1}": rule1,
        "{rule_for_genre2}": rule2,
    }
    text = raw_text
    for key, value in replacements.items():
        text = text.replace(key, value)

    leftovers = re.findall(r'\{(?:selected_|rule_for_)[a-z0-9_]*\}', text)
    if leftovers:
        print(f"⚠️ 치환되지 않은 자리표시자가 남아 있어요: {sorted(set(leftovers))}")
    if rule1 not in text or rule2 not in text:
        print("⚠️ 장르 룰이 시스템 프롬프트에 들어가지 않았어요. system_instruction.txt의 자리표시자를 확인하세요.")
    else:
        print("✅ 두 장르의 이지리스닝 룰이 시스템 프롬프트에 정상 주입되었습니다.")
    return text


def build_image_prompt(extracted):
    """###IMAGE### 섹션이 비었을 때 사용하는 자연스러운 기본 썸네일 프롬프트"""
    return f"""아래 첨부된 [곡 상세 정보]와 [기획 의도]를 바탕으로 유튜브 영상용 16:9 썸네일 이미지를 하나 생성해 줘.

[필수 적용 조건]
1. 화질 및 텍스트 배제: 자연스러운 필름 그레인 감성과 4K 선명도를 조화롭게 구현하고, 화면 내에 노래 제목·영단어·간판 글씨 등 어떠한 텍스트도 절대 넣지 마(no text, no typography).
2. 자연스러운 인물과 공간: 과장된 연출 대신 2025~2026년 K-Pop 앨범 커버처럼 담백하고 세련된 한국 20대 일상 무드로 표현해 줘. 자연광이나 은은한 실내/거리 조명을 활용해 인위적이지 않은 온도를 담아 줘.
3. 분위기 기반 비주얼 매칭: 곡의 시간대, 장소, 장르의 리듬감과 감정선을 색감 팔레트와 카메라 구도(자연스러운 여백과 심도)로 시각화해 줘.

[곡 상세 정보]
{extracted.get('detail', '')}

[기획 의도]
{extracted.get('purpose', '')}"""


# ============================================================
# 한영 혼용 발성 음절 추정 및 후처리 / 검증
# ============================================================
def estimate_english_syllables(word):
    """영어 단어 1개의 대략적인 발성 음절 수 추정"""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    # 묵음 e 처리 (단, 자음+le 끝말은 음절 유지: little, people 등)
    if w.endswith("e") and not (len(w) > 2 and w.endswith("le") and w[-3] not in "aeiouy"):
        w = w[:-1]
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups))


def estimate_line_syllables(line):
    """한 줄(한국어 + 영어 혼용)의 실제 발성 음절 수 추정"""
    # 괄호 안 짧은 추임새((Ooh), (Yeah) 등)는 보조 호흡이므로 제거 후 본문 기준 측정
    core = re.sub(r'\([^)]*\)', '', line).strip()
    if not core:
        return 0
    ko_count = len(re.findall(r'[가-힣]', core))
    en_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", core)
    en_count = sum(estimate_english_syllables(w) for w in en_words)
    return ko_count + en_count


def analyze_lyrics_pacing(clean_lyrics):
    """순수 가사의 박자 밀도(총 음절, 줄 수, 평균 음절, 과밀 줄 수)를 분석"""
    lines = [ln.strip() for ln in clean_lyrics.split('\n') if ln.strip()]
    counts = [estimate_line_syllables(ln) for ln in lines]
    counts = [c for c in counts if c > 0]
    if not counts:
        return {"line_count": 0, "total_syllables": 0, "avg_syllables": 0.0, "max_syllables": 0, "overcrowded": 0}
    overcrowded = sum(1 for c in counts if c > LINE_SYLLABLE_SOFT_MAX)
    return {
        "line_count": len(counts),
        "total_syllables": sum(counts),
        "avg_syllables": round(sum(counts) / len(counts), 2),
        "max_syllables": max(counts),
        "overcrowded": overcrowded,
    }


def build_clean_lyrics(raw_lyrics):
    """LYRICS에서 [...]와 <...> 태그를 제거해 순수 가사 생성"""
    cleaned = re.sub(r'\[[^\]]*\]', '', raw_lyrics)
    cleaned = re.sub(r'<[^>]*>', '', cleaned)
    lines = [re.sub(r'[ \t]{2,}', ' ', ln).strip() for ln in cleaned.split('\n')]
    cleaned = '\n'.join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def sanitize_lyrics_lines(lyrics):
    """가사 내 인위적인 끊김을 유발하는 표기(마크다운, 꺾쇠, 한국어 줄 중간 마침표) 보정"""
    lyrics = lyrics.replace("**", "")
    lyrics = re.sub(r'<([^>]+)>', r'[\1]', lyrics)

    cleaned_lines = []
    for line in lyrics.split('\n'):
        stripped = line.strip()
        # 메타 태그 줄은 그대로 유지
        if stripped.startswith('[') and stripped.endswith(']'):
            cleaned_lines.append(stripped)
            continue
        # 한국어 단어 사이에 들어간 단일 마침표(.)는 부자연스러운 끊김을 유발하므로 쉼표로 완화 (...은 유지)
        line = re.sub(r'(?<=[가-힣])\.(?!\.)(?=\s+[가-힣A-Za-z])', ',', line)
        cleaned_lines.append(line.rstrip())

    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def post_process_data(extracted):
    """사소한 포맷 오류를 파이썬에서 자동 보정"""
    # 1. DETAIL: 마크다운 굵게 제거
    if extracted.get("detail"):
        extracted["detail"] = extracted["detail"].replace("**", "").strip()

    # 2. LYRICS: 태그 기호 통일 및 인위적 끊김 보정
    extracted["lyrics"] = sanitize_lyrics_lines(extracted.get("lyrics", ""))

    # 3. CLEAN_LYRICS: LYRICS에서 태그를 제거해 100% 일관된 순수 가사 생성
    if extracted["lyrics"]:
        extracted["clean_lyrics"] = build_clean_lyrics(extracted["lyrics"])

    # 4. TAG: 중복 제거 및 최대 30개 정리 (# 형식 유지)
    raw_tag = extracted.get("tag", "")
    tags, seen = [], set()
    for t in raw_tag.split(","):
        t = t.strip()
        key = t.lstrip("#").strip().lower()
        if key and key not in seen:
            seen.add(key)
            tags.append(t)
    extracted["tag"] = ", ".join(tags[:30])

    # 5. IMAGE: ###IMAGE### 섹션이 비었을 때만 기본 프롬프트 사용
    if not extracted.get("image", "").strip():
        extracted["image"] = build_image_prompt(extracted)

    return extracted


def has_section(lyrics, name):
    """[Verse 1], [Verse 1: ...] 형태 및 [Rhodes Solo] 같은 솔로 변형 허용"""
    if name == "Instrumental Solo":
        return re.search(r'\[[^\]]*\bsolo\b[^\]]*\]', lyrics, re.IGNORECASE) is not None
    return re.search(r'\[\s*' + re.escape(name) + r'\b', lyrics, re.IGNORECASE) is not None


def validate_result(extracted):
    """
    반환: (fatal_issues, warnings, pacing_stats)
    - fatal_issues: 재생성이 필요한 치명적 오류
    - warnings: 로그 및 Notion 본문 메모만 남기고 재시도하지 않는 경미한 이슈
    """
    fatal, warnings = [], []

    lyrics = extracted.get("lyrics", "")
    clean_lyrics = extracted.get("clean_lyrics", "")
    suno_len = len(extracted.get("suno", ""))
    lyrics_len = len(lyrics)
    pacing = analyze_lyrics_pacing(clean_lyrics)

    # ── 치명적 오류 ──
    empty_fields = [f for f in REQUIRED_FIELDS if not extracted.get(f, "").strip()]
    if empty_fields:
        fatal.append(f"비어 있는 필드(잘림 의심): {', '.join(empty_fields)}")

    if suno_len > SUNO_MAX:
        fatal.append(f"Style(Suno) 1000자 초과 ({suno_len}자)")
    if lyrics_len and lyrics_len < LYRICS_MIN:
        fatal.append(f"가사(Lyrics) 너무 짧음 ({lyrics_len}자)")
    if lyrics_len > LYRICS_MAX:
        fatal.append(f"가사(Lyrics) 5000자 초과 위험 ({lyrics_len}자)")

    if lyrics:
        missing = [sec for sec in REQUIRED_SECTIONS if not has_section(lyrics, sec)]
        if missing:
            fatal.append(f"섹션 누락: {', '.join(missing)}")

    # 박자 급발진 방지: 실제 발성 음절 밀도 검증 (한영 혼용 지원)
    if clean_lyrics:
        if pacing["total_syllables"] < SYLLABLES_MIN:
            fatal.append(f"순수 가사 음절 너무 적음 ({pacing['total_syllables']}음절)")
        elif pacing["total_syllables"] > SYLLABLES_MAX:
            fatal.append(f"순수 가사 음절 과다 - 박자 밀림 위험 ({pacing['total_syllables']}음절 > {SYLLABLES_MAX})")

        if pacing["overcrowded"] >= OVERCROWDED_LINES_FATAL:
            fatal.append(f"과밀 줄({LINE_SYLLABLE_SOFT_MAX}음절 초과) {pacing['overcrowded']}줄 발생")
        if pacing["avg_syllables"] > AVG_LINE_SYLLABLE_FATAL:
            fatal.append(f"줄당 평균 음절 과다 ({pacing['avg_syllables']}음절)")

    # ── 경고 (재시도 없음) ──
    if 0 < suno_len < SUNO_WARN_MIN:
        warnings.append(f"Style(Suno) 다소 짧음 ({suno_len}자)")

    if 0 < pacing["overcrowded"] < OVERCROWDED_LINES_FATAL:
        warnings.append(f"약간 긴 줄({LINE_SYLLABLE_SOFT_MAX}음절 초과) {pacing['overcrowded']}줄 포함 (최대 {pacing['max_syllables']}음절)")

    kyeom_count = len(re.findall(r'(?:^|\s)겸(?:\s|$|[,.!?~])', clean_lyrics))
    if clean_lyrics and kyeom_count != 1:
        warnings.append(f"이스터 에그 '~겸' 출현 {kyeom_count}회 (목표 1회)")

    tag_count = len([t for t in extracted.get("tag", "").split(",") if t.strip()])
    if extracted.get("tag", "").strip() and tag_count != 30:
        warnings.append(f"해시태그 {tag_count}개 (목표 30개)")

    return fatal, warnings, pacing


# ============================================================
# Gemini 호출 및 파싱 (google-genai SDK)
# ============================================================
def get_available_models(client):
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = [
            m.name for m in client.models.list()
            if m.supported_actions and "generateContent" in m.supported_actions
        ]
        print(f"🧐 [참고] 사용 가능한 모델 총 {len(_MODEL_CACHE)}개 확인 완료")
    return _MODEL_CACHE


def find_model_id(available_models, base_name):
    """정확한 모델명 우선 매칭, 없으면 -preview 버전 매칭 (TTS/Live/Image 제외)"""
    candidates = [base_name, f"{base_name}-preview"]
    for cand in candidates:
        for m in available_models:
            if m == cand or m.endswith(f"/{cand}"):
                return m
    return None


def parse_sections(text):
    for m in MARKERS_BASE:
        text = re.sub(r'[*_]*#+\s*' + m + r'\s*#*[*_]*', f'###{m}###', text, flags=re.IGNORECASE)

    markers = [f"###{m}###" for m in MARKERS_BASE]
    extracted = {m.lower(): "" for m in MARKERS_BASE}

    for marker in markers:
        if marker in text:
            part = text.split(marker, 1)[1]
            min_idx = len(part)
            for other_marker in markers:
                if other_marker != marker:
                    idx = part.find(other_marker)
                    if idx != -1 and idx < min_idx:
                        min_idx = idx
            key = marker.replace("#", "").lower()
            extracted[key] = part[:min_idx].strip()

    return post_process_data(extracted)


def generate_lyrics_with_gemini(system_instruction, user_prompt):
    """
    반환: (결과 dict, status)
    status: "ok" | "parse_error" | "api_error" | "no_model" | "no_key"
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("🚨 GEMINI_API_KEY가 설정되지 않았습니다.")
        return {}, "no_key"

    client = genai.Client(api_key=api_key)

    try:
        available_models = get_available_models(client)
    except Exception as api_e:
        print(f"❌ API 모델 리스트 조회 실패: {type(api_e).__name__}")
        return {}, "api_error"

    text = ""
    tried_any = False

    for model_name in PREFERRED_MODELS:
        target = find_model_id(available_models, model_name)
        if not target:
            continue

        tried_any = True
        try:
            print(f"🚀 [{target}] 모델로 생성을 시도합니다...")
            response = client.models.generate_content(
                model=target,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=TEMPERATURE,
                ),
            )

            finish_reason = ""
            try:
                finish_reason = str(response.candidates[0].finish_reason)
            except Exception:
                pass

            text = response.text or ""
            if not text.strip():
                print(f"⚠️ {target} 빈 응답 (finish_reason={finish_reason or '알 수 없음'}) -> 다음 모델로 전환합니다.")
                continue

            if "MAX_TOKENS" in finish_reason:
                print(f"⚠️ {target} 출력이 토큰 한도에서 잘렸을 수 있습니다 (finish_reason={finish_reason}). 검증 단계에서 확인합니다.")

            print(f"✅ {target} 생성 성공!")
            break

        except Exception as e:
            print(f"⚠️ {target} 실패 ({str(e)[:120]}) -> 다음 모델로 전환합니다.")
            text = ""
            time.sleep(2)

    if not text.strip():
        if not tried_any:
            sample = ", ".join(available_models[:10])
            print(f"❌ PREFERRED_MODELS 중 사용 가능한 모델이 없습니다. (목록 일부: {sample} ...)")
            return {}, "no_model"
        print("❌ 시도한 모든 모델이 할당량 초과 또는 오류로 실패했습니다.")
        return {}, "api_error"

    try:
        extracted = parse_sections(text)

        print("\n[4] 파싱된 섹션별 글자 수:")
        for k, v in extracted.items():
            print(f" - {k}: {len(v)}자")

        return extracted, "ok"

    except Exception as e:
        print(f"❌ Gemini 데이터 파싱 에러: {e}")
        return {}, "parse_error"


# ============================================================
# Notion 저장 (Session-Album 스키마 유지)
# ============================================================
def get_chunks(text):
    if not text:
        return [{"text": {"content": " "}}]
    return [{"text": {"content": text[i:i+2000]}} for i in range(0, len(text), 2000)]


def save_to_notion(date_str, genre, prompt, data_dict, notes=None):
    notion_token = os.environ.get("NOTION_TOKEN")
    database_id = os.environ.get("NOTION_DATABASE_ID")

    if not notion_token or not database_id or not data_dict.get("lyrics", "").strip():
        print("❌ 저장할 가사(LYRICS) 데이터가 비어있거나 Notion 설정이 없어 Notion 호출을 취소합니다.")
        return

    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    children_blocks = [{
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": "🎶 Gemini 생성 가사 및 곡 구성"}}]}
    }]

    for para in data_dict["lyrics"].split('\n\n'):
        para = para.strip()
        if not para:
            continue
        while len(para) > 2000:
            split_idx = para.rfind('\n', 0, 2000)
            if split_idx == -1:
                split_idx = para.rfind(' ', 0, 2000)
            if split_idx == -1:
                split_idx = 2000
            chunk = para[:split_idx].strip()
            children_blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": chunk}}]}
            })
            para = para[split_idx:].strip()
        if para:
            children_blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": para}}]}
            })

    children_blocks.append({"object": "block", "type": "divider", "divider": {}})
    children_blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"text": {"content": data_dict.get("tag", "")[:2000] or " "}}]}
    })

    if notes:
        children_blocks.append({"object": "block", "type": "divider", "divider": {}})
        children_blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": ("⚠️ 검증 메모: " + " / ".join(notes))[:2000]}}]}
        })

    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            "Title": {"title": [{"text": {"content": f"{date_str} ({genre})"}}]},
            "Generated Prompt": {"rich_text": [{"text": {"content": prompt[:2000]}}]},
            "Detail": {"rich_text": [{"text": {"content": data_dict.get("detail", "")[:2000]}}]},
            "Purpose": {"rich_text": [{"text": {"content": data_dict.get("purpose", "")[:2000]}}]},
            "Suno": {"rich_text": [{"text": {"content": data_dict.get("suno", "")[:2000]}}]},
            "Exclude_styles": {"rich_text": [{"text": {"content": data_dict.get("exclude_styles", "")[:2000]}}]},
            "Image": {"rich_text": [{"text": {"content": data_dict.get("image", "")[:2000]}}]},
            "Vocal": {"rich_text": [{"text": {"content": data_dict.get("vocal", "")[:2000]}}]},
            "Lyrics": {"rich_text": get_chunks(data_dict.get("clean_lyrics", ""))},
            "E_Lyrics": {"rich_text": get_chunks(data_dict.get("lyrics", ""))},
            "Tag": {"rich_text": [{"text": {"content": data_dict.get("tag", "")[:2000]}}]},
            "Genre": {"rich_text": [{"text": {"content": genre}}]},
            "Upload": {"rich_text": [{"text": {"content": data_dict.get("upload", "")[:2000]}}]}
        },
        "children": children_blocks
    }

    response = requests.post('https://api.notion.com/v1/pages', headers=headers, json=payload, timeout=30)

    print(f"📊 [결과] HTTP 상태 코드: {response.status_code}")
    if response.status_code == 200:
        print("✅ Notion 저장 성공! 모든 데이터가 들어갔습니다.")
    else:
        print(f"❌ Notion 저장 실패! 상세 사유: {response.text}")


# ============================================================
# 메인
# ============================================================
def main():
    try:
        genres1 = load_data('data/genres1.json')
        genres2 = load_data('data/genres2.json')
        times = load_data(TIMES_FILE)
        emotions1 = load_data('data/emotions1.json')
        actions = load_data('data/actions.json')
        places = load_data(PLACES_FILE)
        emotions2 = load_data('data/emotions2.json')
        genre_rules = load_data('data/genre_rules.json')

        raw_system_instruction = load_text(SYSTEM_INSTRUCTION_FILE)
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return

    max_pick_retries = 100
    selected_genre1 = selected_genre2 = None

    for _ in range(max_pick_retries):
        g1 = get_random_item(genres1)
        g2 = get_random_item(genres2)
        if not is_unwanted_combination(g1, g2):
            selected_genre1, selected_genre2 = g1, g2
            break
        print(f"⚠️ 원하지 않는 조합 발생 ({g1}, {g2}) -> 다시 뽑습니다.")

    if not selected_genre1:
        print("❌ 유효한 장르 조합을 찾는 데 실패했습니다.")
        return

    rule_for_genre1 = genre_rules.get(selected_genre1)
    rule_for_genre2 = genre_rules.get(selected_genre2)
    if not rule_for_genre1:
        print(f"⚠️ '{selected_genre1}' 장르 룰이 genre_rules.json에 없어요. 기본 문구로 대체합니다.")
        rule_for_genre1 = f"- {selected_genre1}의 리듬 개성을 살리되, 귀가 편안하고 자연스러운 이지리스닝 질감으로 편곡해."
    if not rule_for_genre2:
        print(f"⚠️ '{selected_genre2}' 장르 룰이 genre_rules.json에 없어요. 기본 문구로 대체합니다.")
        rule_for_genre2 = f"- {selected_genre2}의 리듬 개성을 살리되, 귀가 편안하고 자연스러운 이지리스닝 질감으로 편곡해."

    selected_genre = f"{selected_genre1}, {selected_genre2}"
    selected_time = get_random_item(times)
    selected_emotion1, selected_emotion2 = get_distinct_emotions(emotions1, emotions2)
    selected_action = get_random_item(actions)
    selected_place = get_random_item(places)

    now = datetime.now()
    current_date = now.strftime("%Y년 %m월 %d일")

    # 장르 특성에 따른 동적 리듬 안정 가이드
    pacing_notes = []
    if selected_genre1 in HIGH_BPM_GENRES or selected_genre2 in HIGH_BPM_GENRES:
        pacing_notes.append(
            "빠른 템포의 장르가 포함되어 있으므로, 보컬은 반드시 하프타임(half-time vocal phrasing)으로 여유롭게 호흡하며 박자를 쫓아가지 않도록 설계할 것."
        )
    if selected_genre1 in RAP_FRIENDLY_GENRES or selected_genre2 in RAP_FRIENDLY_GENRES:
        pacing_notes.append(
            "랩이나 리드미컬한 파트를 넣더라도 속사포/더블타임은 절대 금지. 한 줄 8~11음절 안에서 쉼표(,)로 반마디 호흡을 나누는 느긋한 멜로딕 랩-싱잉으로 작성할 것."
        )
    pacing_guidance = "\n- 리듬 안정 가이드: " + " ".join(pacing_notes) if pacing_notes else ""

    system_instruction = build_system_instruction(
        raw_system_instruction,
        selected_genre1, selected_genre2,
        rule_for_genre1, rule_for_genre2
    )

    user_prompt = f"""<Current_Status>
- 진행 단계: 2025~2026년 K-Pop 이지리스닝 감성의 자연스러운 오리지널 트랙 기획 및 최종 가사 완성
- 핵심 목표: 어떤 장르 조합이든 반복 재생해도 귀가 편안하고, 보컬·랩이 박자에 밀리지 않으며, 인위적이지 않은 자연스러운 흐름 구현
</Current_Status>

<Brainstorming_Seed>
- 장르 조합: {selected_genre} (메인: {selected_genre1} / 서브: {selected_genre2})
- 기준 날짜 및 시간: {current_date}, {selected_time}
- 장소: {selected_place}
- 행동·상태: {selected_action}
- 감정 흐름: Verse({selected_emotion1}) → Chorus({selected_emotion2}){pacing_guidance}
</Brainstorming_Seed>

<Action_Steps>
위의 <Brainstorming_Seed>를 바탕으로 다음 단계를 거쳐 작업을 수행해 줘.

1) [내부 구상]: 위 시드를 엮어 과장되지 않고 현실감 있는 장면과 감정 흐름을 설계해. 2025~2026년 K-Pop(자연스러운 대화체 보컬, 편안한 그루브, 매끄러운 한영 혼용)의 감각을 반영해, 한국어 문장과 자연스럽게 어우러지는 영어 문장/구절을 곡의 흐름상 꼭 어울리는 위치(후렴 훅, 프리코러스 전환, 브릿지 여운 등)에 배치할 계획을 세워. (내부 추론은 출력하지 말 것)

2) [스타일 및 가사 최적화]:
   - ###SUNO### (Style of Music): 장르({selected_genre}), 템포(메인 장르 {selected_genre1} 기준), 따뜻하고 유기적인 악기 질감, 여유로운 공간감(unhurried groove, warm analog mix, breathing room)을 280~500자 영어 키워드로 작성할 것. 보컬 관련 키워드와 부정형 지시어는 절대 넣지 마.
   - ###EXCLUDE_STYLES### (제외할 스타일): rushed vocals, fast rap, double-time flow, crowded lyrics, shrill belting, robotic autotune, overcompressed mix 등 인위적이거나 귀가 피로해지는 방해 요소를 쉼표로 구분해 작성할 것.
   - ###LYRICS###:
     a) [박자·호흡 엄수]: Verse/Chorus는 반드시 4줄(2줄 + 빈 줄 + 2줄) 구조를 지키고, 한 줄당 발성 음절을 6~10음절(랩도 최대 11음절)로 간결하게 유지할 것.
     b) [자연스러운 한영 문장 혼용]: 억지 단어 끼워넣기 대신, 멜로디와 리듬이 더 매끄러워지는 구간에 자연스러운 짧은 영어 문장(4~8단어)이나 구절을 자연스럽게 섞어 작성할 것.
     c) [인위적 표기 금지]: 가사 줄 중간에 마침표(.) 남발 금지, `< >` 및 `**` 금지, 로마자 발음·번역 금지. 메타 태그는 섹션 시작 [ ]에만 간결하게 배치할 것.

3) [포맷 출력]: 시스템 프롬프트의 ###DETAIL### 부터 ###UPLOAD### 까지 필수 구분자에 맞춰 최종 결과물만 출력해.
</Action_Steps>"""

    print(f"\n[1] 생성된 시드 프롬프트:\n{user_prompt}\n")
    print(f"    - 선택 장르: {selected_genre}")

    best = None  # (score, data, fatal, warnings, pacing)

    for attempt in range(MAX_RETRIES):
        print(f"\n[2] Gemini 가사 생성 중... (시도 {attempt + 1}/{MAX_RETRIES})")
        result_data, status = generate_lyrics_with_gemini(system_instruction, user_prompt)

        if status in ("no_key", "no_model", "api_error"):
            print("🛑 재시도해도 해결되지 않는 상태라 생성을 중단합니다.")
            break

        if status == "parse_error" or not result_data:
            print("   ⚠️ 결과를 파싱하지 못했습니다. 재시도합니다.")
        else:
            fatal, warnings, pacing = validate_result(result_data)
            print(
                f"   🎵 [호흡/밀도 분석] 총 {pacing['line_count']}줄 / "
                f"예상 {pacing['total_syllables']}음절 / "
                f"줄당 평균 {pacing['avg_syllables']}음절 / "
                f"최대 {pacing['max_syllables']}음절 / "
                f"과밀 줄 {pacing['overcrowded']}개"
            )
            for w in warnings:
                print(f"   ℹ️ 경고: {w}")

            score = len(fatal) * 10 + len(warnings)
            if result_data.get("lyrics", "").strip() and (best is None or score < best[0]):
                best = (score, result_data, fatal, warnings, pacing)

            if not fatal:
                print("   ✅ 치명적 오류 없음! 호흡·길이 검증 통과.")
                break
            print(f"   ⚠️ 검증 미달 ({' / '.join(fatal)})")

        if attempt < MAX_RETRIES - 1:
            print(f"   ⏳ {RETRY_WAIT_SEC}초 대기 후 재시도합니다...")
            time.sleep(RETRY_WAIT_SEC)

    if best is None:
        print("❌ 유효한 데이터를 생성하지 못해 파이프라인을 종료합니다.")
        return

    _, final_data, final_fatal, final_warnings, final_pacing = best
    notes = [f"[오류] {x}" for x in final_fatal] + [f"[경고] {x}" for x in final_warnings]
    notes.append(
        f"[밀도] {final_pacing['line_count']}줄 / {final_pacing['total_syllables']}음절 (평균 {final_pacing['avg_syllables']})"
    )
    if final_fatal:
        print("⚠️ 모든 시도에서 치명적 오류가 남아, 이슈가 가장 적은 결과를 저장합니다.")

    print("\n[3] Notion 저장 시도...")
    save_to_notion(current_date, selected_genre, user_prompt, final_data, notes)


if __name__ == "__main__":
    main()
