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
# 설정값 (Session-Album : Winter / Christmas Session)
# ============================================================
SYSTEM_INSTRUCTION_FILE = 'data/system_instruction_winter.txt'
TIMES_FILE = 'data/times_winter.json'
PLACES_FILE = 'data/places_winter.json'

MAX_OUTPUT_TOKENS = 16384   # 사고(thinking) 토큰 포함 모델 대비 넉넉하게 설정
TEMPERATURE = 1.0           # Gemini 3.x 이상 계열 권장 기본값
MAX_RETRIES = 3
RETRY_WAIT_SEC = 10

# 글자 수 검증 기준
# - Suno Style 한도: 1000자 미만 (권장 300~850자, 300자 미만은 경고만 출력)
# - Suno Lyrics 한도: 5000자 미만 (1000자 미만 또는 4900자 초과는 치명적 오류)
SUNO_WARN_MIN, SUNO_MAX = 300, 999
LYRICS_MIN, LYRICS_MAX = 1000, 4900

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

# IMAGE / CLEAN_LYRICS 모두 호환되도록 마커 등록 (CLEAN_LYRICS는 파이썬 후처리로 최종 보정)
MARKERS_BASE = [
    "DETAIL", "PURPOSE", "SUNO", "EXCLUDE_STYLES", "VOCAL",
    "IMAGE", "LYRICS", "CLEAN_LYRICS", "TAG", "UPLOAD",
]

REQUIRED_SECTIONS = [
    "Intro", "Verse 1", "Pre-Chorus 1", "Chorus 1", "Verse 2", "Pre-Chorus 2",
    "Chorus 2", "Bridge", "Instrumental Solo", "Chorus 3", "Outro",
]
REQUIRED_FIELDS = ["detail", "purpose", "suno", "vocal", "lyrics", "tag", "upload"]

# 겨울 시즌 장르 가중치 (기본값 1). 높을수록 자주 뽑혀요.
WINTER_GENRE_WEIGHTS = {
    "R&B": 3,
    "Contemporary R&B": 3,
    "New Jack Swing": 3,
    "City Pop": 3,
    "Synth Pop": 3,
    "Electro Pop": 3,
    "Liquid Drum & Bass": 2,
    "UK Garage": 2,
    "House": 2,
    "Old-school Hip Hop": 2,
    # 아래는 여름 파티 색이 강한 장르 -> 기본 가중치 1 유지
    "Moombahton": 1,
    "Favela Funk": 1,
    "Miami Bass": 1,
    "Baltimore Club": 1,
    "Jersey Club": 1,
    "Hip Hop": 1,
}

# 둘 다 뽑히면 겨울 무드에 너무 뜨거워지는 파티 계열 장르
PARTY_HEAVY_GENRES = {"Moombahton", "Favela Funk", "Miami Bass", "Baltimore Club", "Jersey Club"}

# 겨울 시즌에 어색한 개별 조합 (순서 무관)
UNWANTED_PAIRS = {
    frozenset(("Favela Funk", "Liquid Drum & Bass")),
    frozenset(("Baltimore Club", "City Pop")),
    frozenset(("Jersey Club", "City Pop")),
    frozenset(("Miami Bass", "Contemporary R&B")),
    frozenset(("Moombahton", "New Jack Swing")),
    frozenset(("Favela Funk", "Old-school Hip Hop")),
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


def get_weighted_genre(genres):
    weights = [WINTER_GENRE_WEIGHTS.get(g, 1) for g in genres]
    return random.choices(genres, weights=weights, k=1)[0]


def is_unwanted_combination(genre1, genre2):
    # 같은 장르가 두 번 뽑히면 두 장르 하이브리드 의미가 없어서 제외
    if genre1 == genre2:
        return True
    # 파티 계열끼리 겹치면 겨울 무드에 너무 뜨거워짐
    if genre1 in PARTY_HEAVY_GENRES and genre2 in PARTY_HEAVY_GENRES:
        return True
    return frozenset((genre1, genre2)) in UNWANTED_PAIRS


# ============================================================
# 크리스마스 시즌 컨텍스트
# ============================================================
def get_christmas_context(now):
    """오늘 날짜를 기준으로 크리스마스까지의 시기감을 문장으로 만들어요."""
    christmas = datetime(now.year, 12, 25)
    d_day = (christmas.date() - now.date()).days

    if now.month in (3, 4, 5, 6, 7, 8, 9, 10):
        return "겨울 시즌 외 기간: 겨울을 그리워하는 듯한 포근하고 시원한 상상 속 겨울 장면으로 풀어주세요."
    if now.month == 12 and now.day > 25:
        return f"크리스마스 직후 연말 (12월 {now.day}일): 한 해를 돌아보는 아련하고 따뜻한 회상, 조용한 연말의 밤."
    if now.month in (1, 2):
        return "새해와 깊은 겨울: 고요하고 맑은 겨울 공기, 새 출발의 담담한 설렘."
    if d_day == 0:
        return "크리스마스 당일 (D-Day): 포근하고 반짝이는 행복, 사랑하는 사람과 보내는 따뜻한 하루."
    if 1 <= d_day <= 14:
        return f"크리스마스 D-{d_day}: 카운트다운이 시작된 설렘, 선물과 약속, 들뜨지만 여유로운 기다림."
    return f"크리스마스 D-{d_day}: 첫눈과 하나둘 켜지는 트리 불빛을 기다리는 초겨울, 서서히 온도가 내려가는 설렘."


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
        print("⚠️ 장르 룰이 시스템 프롬프트에 들어가지 않았어요. system_instruction 파일의 {rule_for_genre1}/{rule_for_genre2} 자리를 확인하세요.")
    else:
        print("✅ 두 장르의 룰이 시스템 프롬프트에 정상 주입되었습니다.")
    return text


def build_image_prompt(extracted):
    return f"""아래 첨부된 [곡 상세 정보]와 [기획 의도]를 꼼꼼히 분석하여, 유튜브 영상용 16:9 썸네일 이미지를 하나 생성해 줘.

[필수 적용 조건]
1. 화질 및 텍스트 배제: 노이즈가 없는 4K 이상의 초고화질(ISO 100 수준의 선명함)로 렌더링하고, 화면 내에 어떠한 텍스트, 영단어, 간판 글씨도 절대 들어가지 않게 해 줘.
2. 인물 성별 및 스타일링 자동 최적화: 곡의 보컬 성별과 서사를 분석하여 가장 잘 어울리는 주인공의 성별을 설정해 줘. 인물은 '수수하면서도 트렌디한 한국 20대 연예인(배우/아이돌) 스타일'로 묘사하되, 맑은 피부톤과 자연스러운 메이크업을 기본으로 해 줘.
3. 분위기 기반 비주얼 매칭: 차분하고 서정적인 곡이라면 은은하고 깊이 있는 이목구비를, 트렌디하고 리드미컬한 곡이라면 선이 또렷하고 매혹적인 이목구비를 적용해 줘. 표정 역시 곡의 핵심 감정에 맞춰 과하지 않게 조율해 줘.
4. 감성 시각화 및 의상: 겨울/크리스마스 시즌의 계절감에 완벽히 들어맞는 세련된 '꾸안꾸(Effortless Cool)' 겨울 패션(니트, 코트, 머플러 등)을 입혀주고, 장르에서 느껴지는 온도와 핵심 감정선을 따뜻한 조명(트리 불빛, 가로등 보케, 차가운 겨울 밤공기 대비 등)과 색감으로 생생하게 시각화해 줘.

[곡 상세 정보]
{extracted.get('detail', '')}

[기획 의도]
{extracted.get('purpose', '')}"""


# ============================================================
# 후처리 및 검증
# ============================================================
def build_clean_lyrics(raw_lyrics):
    """LYRICS에서 [...]와 <...> 태그를 제거해 순수 가사 생성"""
    cleaned = re.sub(r'\[[^\]]*\]', '', raw_lyrics)
    cleaned = re.sub(r'<[^>]*>', '', cleaned)
    lines = [re.sub(r'[ \t]{2,}', ' ', ln).strip() for ln in cleaned.split('\n')]
    cleaned = '\n'.join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def post_process_data(extracted):
    """사소한 포맷 오류를 파이썬에서 자동 보정"""
    # 1. LYRICS: 마크다운 굵게(**) 제거, <...>를 [...]로 통일
    lyrics = extracted.get("lyrics", "")
    lyrics = lyrics.replace("**", "")
    lyrics = re.sub(r'<([^>]+)>', r'[\1]', lyrics)
    extracted["lyrics"] = lyrics.strip()

    # 2. CLEAN_LYRICS: LYRICS에서 태그를 제거해 100% 일관된 순수 가사 생성
    if extracted["lyrics"]:
        extracted["clean_lyrics"] = build_clean_lyrics(extracted["lyrics"])

    # 3. TAG: 중복 제거 및 최대 30개 정리 (기존 # 형식 유지)
    raw_tag = extracted.get("tag", "")
    tags, seen = [], set()
    for t in raw_tag.split(","):
        t = t.strip()
        key = t.lstrip("#").strip().lower()
        if key and key not in seen:
            seen.add(key)
            tags.append(t)
    extracted["tag"] = ", ".join(tags[:30])

    # 4. IMAGE: 프롬프트에서 ###IMAGE###를 생성하지 않았으면 기본 겨울 썸네일 프롬프트 조립
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
    반환: (fatal_issues, warnings)
    - fatal_issues: 재생성이 필요한 치명적 오류
    - warnings: 로그 및 Notion 본문 메모만 남기고 재시도하지 않는 경미한 이슈
    """
    fatal, warnings = [], []

    lyrics = extracted.get("lyrics", "")
    clean_lyrics = extracted.get("clean_lyrics", "")
    suno_len = len(extracted.get("suno", ""))
    lyrics_len = len(lyrics)

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

    # ── 경고 (재시도 없음) ──
    if 0 < suno_len < SUNO_WARN_MIN:
        warnings.append(f"Style(Suno) 다소 짧음 ({suno_len}자)")

    kyeom_count = len(re.findall(r'(?:^|\s)겸(?:\s|$|[,.!?~])', clean_lyrics))
    if clean_lyrics and kyeom_count != 1:
        warnings.append(f"이스터 에그 '~겸' 출현 {kyeom_count}회 (목표 1회)")

    tag_count = len([t for t in extracted.get("tag", "").split(",") if t.strip()])
    if extracted.get("tag", "").strip() and tag_count != 30:
        warnings.append(f"해시태그 {tag_count}개 (목표 30개)")

    return fatal, warnings


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
# Notion 저장 (Session-Album 스키마 — Weather 속성 없음)
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
        g1 = get_weighted_genre(genres1)
        g2 = get_weighted_genre(genres2)
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
        rule_for_genre1 = f"- {selected_genre1}의 특성을 겨울 시즌의 포근한 질감으로 잘 살려서 편곡해."
    if not rule_for_genre2:
        print(f"⚠️ '{selected_genre2}' 장르 룰이 genre_rules.json에 없어요. 기본 문구로 대체합니다.")
        rule_for_genre2 = f"- {selected_genre2}의 특성을 겨울 시즌의 포근한 질감으로 잘 살려서 편곡해."

    selected_genre = f"{selected_genre1}, {selected_genre2}"
    selected_time = get_random_item(times)
    selected_emotion1, selected_emotion2 = get_distinct_emotions(emotions1, emotions2)
    selected_action = get_random_item(actions)
    selected_place = get_random_item(places)

    now = datetime.now()
    current_date = now.strftime("%Y년 %m월 %d일")
    season_context = get_christmas_context(now)

    system_instruction = build_system_instruction(
        raw_system_instruction,
        selected_genre1, selected_genre2,
        rule_for_genre1, rule_for_genre2
    )

    user_prompt = f"""<Current_Status>
- 진행 단계: 초기 컨셉 브레인스토밍 및 구간별 감정 흐름을 반영한 최종 음원 데이터 완성
- 타겟 결과물: 유튜브 및 오디오 플랫폼 업로드용 겨울/크리스마스 시즌 기획안 및 가사
</Current_Status>

<Brainstorming_Seed>
- 장르 조합: {selected_genre} (메인: {selected_genre1} / 서브: {selected_genre2})
- 기준 날짜 및 시간: {current_date}, {selected_time}
- 시즌 컨텍스트: {season_context}
- 장소: {selected_place}
- 행동·상태: {selected_action}
- 감정선: Verse({selected_emotion1}) → Chorus({selected_emotion2})
</Brainstorming_Seed>

<Action_Steps>
위의 <Brainstorming_Seed>를 바탕으로 다음 단계를 거쳐 작업을 수행해 줘.

1) [내부 구상]: 위 시드를 엮어 곡의 기승전결에 따른 구간별 감정 흐름(Emotional Flow)을 치밀하게 설계해. 시즌 컨텍스트에 담긴 크리스마스까지의 시기감(초겨울 설렘 / 카운트다운 / 당일의 행복 / 연말의 아련함 등)이 곡 전체의 온도에 자연스럽게 녹아들게 해. 동시에 겨울 시즌의 'Cozy Cool · Warm Chill' 무드에 어울리는 감각적인 1음절 영단어 훅 후보를 3가지 도출해 가장 잘 어울리는 표현을 확정해. (내부 추론은 출력하지 말 것)

2) [스타일 및 가사 최적화]:
   - ###SUNO### (Style of Music): 곡의 장르, 템포(메인 장르 {selected_genre1} 기준), 핵심 겨울 질감 악기, 분위기를 간결한 영어 키워드로 작성할 것. 중요한 키워드를 앞쪽에 배치하고, 보컬 관련 내용과 부정형 지시어(no, without 등)는 절대 사용하지 마.
   - ###EXCLUDE_STYLES### (제외할 스타일): sudden start, loud intro, heavy sleigh bell loops 등 곡에 원치 않는 음악적 방해 요소만 쉼표로 구분해 작성할 것. 가사에 쓰는 영단어(훅 단어)는 이 칸에 넣지 마.
   - ###LYRICS###: 설계한 구간별 감정선이 Suno AI에 정확히 전달되도록 아래 규칙을 지켜 작성해.
     a) [메타 태그 통일]: 모든 구조·악기·보컬 지시어는 대괄호([ ])만 사용할 것 (`< >` 및 마크다운 `**` 금지). 각 섹션 시작 부분에 보컬 창법과 분위기를 명시해.
     b) [장면화]: 시드의 영어 단어를 그대로 가사에 복붙하지 말고, 입김·목도리·가로수 전구 같은 구체적인 겨울 장면과 행동으로 풀어서 표현해.
     c) [훅 타격감]: 훅 메이킹 시 한국어의 부드러운 자음(ㅇ, ㄹ) 뒤에 영어의 파열음(P, B, D, K)이나 마찰음(F, V, Sh, Th)이 살아있는 1음절 겨울 영단어를 배치해 쫀득한 리듬감을 만들어.

3) [포맷 출력]: 확정한 아이디어를 바탕으로, 시스템 프롬프트에서 요구한 필수 구분자 포맷에 맞추어 최종 결과물만 출력해.
</Action_Steps>"""

    print(f"\n[1] 생성된 시드 프롬프트:\n{user_prompt}\n")
    print(f"    - 선택 장르: {selected_genre}")
    print(f"    - 시즌 컨텍스트: {season_context}")

    best = None  # (score, data, fatal, warnings)

    for attempt in range(MAX_RETRIES):
        print(f"\n[2] Gemini 가사 생성 중... (시도 {attempt + 1}/{MAX_RETRIES})")
        result_data, status = generate_lyrics_with_gemini(system_instruction, user_prompt)

        if status in ("no_key", "no_model", "api_error"):
            print("🛑 재시도해도 해결되지 않는 상태라 생성을 중단합니다.")
            break

        if status == "parse_error" or not result_data:
            print("   ⚠️ 결과를 파싱하지 못했습니다. 재시도합니다.")
        else:
            fatal, warnings = validate_result(result_data)
            for w in warnings:
                print(f"   ℹ️ 경고: {w}")

            score = len(fatal) * 10 + len(warnings)
            if result_data.get("lyrics", "").strip() and (best is None or score < best[0]):
                best = (score, result_data, fatal, warnings)

            if not fatal:
                print("   ✅ 치명적 오류 없음! 검증 통과.")
                break
            print(f"   ⚠️ 검증 미달 ({' / '.join(fatal)})")

        if attempt < MAX_RETRIES - 1:
            print(f"   ⏳ {RETRY_WAIT_SEC}초 대기 후 재시도합니다...")
            time.sleep(RETRY_WAIT_SEC)

    if best is None:
        print("❌ 유효한 데이터를 생성하지 못해 파이프라인을 종료합니다.")
        return

    _, final_data, final_fatal, final_warnings = best
    notes = [f"[오류] {x}" for x in final_fatal] + [f"[경고] {x}" for x in final_warnings]
    if final_fatal:
        print("⚠️ 모든 시도에서 치명적 오류가 남아, 이슈가 가장 적은 결과를 저장합니다.")

    print("\n[3] Notion 저장 시도...")
    save_to_notion(current_date, selected_genre, user_prompt, final_data, notes)


if __name__ == "__main__":
    main()
