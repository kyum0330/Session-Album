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
# 설정값 (Session-Album : Winter / Christmas — English Pop)
# ============================================================
TIMES_FILE = 'data/times_winter.json'
PLACES_FILE = 'data/places_winter.json'

MAX_OUTPUT_TOKENS = 16384   # 사고(thinking) 토큰 포함 모델 대비 넉넉하게 설정
TEMPERATURE = 1.0           # Gemini 3.x 이상 계열 권장 기본값
MAX_RETRIES = 3
RETRY_WAIT_SEC = 10

# 글자 수 검증 기준
# - Suno Style 한도: 1000자 미만 (권장 350~600자)
# - Suno Lyrics 한도: 5000자 미만 (영어 가사는 한글보다 글자 수가 많아 1200~4900자 허용)
SUNO_WARN_MIN, SUNO_MAX = 300, 999
LYRICS_MIN, LYRICS_MAX = 1200, 4900

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

MARKERS_BASE = [
    "DETAIL", "PURPOSE", "SUNO", "EXCLUDE_STYLES", "VOCAL",
    "IMAGE", "LYRICS", "CLEAN_LYRICS", "TAG", "UPLOAD",
]

REQUIRED_SECTIONS = [
    "Intro", "Verse 1", "Pre-Chorus 1", "Chorus 1", "Verse 2", "Pre-Chorus 2",
    "Chorus 2", "Bridge", "Instrumental Solo", "Chorus 3", "Outro",
]
REQUIRED_FIELDS = ["detail", "purpose", "suno", "vocal", "lyrics", "tag", "upload"]

# 겨울 팝송 시즌 장르 가중치 (기본값 1). 높을수록 자주 뽑혀요.
WINTER_GENRE_WEIGHTS = {
    "R&B": 3,
    "Contemporary R&B": 3,
    "Synth Pop": 3,
    "Electro Pop": 3,
    "New Jack Swing": 3,
    "City Pop": 2,
    "UK Garage": 2,
    "House": 2,
    "Liquid Drum & Bass": 2,
    "Old-school Hip Hop": 2,
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
def load_data(file_path, fallback_path=None):
    target = file_path
    if not os.path.exists(target) and fallback_path and os.path.exists(fallback_path):
        print(f"ℹ️ '{file_path}'가 없어 '{fallback_path}'를 대신 사용합니다.")
        target = fallback_path
    with open(target, 'r', encoding='utf-8') as file:
        return json.load(file)


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
    if genre1 == genre2:
        return True
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
# 프롬프트 조립 (English Pop Winter Session)
# ============================================================
def build_pop_system_instruction(genre1, genre2, rule1, rule2):
    return f"""[멜로디 및 사운드 디자인 (Meta Tags) 강제 규칙 - Winter Pop Session (English)]
너는 빌보드와 글로벌 팝 시장을 이끄는 세계적인 음반사의 천재적인 팝 작사가이자, 곡의 다이내믹을 설계하는 총괄 프로듀서에요.
제시된 [장르], [시간], [시즌 컨텍스트], [장소], [감정], [행동] 데이터를 활용해, 선택된 두 장르({genre1}, {genre2})의 비트감과 감정선이 매력적으로 어우러지는 세련된 **영어 팝송(English Pop)** 오리지널 곡을 만들어야 해요.

이번 시즌은 '겨울, 그리고 크리스마스'예요. 작사 배경에 주어진 날짜와 시즌 컨텍스트를 보고 크리스마스(12월 25일)까지의 시기감을 곡의 온도에 자연스럽게 반영해 주세요.
- 11월 말 ~ 12월 초: 첫눈, 거리에 하나둘 켜지는 트리 불빛, 설레는 기다림
- 12월 중순 ~ 24일: 크리스마스 카운트다운, 선물, 약속, 들뜬 설렘
- 12월 25일 전후: 크리스마스 당일의 포근하고 반짝이는 행복
- 12월 26일 ~ 31일: 연말, 한 해를 돌아보는 아련함과 따뜻한 회상
- 1월 이후: 새해, 깊은 겨울의 고요함과 새 출발
단, 기존 유명 캐럴이나 팝송의 멜로디·가사를 인용하거나 변형하지 말고, 완전히 새로운 오리지널 영어 팝송으로 만드세요.

[작사 핵심 및 메타 태그 규칙]
1. 보컬 및 페르소나 (###VOCAL### 및 ###LYRICS### 기준)
   - 기본 보컬 방향: Smooth alto female vocal, natural native English diction, melodic delivery, effortless singing, gentle resonance, subtle R&B vocal runs, relaxed phrasing, mellow dynamics, Solo.
   - Suno AI가 쨍한 고음 벨팅이나 시 낭송(Spoken)으로 빠지지 않도록, 확실하게 멜로디를 타는(Melodic) 담백하고 따뜻한 중저음 팝 보컬 톤을 유지해요. (Spoken, Talking, Speech, Monologue 절대 금지)

   1-1. 섹션별 보컬 메타 태그 적용 (###LYRICS### 영역)
        각 섹션 시작 부분의 대괄호([ ]) 안에 아래 태그를 상황에 맞게 조합해 창법을 제어하세요.
        - [Soft melodic chorus]: 부드럽게 멜로디를 타는 후렴구
        - [Clear Smooth Falsetto]: 공기 소리를 줄이고 선명도를 높인 맑고 부드러운 가성
        - [Warm Gentle High Notes]: 쨍하지 않고 따뜻하게 감싸듯 올라가는 편안한 고음
        - [Controlled Vocal]: 감정은 담되 에너지가 과하지 않게 절제된 보컬
        - [Mellow dynamics]: 튀는 구간 없이 차분하고 부드러운 다이내믹
        - [Soft vocal delivery]: 처음부터 끝까지 부드럽게 내뱉는 보컬 표현
        - [Laid-back phrasing]: 여유롭고 힘을 뺀 보컬 리듬 스타일

   1-2. 보컬 스타일 키워드 (###VOCAL### 영역 전용)
        아래 긍정형 키워드를 ###VOCAL### 섹션에 조합하세요. (이 보컬 키워드들은 ###SUNO###에는 절대 넣지 마세요.)
        - mellow dynamics, melodic delivery, relaxed phrasing, soft vocal delivery, native English phrasing

   1-3. 악기 및 반주 에너지 조절 (###SUNO### Style of Music 영역 전용)
        반주가 너무 크면 보컬이 자동으로 소리를 지르게 돼요. ###SUNO###에는 아래 키워드를 적절히 섞어 반주의 온도와 밀도를 조절하세요.
        - 기본 밸런스: chill, lo-fi, soft instrumentation, minimalist, mellow dynamics, modern English pop
        - 겨울 질감 포인트 (곡 분위기에 맞춰 2~3개만 은은하게 선택): soft sleigh bells, glockenspiel, celesta, music box, warm Rhodes, gentle chimes, warm synth pad, vinyl crackle, muted brass, cozy analog warmth

   1-4. 호흡 조절: 고음이나 호흡 정리가 필요한 부분에만 단어 사이에 ',', '.'를 적절히 삽입해 자연스럽게 숨을 고르게 하세요.

   1-5. 고음 내지름(Belting) 및 독백 방지 규칙:
        - 도입부/1절 (확실하게 깔아주는 저음): [Smooth relaxed female vocal, gentle resonance]
        - 힘을 완전히 빼는 파트: [Subdued melodic vocal, relaxed phrasing]
        - 코러스/고음 진입 파트 (에너지 억제): [Controlled alto vocal, warm delivery]

   1-6. 랩 중심 장르 예외 규칙:
        선택된 장르에 Hip Hop, Old-school Hip Hop, Baltimore Club이 포함된 경우에 한해, Verse 파트에서만 [Rhythmic melodic rap-singing, smooth low delivery, laid-back flow]를 허용해요. 이때도 Spoken/Talking/Monologue는 금지이며, Chorus는 반드시 멜로디 보컬로 돌아와야 해요.

2. 비트 및 다이내믹 (장르 융합 규칙)
   이번 곡의 메인 장르는 {genre1}, 서브 장르는 {genre2} 예요. 아래 두 장르의 규칙을 모두 내부 지시어로 반영하세요.

[{genre1} 룰 (메인 장르)]
{rule1}

[{genre2} 룰 (서브 장르)]
{rule2}

   - 템포 기준: 기준 BPM은 메인 장르({genre1})의 '권장 템포' 범위 안에서 정하고, 서브 장르({genre2})는 리듬·악기 질감이나 하프타임/더블타임으로 융합해요.
   - 구조 배분: Verse와 Pre-Chorus는 메인 장르({genre1})의 무드를, Chorus는 서브 장르({genre2})의 리듬과 에너지를 중심으로 교차하거나 하이브리드로 섞어주세요. Bridge와 Chorus 3에서는 두 장르를 함께 결합해요.
   - 겨울 질감: 두 장르 모두 거친 모서리를 한 단계 낮춘 듯한 포근하고 세련된 겨울 팝 질감으로 풀어내세요.

3. 트렌디한 글로벌 겨울 팝 작사 및 훅(Hook) 가이드
   2026년 겨울 글로벌 팝 트렌드인 'Cozy Cool (무심하지만 따뜻한 멋)'과 'Warm Chill (세련된 여유)' 무드를 구현하세요.
   - **100% 자연스러운 영어 가사**: ###LYRICS###의 모든 가사는 원어민이 일상에서 쓰고 노래하는 자연스럽고 세련된 영어(Contemporary Conversational English)로 작성하세요. 한국어 가사나 번역 문장을 절대 넣지 마세요.
   - **라임과 리듬 설계**: 단순 끝운(End Rhyme)뿐 아니라 내부 운율(Internal Rhyme)과 모음 조화(Assonance)를 살려 입에 착 감기는 팝 멜로디를 유도하세요.
   - **겨울 무드 핵심 1음절 훅 키워드 참고** (매번 같은 단어만 쓰지 말고 자연스럽게 변주):
     * Cozy & Warm: Hush, Warm, Wrap, Glow, Ember, Close
     * Crisp & Clear: Frost, Breath, Haze, Drift, Snow, Crisp
     * Christmas Sparkle: Spark, Bell, Gift, Melt, Shine, Blink
     * Year-end Nostalgia: Fade, Echo, Still, Dawn, Hold, Stay
   - **코러스 훅 반복**: 기억에 남는 짧은 영어 훅 문구를 정해 Chorus 1·2·3에서 2~3회 자연스럽게 반복하세요.

4. 이스터 에그 (무심한 여유 표현)
   Verse 파트 중 한 곳에 영어 구어체 표현인 **"just to"** (예: "went out just to feel the snow", "stayed up just to watch the lights")를 딱 1회 자연스럽게 삽입해 무심하고 여유로운 태도를 연출하세요.

5. 악기 솔로 구간 (Instrumental Solo)
   Bridge 이후 Chorus 3 직전에 두 장르 규칙의 '솔로 후보' 중 하나를 골라 반드시 대괄호([ ]) 형식으로 영어로 삽입하세요. (`< >` 기호 절대 사용 금지)
   - 형식 예시: [Instrumental Solo: Warm Rhodes Solo], [Instrumental Solo: Soft Synth Lead Solo]

6. 메타 태그 통일 및 가사 규칙
   - 가사 내 모든 구조·악기·보컬 지시어는 오직 대괄호([ ])만 사용하세요. `< >` 기호와 마크다운 굵게(**)는 절대 사용하지 마세요.
   - 각 섹션 시작 부분에 보컬·악기 지시를 집중 배치하고, 가사 줄 중간에 태그를 과도하게 남발하지 마세요.
   - 가사 옆에 한국어 번역이나 발음 주석을 절대 덧붙이지 마세요.

7. 가사 밀도 최적화 및 여백 확보 (심플 룰)
   - 한 줄당 4~9단어 안팎으로 간결하고 리드미컬하게 작성하세요. (랩 파트는 예외)
   - 문장이 길게 늘어지지 않게 끊어 쓰고, 고음이나 감정 고조 구간에는 가사 밀도를 줄여 보컬의 호흡과 여백을 만드세요.

8. 겨울 감성 가이드 (클리셰 방지)
   - 시드의 영어 단어를 기계적으로 나열하지 말고, 김 서린 카페 창가, 따뜻한 머그잔, 가로등 아래 입김, 목도리에 묻은 찬 공기, 늦은 밤 드라이브의 트리 불빛 같은 구체적인 장면으로 그리세요.
   - 종교적 표현이나 뻔한 캐럴 문구 대신 보편적인 사랑·설렘·연말의 따뜻한 회상에 집중하세요.
   - 감정선은 포근함 70%, 설렘 또는 아련함 30%의 균형을 유지하세요.

모든 답변은 반드시 아래의 [구분자]를 정확히 사용하여 섹션을 나누어 출력하세요. (순수 가사 정리본은 프로그램이 자동 추출하므로 CLEAN_LYRICS 섹션은 출력하지 마세요.)

###DETAIL###
노래 제목(Subject: 세련된 영어 키워드 1~3단어), 장르(Genre), Tempo(BPM), Key, 악기 구성, 작사 배경 및 분위기 구성을 **한국어**로 띄어쓰기 포함 800자 이내로 작성하세요.
* 제목 및 정보 항목에 마크다운 굵게(**)는 절대 사용하지 마세요.

###PURPOSE###
'작사가의 한마디'로 이 영어 팝송의 기획 의도와 구간별 감정선 흐름을 **한국어**로 자연스럽게 소개하세요.

###SUNO###
위 DETAIL의 '장르, Tempo, 핵심 악기 구성, 겨울 사운드 질감, 분위기'를 Suno AI의 'Style of Music' 입력란에 바로 붙여넣을 수 있도록 영어 키워드로 작성하세요.
* English Pop 감성을 기본으로, 핵심 장르, 템포(BPM), 시그니처 악기, 겨울 질감, 분위기 순으로 중요한 키워드를 앞쪽에 배치하세요.
* 권장 길이: 띄어쓰기 포함 350자 ~ 600자 (최대 900자 이하, 절대 1000자 초과 금지).
* 보컬 관련 내용과 부정형 지시어(no, without, don't 등)는 절대 작성하지 마세요.
* 예시: Modern English Pop, Warm Contemporary R&B, Synth Pop, 94 BPM, warm Rhodes electric piano, analog synth pads, smooth bassline, gentle glockenspiel accents, subtle vinyl crackle, cozy analog warmth, soft instrumentation, mellow dynamics, winter city night lights

###EXCLUDE_STYLES###
Suno AI가 피해야 할 음악적 요소만 영어 키워드(쉼표 구분)로 200자 이내로 나열하세요.
* 기본 배제 후보: k-pop, korean lyrics, high pitch vocal, shrill soprano, belting, screaming, loud brass, aggressive drums, spoken, monologue, sudden start, loud intro
* 겨울 시즌 방해 요소 (상황에 맞게 포함): heavy sleigh bell loops, traditional carol choir, children's choir, cheesy holiday jingle
* 가사에 사용하는 영단어(훅 단어)나 선택된 두 장르의 핵심 특징과 충돌하는 단어는 넣지 마세요.

###VOCAL###
이 곡에 어울리는 영어 팝 보컬 스타일을 영어로 구체적으로 작성하세요. (150자 ~ 280자)
* 규칙 1-2의 보컬 키워드(mellow dynamics, melodic delivery, relaxed phrasing, soft vocal delivery, native English phrasing 등)를 자연스럽게 반영하세요.
* 형식: [성별], [음역 및 톤], [창법 및 스타일], [솔로/듀엣/그룹 여부]
* 예시: Female vocal, warm low-pitched alto, smooth chest resonance, clear native English diction, melodic delivery, effortless singing, relaxed phrasing, soft vocal delivery, mellow dynamics, cozy winter pop tone, Solo.

###LYRICS###
곡의 구조는 반드시 아래 11개 섹션으로 구성하세요.
[Intro] - [Verse 1] - [Pre-Chorus 1] - [Chorus 1] - [Verse 2] - [Pre-Chorus 2] - [Chorus 2] - [Bridge] - [Instrumental Solo: 악기명] - [Chorus 3] - [Outro]
* 모든 가사는 **100% 자연스러운 영어**로만 작성하세요. 한국어 번역이나 주석은 절대 넣지 마세요.
* 섹션 표시에 마크다운 굵게(**)와 `< >` 기호는 절대 사용하지 마세요.
* 권장 분량: 띄어쓰기와 대괄호 태그, 가사를 모두 포함해 총 1,800자 ~ 3,200자 사이로 작성하세요. (최소 1,400자 이상, 절대 4,500자 초과 금지)
* 가사가 반복되더라도 "(Repeat)" 등으로 축약하지 말고 모든 텍스트를 온전히 다 적으세요.

###TAG###
이 곡과 어울리는 유튜브 노출용 트렌디 해시태그를 영어와 한글을 번갈아 섞어 정확히 30개 작성하세요. 겨울/크리스마스 팝송 키워드(#WinterPop, #겨울팝송, #ChristmasPlaylist, #크리스마스팝송 등)를 상단에 배치하고, 각 태그마다 ','로 구분하며 노출 가능성이 큰 순서대로 나열하세요.

###UPLOAD###
유튜브 업로드용 요약 양식으로 작성하세요. 괄호 안의 안내문은 실제 내용으로 채우고 괄호 자체는 출력하지 마세요.
1) 해시태그 5개: 제목 단어와 한영 중복을 제외하고 유튜브 검색 노출 우선 키워드 5개 (#태그1 #태그2 #태그3 #태그4 #태그5)
2) 빈 줄 후: 날짜와 겨울 감정 기반 짧은 소개글 (한글 2~4문장)
3) 빈 줄 후: 위 한글 소개글의 자연스러운 영어 번역
4) 빈 줄 후:
제목: (곡 제목)
장르: (장르)
Tempo: (BPM)
Key: (키)
악기: (주요 악기 구성)
"""


def build_image_prompt(extracted):
    return f"""아래 첨부된 [곡 상세 정보]와 [기획 의도]를 꼼꼼히 분석하여, 유튜브 영상용 16:9 썸네일 이미지를 하나 생성해 줘.

[필수 적용 조건]
1. 화질 및 텍스트 배제: 노이즈가 없는 4K 이상의 초고화질(ISO 100 수준의 선명함)로 렌더링하고, 화면 내에 어떠한 텍스트, 영단어, 간판 글씨도 절대 들어가지 않게 해 줘.
2. 글로벌 팝 감성 인물 및 스타일링: 곡의 보컬과 서사를 분석하여 가장 잘 어울리는 주인공을 설정해 줘. 세련되고 감각적인 20대 글로벌 인디/팝 아티스트 무드로 묘사하되, 자연스러운 피부톤과 과하지 않은 메이크업을 기본으로 해 줘.
3. 분위기 기반 비주얼 매칭: 차분하고 서정적인 곡이라면 은은하고 깊이 있는 눈빛과 시네마틱한 구도를, 리드미컬한 곡이라면 매력적이고 세련된 시선과 도시 야경의 생동감을 적용해 줘.
4. 감성 시각화 및 겨울 의상: 겨울/크리스마스 시즌(뉴욕, 런던, 또는 따뜻한 북유럽 감성의 겨울 거리·카페)에 어울리는 세련된 '꾸안꾸(Effortless Cool)' 겨울 패션(울 코트, 청키 니트, 머플러 등)을 입혀주고, 따뜻한 실내 조명·트리 불빛 보케와 차가운 겨울 밤공기의 색감 대비를 생생하게 표현해 줘.

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
    lyrics = extracted.get("lyrics", "")
    lyrics = lyrics.replace("**", "")
    lyrics = re.sub(r'<([^>]+)>', r'[\1]', lyrics)
    extracted["lyrics"] = lyrics.strip()

    if extracted["lyrics"]:
        extracted["clean_lyrics"] = build_clean_lyrics(extracted["lyrics"])

    raw_tag = extracted.get("tag", "")
    tags, seen = [], set()
    for t in raw_tag.split(","):
        t = t.strip()
        key = t.lstrip("#").strip().lower()
        if key and key not in seen:
            seen.add(key)
            tags.append(t)
    extracted["tag"] = ", ".join(tags[:30])

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

    # 팝송 가사에 한글이 섞였는지 확인
    if clean_lyrics and re.search(r'[가-힣]', clean_lyrics):
        fatal.append("영어 팝송 가사(LYRICS)에 한글이 포함되어 있음")

    # ── 경고 (재시도 없음) ──
    if 0 < suno_len < SUNO_WARN_MIN:
        warnings.append(f"Style(Suno) 다소 짧음 ({suno_len}자)")

    just_to_count = len(re.findall(r'\bjust to\b', clean_lyrics, re.IGNORECASE))
    if clean_lyrics and just_to_count != 1:
        warnings.append(f"이스터 에그 'just to' 출현 {just_to_count}회 (목표 1회)")

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
# Notion 저장 (Session-Album 스키마 유지 / [POP] 접두어 구분)
# ============================================================
def get_chunks(text):
    if not text:
        return [{"text": {"content": " "}}]
    return [{"text": {"content": text[i:i+2000]}} for i in range(0, len(text), 2000)]


def save_to_notion(date_str, genre, prompt, data_dict, notes=None):
    notion_token = os.environ.get("NOTION_TOKEN")
    database_id = os.environ.get("NOTION_DATABASE_ID_POP") or os.environ.get("NOTION_DATABASE_ID")

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
        "heading_2": {"rich_text": [{"text": {"content": "🎶 Gemini 생성 가사 및 곡 구성 (English Pop)"}}]}
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
            "Title": {"title": [{"text": {"content": f"{date_str} [POP] ({genre})"}}]},
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
            "Genre": {"rich_text": [{"text": {"content": f"POP / {genre}"}}]},
            "Upload": {"rich_text": [{"text": {"content": data_dict.get("upload", "")[:2000]}}]}
        },
        "children": children_blocks
    }

    response = requests.post('https://api.notion.com/v1/pages', headers=headers, json=payload, timeout=30)

    print(f"📊 [결과] HTTP 상태 코드: {response.status_code}")
    if response.status_code == 200:
        print("✅ Notion 저장 성공! [POP] 데이터가 들어갔습니다.")
    else:
        print(f"❌ Notion 저장 실패! 상세 사유: {response.text}")


# ============================================================
# 메인
# ============================================================
def main():
    print("🎵 === [main_2.py] Session-Album : Winter English Pop 시작 ===")
    try:
        genres1 = load_data('data/genres1.json')
        genres2 = load_data('data/genres2.json')
        times = load_data(TIMES_FILE, fallback_path='data/times.json')
        emotions1 = load_data('data/emotions1.json')
        actions = load_data('data/actions.json')
        places = load_data(PLACES_FILE, fallback_path='data/places.json')
        emotions2 = load_data('data/emotions2.json')
        genre_rules = load_data('data/genre_rules.json')
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
        rule_for_genre1 = f"- {selected_genre1}의 특성을 겨울 팝송의 포근한 질감으로 잘 살려서 편곡해."
    if not rule_for_genre2:
        print(f"⚠️ '{selected_genre2}' 장르 룰이 genre_rules.json에 없어요. 기본 문구로 대체합니다.")
        rule_for_genre2 = f"- {selected_genre2}의 특성을 겨울 팝송의 포근한 질감으로 잘 살려서 편곡해."

    selected_genre = f"{selected_genre1}, {selected_genre2}"
    selected_time = get_random_item(times)
    selected_emotion1, selected_emotion2 = get_distinct_emotions(emotions1, emotions2)
    selected_action = get_random_item(actions)
    selected_place = get_random_item(places)

    now = datetime.now()
    current_date = now.strftime("%Y년 %m월 %d일")
    season_context = get_christmas_context(now)

    system_instruction = build_pop_system_instruction(
        selected_genre1, selected_genre2,
        rule_for_genre1, rule_for_genre2
    )

    user_prompt = f"""<Current_Status>
- 진행 단계: 글로벌 겨울/크리스마스 영어 팝송 컨셉 기획 및 최종 음원 데이터 완성
- 타겟 결과물: 유튜브 및 오디오 플랫폼 업로드용 영어 팝송(English Pop) 기획안 및 가사
</Current_Status>

<Brainstorming_Seed>
- 스타일 축: Modern English Pop (Winter / Christmas Session)
- 장르 조합: {selected_genre} (메인: {selected_genre1} / 서브: {selected_genre2})
- 기준 날짜 및 시간: {current_date}, {selected_time}
- 시즌 컨텍스트: {season_context}
- 장소: {selected_place}
- 행동·상태: {selected_action}
- 감정선: Verse({selected_emotion1}) → Chorus({selected_emotion2})
</Brainstorming_Seed>

<Action_Steps>
위의 <Brainstorming_Seed>를 바탕으로 다음 단계를 거쳐 작업을 수행해 줘.

1) [내부 구상]: 위 시드를 엮어 영어 팝송에 어울리는 기승전결 감정 흐름(Emotional Flow)을 설계해. 시즌 컨텍스트의 크리스마스 시기감이 곡의 온도에 자연스럽게 스며들게 하고, 원어민 감성의 중독성 있는 영어 코러스 훅(Hook) 후보 3가지를 구상해 가장 세련된 1가지를 확정해. (내부 추론은 출력하지 말 것)

2) [스타일 및 가사 최적화]:
   - ###SUNO### (Style of Music): Modern English Pop 정체성과 두 장르({selected_genre}), 템포(메인 장르 {selected_genre1} 기준), 겨울 질감 악기, 분위기를 간결한 영어 키워드로 작성할 것. 보컬 관련 키워드와 부정형 지시어는 절대 넣지 마.
   - ###EXCLUDE_STYLES### (제외할 스타일): k-pop, korean lyrics, sudden start, loud intro, heavy sleigh bell loops 등 방해 요소만 쉼표로 구분해 작성할 것.
   - ###LYRICS###:
     a) [언어]: 반드시 100% 자연스러운 영어 가사로만 작성할 것 (한국어/번역 절대 금지).
     b) [메타 태그 통일]: 모든 구조·악기·보컬 지시어는 대괄호([ ])만 사용할 것 (`< >` 및 마크다운 `**` 금지).
     c) [장면화 & 라임]: 시드 단어를 그대로 나열하지 말고 구체적인 겨울 장면으로 묘사하며, 리듬감 있는 내부 라임과 4~9단어의 간결한 호흡을 유지할 것.

3) [포맷 출력]: 시스템 프롬프트에서 요구한 필수 구분자 포맷(###DETAIL### ~ ###UPLOAD###)에 맞추어 최종 결과물만 출력해.
</Action_Steps>"""

    print(f"\n[1] 생성된 시드 프롬프트 (POP):\n{user_prompt}\n")
    print(f"    - 선택 장르: {selected_genre}")
    print(f"    - 시즌 컨텍스트: {season_context}")

    best = None  # (score, data, fatal, warnings)

    for attempt in range(MAX_RETRIES):
        print(f"\n[2] Gemini 영어 팝송 가사 생성 중... (시도 {attempt + 1}/{MAX_RETRIES})")
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

    print("\n[3] Notion 저장 시도 (POP)...")
    save_to_notion(current_date, selected_genre, user_prompt, final_data, notes)


if __name__ == "__main__":
    main()
