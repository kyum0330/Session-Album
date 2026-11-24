import json
import random
import os
import re
import time
from datetime import datetime
import requests
import google.generativeai as genai

# ============================================================
# 설정값 (Session-Album : Winter / Christmas Session)
# ============================================================
SYSTEM_INSTRUCTION_FILE = 'data/system_instruction_winter.txt'

# 글자 수 검증 기준 (프롬프트: SUNO 800~900자 / LYRICS 3500~4500자)
SUNO_MIN, SUNO_MAX = 700, 999        # Suno Style 한도 1000자 미만
LYRICS_MIN, LYRICS_MAX = 2500, 4950  # Suno Lyrics 한도 5000자 미만

GENERATION_RETRIES = 5
GENERATION_WAIT_SEC = 15

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


# ============================================================
# 데이터 로드 / 랜덤 선택
# ============================================================
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def get_random_item(data):
    if isinstance(data, dict):
        all_items = []
        for items_in_category in data.values():
            all_items.extend(items_in_category)
        return random.choice(all_items)
    elif isinstance(data, list):
        return random.choice(data)


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


# ============================================================
# Gemini 호출 및 파싱
# ============================================================
def pick_model_candidates(available_models, preferred_models):
    """정확히 일치하는 모델을 먼저, 없으면 부분 일치(첫 번째)를 사용해요."""
    candidates = []
    for model_name in preferred_models:
        if model_name in available_models:
            candidates.append(model_name)
            continue
        short = model_name.split('/')[-1]
        matched = [m for m in available_models if short in m]
        if matched:
            candidates.append(matched[0])
        else:
            print(f"⚠️ {model_name} 모델은 현재 계정 목록에 없어 건너뜁니다.")
    return candidates


def build_image_prompt(extracted):
    return f"""아래 첨부된 [곡 상세 정보]와 [기획 의도]를 꼼꼼히 분석하여, 유튜브 영상용 16:9 썸네일 이미지를 하나 생성해 줘. 

[필수 적용 조건]
1. 화질 및 텍스트 배제: 노이즈가 없는 4K 이상의 초고화질(ISO 100 수준의 선명함)로 렌더링하고, 화면 내에 어떠한 텍스트, 영단어, 간판 글씨도 절대 들어가지 않게 해 줘.
2. 인물 성별 및 스타일링 자동 최적화: 곡의 보컬 성별과 서사를 분석하여 가장 잘 어울리는 주인공의 성별을 설정해 줘. 인물은 '수수하면서도 트렌디한 한국 20대 연예인(배우/아이돌) 스타일'로 묘사하되, 맑은 피부톤과 자연스러운 메이크업을 기본으로 해 줘.
3. 분위기 기반 비주얼 매칭: 차분하고 서정적인 곡이라면 은은하고 깊이 있는 이목구비를, 트렌디하고 리드미컬한 곡이라면 선이 또렷하고 매혹적인 이목구비를 적용해 줘. 표정 역시 곡의 핵심 감정에 맞춰 과하지 않게 조율해 줘.
4. 감성 시각화 및 의상: 곡의 날씨와 계절감(겨울/크리스마스 시즌)에 완벽히 들어맞는 세련된 '꾸안꾸(Effortless Cool)' 겨울 패션(니트, 코트, 머플러 등)을 입혀주고, 장르에서 느껴지는 온도와 핵심 감정선을 따뜻한 조명(트리 불빛, 가로등 등)과 색감으로 생생하게 시각화해 줘.

[곡 상세 정보]
{extracted.get('detail', '')}

[기획 의도]
{extracted.get('purpose', '')}"""


def generate_lyrics_with_gemini(full_prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("🚨 GEMINI_API_KEY가 설정되지 않았습니다.")
        return {}

    genai.configure(api_key=api_key)
    text = ""

    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"🧐 [참고] 사용 가능한 모델 총 {len(available_models)}개 확인 완료")

        preferred_models = [
            'models/gemini-3.5-flash',
            'models/gemini-2.5-flash',
            'models/gemini-1.5-flash',
            'models/gemini-3.1-flash-lite',
            'models/gemini-2.0-flash-lite'
        ]

        success = False
        for target in pick_model_candidates(available_models, preferred_models):
            try:
                print(f"🚀 [{target}] 모델로 생성을 시도합니다...")
                model = genai.GenerativeModel(target)
                response = model.generate_content(full_prompt)
                text = response.text
                print(f"✅ {target} 생성 성공!")
                success = True
                break
            except Exception as e:
                print(f"⚠️ {target} 실패 (사유: {str(e)[:120]}) -> 다음 모델로 자동 전환합니다.")
                time.sleep(3)

        if not success:
            print("❌ 준비된 모든 대체 모델이 실패했습니다. 할당량 초과라면 자정이 지나길 기다리거나 결제 연동이 필요합니다.")
            return {}

    except Exception as api_e:
        print(f"❌ API 모델 리스트를 불러오지 못했습니다: {api_e}")
        return {}

    try:
        markers_base = ["DETAIL", "PURPOSE", "SUNO", "EXCLUDE_STYLES", "VOCAL", "LYRICS", "CLEAN_LYRICS", "TAG", "UPLOAD"]
        for m in markers_base:
            text = re.sub(r'[*_]*#+\s*' + m + r'\s*#*[*_]*', f'###{m}###', text, flags=re.IGNORECASE)

        markers = [f"###{m}###" for m in markers_base]
        extracted = {m.lower(): "" for m in markers_base}
        extracted["image"] = ""

        for marker in markers:
            if marker in text:
                part = text.split(marker)[1]
                min_idx = len(part)
                for other_marker in markers:
                    if other_marker != marker:
                        idx = part.find(other_marker)
                        if idx != -1 and idx < min_idx:
                            min_idx = idx

                key = marker.replace("#", "").lower()
                extracted[key] = part[:min_idx].strip()

        # 이미지 프롬프트는 파싱이 모두 끝난 뒤 한 번만 생성
        extracted["image"] = build_image_prompt(extracted)

        print("\n[4] 파싱된 섹션별 글자 수 (0이면 AI가 생성을 빼먹은 것입니다):")
        for k, v in extracted.items():
            print(f" - {k}: {len(v)}자")

        return extracted

    except Exception as e:
        print(f"Gemini 데이터 파싱 에러: {e}")
        return {}


# ============================================================
# Notion 저장
# ============================================================
def get_chunks(text):
    return [{"text": {"content": text[i:i+2000]}} for i in range(0, max(1, len(text)), 2000)]


def save_to_notion(date_str, genre, prompt, data_dict):
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

    children_blocks = [{"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "🎶 Gemini 생성 가사 및 곡 구성"}}]}}]

    for para in data_dict["lyrics"].split('\n\n'):
        para = para.strip()
        if not para:
            continue

        if len(para) > 2000:
            while len(para) > 2000:
                split_idx = para.rfind('\n', 0, 2000)
                if split_idx == -1:
                    split_idx = para.rfind(' ', 0, 2000)
                if split_idx == -1:
                    split_idx = 2000

                chunk = para[:split_idx].strip()
                children_blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": chunk}}]}})
                para = para[split_idx:].strip()

        if para:
            children_blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": para}}]}})

    children_blocks.append({"object": "block", "type": "divider", "divider": {}})
    children_blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": data_dict.get("tag", "")[:2000]}}]}})

    clean_lyrics_content = data_dict.get("clean_lyrics", "")
    clean_lyrics_chunks = [{"text": {"content": clean_lyrics_content[i:i+2000]}} for i in range(0, max(1, len(clean_lyrics_content)), 2000)] if clean_lyrics_content else [{"text": {"content": " "}}]

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
            "Lyrics": {"rich_text": clean_lyrics_chunks},
            "E_Lyrics": {"rich_text": get_chunks(data_dict.get("lyrics", " "))},
            "Tag": {"rich_text": [{"text": {"content": data_dict.get("tag", "")[:2000]}}]},
            "Genre": {"rich_text": [{"text": {"content": genre}}]},
            "Upload": {"rich_text": [{"text": {"content": data_dict.get("upload", "")[:2000]}}]}
        },
        "children": children_blocks
    }

    response = requests.post('https://api.notion.com/v1/pages', headers=headers, json=payload)

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
        times = load_data('data/times.json')
        emotions1 = load_data('data/emotions1.json')
        actions = load_data('data/actions.json')
        places = load_data('data/places.json')
        emotions2 = load_data('data/emotions2.json')
        genre_rules = load_data('data/genre_rules.json')

        # 외부 텍스트 파일에서 시스템 프롬프트 불러오기
        raw_system_instruction = load_text(SYSTEM_INSTRUCTION_FILE)

    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return

    max_pick_retries = 100
    pick_retry_count = 0
    selected_genre1 = selected_genre2 = None

    while pick_retry_count < max_pick_retries:
        selected_genre1 = get_weighted_genre(genres1)
        selected_genre2 = get_weighted_genre(genres2)

        if not is_unwanted_combination(selected_genre1, selected_genre2):
            break

        pick_retry_count += 1
        print(f"⚠️ 원하지 않는 조합 발생 ({selected_genre1}, {selected_genre2}) -> 다시 뽑습니다.")

    if pick_retry_count == max_pick_retries:
        print("❌ 유효한 장르 조합을 찾는 데 실패했습니다.")
        return

    # 장르 룰 확인 (없으면 경고 후 기본 문구 사용)
    rule_for_genre1 = genre_rules.get(selected_genre1)
    rule_for_genre2 = genre_rules.get(selected_genre2)
    if not rule_for_genre1:
        print(f"⚠️ '{selected_genre1}' 장르 룰이 genre_rules.json에 없어요. 기본 문구로 대체합니다.")
        rule_for_genre1 = f"{selected_genre1}의 특성을 겨울 시즌의 포근한 질감으로 잘 살려서 편곡해줘."
    if not rule_for_genre2:
        print(f"⚠️ '{selected_genre2}' 장르 룰이 genre_rules.json에 없어요. 기본 문구로 대체합니다.")
        rule_for_genre2 = f"{selected_genre2}의 특성을 겨울 시즌의 포근한 질감으로 잘 살려서 편곡해줘."

    selected_genre = f"{selected_genre1}, {selected_genre2}"
    selected_time = get_random_item(times)
    selected_emotion1 = get_random_item(emotions1)
    selected_action = get_random_item(actions)
    selected_place = get_random_item(places)
    selected_emotion2 = get_random_item(emotions2)

    now = datetime.now()
    current_date = now.strftime("%Y년 %m월 %d일")
    season_context = get_christmas_context(now)

    system_instruction = build_system_instruction(
        raw_system_instruction,
        selected_genre1, selected_genre2,
        rule_for_genre1, rule_for_genre2
    )

    user_prompt = f"""
<Current_Status>
- 진행 단계: 초기 컨셉 브레인스토밍 및 구간별 감정 흐름을 반영한 최종 음원 데이터 완성
- 타겟 결과물: 유튜브 및 오디오 플랫폼 업로드용 겨울/크리스마스 시즌 기획안 및 가사
</Current_Status>

<Brainstorming_Seed>
- 장르: {selected_genre}
- 배경/시간: {current_date}, {selected_time}
- 시즌 컨텍스트: {season_context}
- 장소 및 상황: {selected_place}에서 {selected_action} 하는 중
- 감정선: {selected_emotion1} 분위기 속에서 느껴지는 {selected_emotion2}
</Brainstorming_Seed>

<Action_Steps>
위의 <Brainstorming_Seed>를 바탕으로 다음 단계를 거쳐 작업을 수행해 줘.

1) [내부 구상]: 이 키워드들을 엮어서 곡의 기승전결에 따른 구간별 감정 흐름(Emotional Flow)을 치밀하게 설계해. 시즌 컨텍스트에 담긴 크리스마스까지의 시기감(설렘/당일의 행복/연말의 아련함 등)이 곡 전체의 온도에 자연스럽게 녹아들게 해. 동시에 2026년 겨울 Z세대와 숏폼 플랫폼에서 자주 쓰이는 트렌디한 밈이나 힙한 영단어 슬랭을 3가지 도출하여 훅(Hook) 설계에 반영해 봐. (이 과정은 너의 내부 추론을 위한 것이며 출력하지 않아도 됨)

2) [스타일 및 가사 최적화]:
   - ###SUNO### (Style of Music): 곡의 장르와 함께 곡을 관통하는 핵심 분위기와 감정을 묘사하는 간결한 영어 키워드를 포함할 것. 부정형 지시어는 절대 사용하지 마.
   - ###EXCLUDE_STYLES### (제외할 스타일): sudden start, loud intro 등 곡에 원치 않는 음악적 방해 요소들을 쉼표로 구분해 작성할 것. 가사에 쓰는 영단어(훅 단어)는 이 칸에 넣지 마.
   - ###LYRICS###: 네가 설계한 구간별 감정선이 Suno AI에 직접 전달되도록, 아래 3가지 규칙을 엄격히 지켜서 작성해.
     a) [메타 태그 지휘]: 각 단락(Verse, Chorus 등) 시작 부분의 대괄호 [ ] 안에 곡의 구조뿐만 아니라 보컬의 감정, 창법, 분위기를 구체적으로 명시할 것. (예: [Smooth Relaxed Female Vocal], 고음부 절제를 위한 [Clear Smooth Falsetto] 또는 [Controlled Vocal])
     b) [트렌드 융합]: 내부 구상에서 도출한 최신 밈 단어를 훅(Hook)에 반드시 1개 이상 자연스럽게 융합할 것.
     c) [타격감 극대화]: 훅 메이킹 시, 한국어의 부드러운 자음(ㅇ, ㄹ) 뒤에 영어의 강한 파열음(P, B, D, K)이나 마찰음(V, Z, Sh)으로 끝나는 1음절 영단어를 배치하여 쫀득한 타격감과 리듬감을 새롭게 창조할 것.

3) [포맷 출력]: 확정한 아이디어를 바탕으로, 시스템 프롬프트에서 요구한 ###DETAIL### 부터 ###UPLOAD### 까지의 9가지 필수 구분자 포맷에 맞추어 완벽한 최종 결과물만 출력해.
</Action_Steps>
"""

    full_prompt = f"{system_instruction}\n\n[작사 배경]\n{user_prompt}"

    print(f"\n[1] 생성된 프롬프트: {full_prompt[:500]} ... (중략) ... \n[토큰 최적화 완료!]")
    print(f"    - 선택 장르: {selected_genre}")
    print(f"    - 시즌 컨텍스트: {season_context}")

    result_data = {}

    for attempt in range(GENERATION_RETRIES):
        print(f"\n[2] Gemini 가사 생성 중... (시도 {attempt + 1}/{GENERATION_RETRIES})")
        result_data = generate_lyrics_with_gemini(full_prompt)

        suno_len = len(result_data.get("suno", ""))
        lyrics_len = len(result_data.get("lyrics", ""))

        print(f"   -> 생성된 Style(Suno) 글자 수: {suno_len}자")
        print(f"   -> 생성된 가사(Lyrics) 글자 수: {lyrics_len}자")

        if SUNO_MIN <= suno_len <= SUNO_MAX and LYRICS_MIN <= lyrics_len <= LYRICS_MAX:
            print("   ✅ 글자 수 한계치 통과! 완벽합니다.")
            break
        else:
            print("   ⚠️ 글자 수 제한 초과 또는 미달! 다시 생성합니다.")
            if suno_len > SUNO_MAX:
                print("      - 사유: Style(Suno) 1000자 초과")
            if suno_len < SUNO_MIN:
                print("      - 사유: Style(Suno)가 너무 짧음 (생성 누락 의심)")
            if lyrics_len > LYRICS_MAX:
                print("      - 사유: 가사(Lyrics) 5000자 초과 위험")
            if lyrics_len < LYRICS_MIN:
                print("      - 사유: 가사(Lyrics)가 너무 짧음 (생략 발생 의심)")

            if attempt < GENERATION_RETRIES - 1:
                print(f"   ⏳ API 할당량 보호를 위해 {GENERATION_WAIT_SEC}초 대기 후 재시도합니다...")
                time.sleep(GENERATION_WAIT_SEC)

    if not result_data.get("lyrics", "").strip():
        print("❌ 유효한 길이의 데이터를 생성하는 데 실패했습니다. 파이프라인을 종료합니다.")
        return

    print("\n[3] Notion 저장 시도...")
    save_to_notion(current_date, selected_genre, user_prompt, result_data)


if __name__ == "__main__":
    main()
