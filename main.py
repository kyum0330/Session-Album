import json
import random
import os
import re
import time
from datetime import datetime
import requests
from google import genai
from google.genai import types


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
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


def get_two_distinct_emotions(emotions_dict):
    """서로 다른 감정 카테고리에서 Verse용, Chorus용 감정을 하나씩 추출"""
    categories = list(emotions_dict.keys())
    if len(categories) >= 2:
        cat1, cat2 = random.sample(categories, 2)
        emo1 = clean_seed(random.choice(emotions_dict[cat1]))
        emo2 = clean_seed(random.choice(emotions_dict[cat2]))
        return (cat1, emo1), (cat2, emo2)
    emo1 = get_random_item(emotions_dict)
    emo2 = get_random_item(emotions_dict)
    return ("Emotion", emo1), ("Emotion", emo2)


def get_seoul_weather():
    """서울 날씨 조회. 실패 시 None 반환 (프롬프트에서 자동 생략)"""
    api_key = os.environ.get("WEATHER_API_KEY")
    if api_key:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q=Seoul&appid={api_key}&lang=kr"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['weather'][0]['description']
        except Exception as e:
            print(f"⚠️ 날씨 정보를 가져오는 중 에러 발생 (날씨 시드 생략): {e}")
    return None


def is_unwanted_combination(genre1, genre2):
    if genre1 == genre2:
        return True
    unwanted_pairs = {
        ("Folk", "Dubstep"), ("Country", "Hyperpop"), ("Bossa Nova", "Industrial"),
        ("Operatic Pop", "Jersey Club"), ("Blues", "EDM"), ("Ballad", "Afrobeat"),
        ("Jazz", "Hyperpop"), ("Reggae", "Industrial"), ("Lo-fi", "Trap"),
        ("Operatic Pop", "Trap"), ("Folk", "Hyperpop"), ("Country", "Industrial"),
        ("Bossa Nova", "Dubstep"), ("Ballad", "Jersey Club"), ("Blues", "Hyperpop"),
        ("Lo-fi", "Dubstep"), ("Soul", "Industrial"), ("Jazz", "Dubstep"),
        ("Reggae", "Hyperpop"), ("Operatic Pop", "UK Garage"), ("Country", "Deep House"),
        ("Folk", "Jersey Club"), ("Bossa Nova", "Trap"), ("Ballad", "Industrial"), ("Indie", "EDM")
    }
    return (genre1, genre2) in unwanted_pairs or (genre2, genre1) in unwanted_pairs


def build_clean_lyrics(raw_lyrics):
    """LYRICS에서 [...]와 <...> 태그를 제거해 순수 가사 생성"""
    cleaned = re.sub(r'\[[^\]]*\]', '', raw_lyrics)
    cleaned = re.sub(r'<[^>]*>', '', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def post_process_data(extracted):
    """사소한 포맷 오류를 파이썬에서 자동 보정"""
    # 1. LYRICS: 마크다운 굵게(**) 제거, <...>를 [...]로 통일
    lyrics = extracted.get("lyrics", "")
    lyrics = lyrics.replace("**", "")
    lyrics = re.sub(r'<([^>]+)>', r'[\1]', lyrics)
    extracted["lyrics"] = lyrics.strip()

    # 2. CLEAN_LYRICS: 파이썬 정규식으로 자동 생성
    extracted["clean_lyrics"] = build_clean_lyrics(extracted["lyrics"])

    # 3. TAG: # 제거 및 정확히 최대 30개로 정리
    raw_tag = extracted.get("tag", "").replace("#", "")
    tags = [t.strip() for t in raw_tag.split(",") if t.strip()]
    extracted["tag"] = ", ".join(tags[:30])

    # 4. IMAGE: Gemini가 생성한 시각 프롬프트가 비었을 때만 안전망 텍스트 사용
    if not extracted.get("image", "").strip():
        extracted["image"] = (
            "16:9 ratio trendy YouTube music video thumbnail, no text, no typography, "
            "cinematic mood matching the song atmosphere.\n\n"
            f"[곡 상세 정보]\n{extracted.get('detail', '')}"
        )

    return extracted


def validate_result(extracted):
    """재생성이 필요한 핵심 오류만 검사"""
    issues = []
    suno_len = len(extracted.get("suno", ""))
    lyrics = extracted.get("lyrics", "")
    lyrics_len = len(lyrics)

    if suno_len == 0 or suno_len >= 1000:
        issues.append(f"Style(Suno) 길이 이상 ({suno_len}자)")
    if lyrics_len < 1500:
        issues.append(f"가사(Lyrics) 너무 짧음 ({lyrics_len}자)")
    if lyrics_len > 4800:
        issues.append(f"가사(Lyrics) 너무 김 ({lyrics_len}자)")

    required_sections = ["[Intro]", "[Verse 1]", "[Chorus 1]", "[Outro]"]
    missing = [sec for sec in required_sections if sec.lower() not in lyrics.lower()]
    if missing:
        issues.append(f"핵심 섹션 누락: {', '.join(missing)}")

    return issues


def generate_lyrics_with_gemini(system_instruction, user_prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("🚨 GEMINI_API_KEY가 설정되지 않았습니다.")
        return {}, False

    client = genai.Client(api_key=api_key)
    text = ""

    try:
        available_models = [
            m.name for m in client.models.list()
            if m.supported_actions and "generateContent" in m.supported_actions
        ]
        print(f"🧐 [참고] 사용 가능한 모델 총 {len(available_models)}개 확인 완료")

        preferred_models = [
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
        ]

        success = False
        for model_name in preferred_models:
            matched = [
                m for m in available_models
                if m.endswith(f"/{model_name}") or m == model_name
            ]
            if not matched:
                continue

            target = matched[0]
            try:
                print(f"🚀 [{target}] 모델로 생성을 시도합니다...")
                response = client.models.generate_content(
                    model=target,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=8192,
                        temperature=0.85,
                    ),
                )
                text = response.text or ""
                if text.strip():
                    print(f"✅ {target} 생성 성공!")
                    success = True
                    break
            except Exception as e:
                print(f"⚠️ {target} 실패 ({str(e)[:120]}) -> 다음 모델로 전환합니다.")
                time.sleep(2)

        if not success:
            print("❌ 준비된 모든 모델이 할당량 초과 또는 오류로 실패했습니다.")
            return {}, True  # quota_exhausted = True

    except Exception as api_e:
        print(f"❌ API 모델 리스트 조회 실패: {api_e}")
        return {}, True

    try:
        markers_base = ["DETAIL", "PURPOSE", "SUNO", "EXCLUDE_STYLES", "VOCAL", "IMAGE", "LYRICS", "TAG", "UPLOAD"]
        for m in markers_base:
            text = re.sub(r'[*_]*#+\s*' + m + r'\s*#*[*_]*', f'###{m}###', text, flags=re.IGNORECASE)

        markers = [f"###{m}###" for m in markers_base]
        extracted = {m.lower(): "" for m in markers_base}

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

        extracted = post_process_data(extracted)

        print("\n[4] 파싱된 섹션별 글자 수:")
        for k, v in extracted.items():
            print(f" - {k}: {len(v)}자")

        return extracted, False

    except Exception as e:
        print(f"❌ Gemini 데이터 파싱 에러: {e}")
        return {}, False


def get_chunks(text):
    if not text:
        return [{"text": {"content": " "}}]
    return [{"text": {"content": text[i:i+2000]}} for i in range(0, len(text), 2000)]


def save_to_notion(date_str, genre, weather, prompt, data_dict):
    notion_token = os.environ.get("NOTION_TOKEN")
    database_id = os.environ.get("NOTION_DATABASE_ID")

    if not notion_token or not database_id or not data_dict.get("lyrics", "").strip():
        print("❌ 저장할 가사(LYRICS) 데이터가 비어있어 Notion 호출을 취소합니다.")
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
        "paragraph": {"rich_text": [{"text": {"content": data_dict.get("tag", "")[:2000]}}]}
    })

    weather_display = weather if weather else "날씨 정보 없음"

    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            "Title": {"title": [{"text": {"content": f"{date_str} ({genre})"}}]},
            "Weather": {"rich_text": [{"text": {"content": weather_display}}]},
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


def main():
    try:
        genres1 = load_data('data/genres1.json')
        genres2 = load_data('data/genres2.json')
        times = load_data('data/times.json')
        emotions1 = load_data('data/emotions1.json')
        actions = load_data('data/actions.json')
        places = load_data('data/places.json')
        genre_rules = load_data('data/genre_rules.json')
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return

    max_genre_retries = 100
    selected_genre1, selected_genre2 = None, None
    for _ in range(max_genre_retries):
        g1 = get_random_item(genres1)
        g2 = get_random_item(genres2)
        if not is_unwanted_combination(g1, g2):
            selected_genre1, selected_genre2 = g1, g2
            break

    if not selected_genre1:
        print("❌ 유효한 장르 조합을 찾는 데 실패했습니다.")
        return

    rule_for_genre1 = genre_rules.get(selected_genre1, f"{selected_genre1}의 특성을 살려 편곡해.")
    rule_for_genre2 = genre_rules.get(selected_genre2, f"{selected_genre2}의 특성을 살려 편곡해.")

    selected_genre = f"{selected_genre1}, {selected_genre2}"
    selected_time = get_random_item(times)
    (emo1_cat, selected_emotion1), (emo2_cat, selected_emotion2) = get_two_distinct_emotions(emotions1)
    selected_action = get_random_item(actions)
    selected_place = get_random_item(places)
    selected_vocal_type = random.choice([
        "Female Solo", "Male Solo", "Female Solo", "Male Solo", "Mixed Duet"
    ])

    current_date = datetime.now().strftime("%Y년 %m월 %d일")
    current_weather = get_seoul_weather()

    system_instruction = f"""너는 감성을 자극하는 음반 회사의 작사가이자 곡의 다이내믹을 설계하는 총괄 프로듀서다.
제시된 시드 데이터를 활용해, 선택된 두 장르({selected_genre})의 비트감과 감정선이 매력적으로 어우러지는 트렌디한 곡을 기획하고 작사하라.
가사는 한 문장이 너무 길지 않으면서도 중독성 있고 세련되게 작성한다.

[우선순위 및 충돌 해결 규칙]
1. 보컬 기본값 vs 장르 룰:
   - 기본 보컬 방향은 과도한 찢어짐(Shrill Belting) 없이 담백하고 매력적인 중저음·절제된 표현(mellow dynamics, controlled vocal, intimate vocal, laid-back)이다.
   - 단, 아래 [장르별 규칙]이 폭발적인 에너지나 특정 창법(예: EDM, Dubstep, Soul, Operatic Pop 등)을 요구해 기본값과 충돌할 경우, **장르별 규칙을 우선** 적용한다.
2. BPM 및 템포 충돌:
   - 두 장르의 템포가 다르면 **{selected_genre1}을 기준 BPM**으로 삼고, {selected_genre2}는 하프타임/더블타임 또는 리듬·악기 질감으로 융합한다.
3. 시드 충돌:
   - 시간·장소·날씨 시드가 서로 충돌하면 **장소·시간 시드를 우선**하고, 실제 날씨는 분위기 참고용으로만 사용한다.
4. 메타 태그 기호 통일:
   - 가사 내 모든 구조·악기·보컬 지시어는 반드시 대괄호(`[ ]`)만 사용한다. (`< >` 및 마크다운 `**` 사용 금지)
   - 각 섹션 시작 부분에 보컬/악기 지시를 집중 배치하고, 가사 중간에는 과도한 태그 남발을 피한다.
   - 고음이나 호흡 조절이 필요한 구간에만 `,` `.`를 적절히 배치한다.
5. 한국어 가사 규칙:
   - 한국어 발음을 영문 로마자로 표기하지 않는다. 순수 한국어 문장과 자연스러운 영단어 훅(Hook)만 사용한다.
6. 이스터 에그:
   - Verse 파트 중 한 곳에 '~할 겸'(예: 바람 쐴 겸, 생각 지울 겸) 표현을 자연스럽게 딱 1회 삽입한다.

[장르별 규칙]
- 메인 장르 ({selected_genre1}):
{rule_for_genre1}

- 서브 장르 ({selected_genre2}):
{rule_for_genre2}

[출력 포맷 — 아래 구분자를 정확히 지켜 출력]

###DETAIL###
제목(Subject: 소재 나열이 아닌 핵심 키워드 1~2단어), 장르(Genre), Tempo(BPM), Key, 악기 구성, 작사 배경 및 분위기 구성을 띄어쓰기 포함 800자 이내로 작성한다.

###PURPOSE###
'작사가의 한마디'로 이 곡의 기획 의도와 감정선 흐름을 자연스럽게 소개한다.

###SUNO###
Suno AI의 'Style of Music' 입력란용 영어 키워드 문자열.
- 핵심 장르, 템포, 시그니처 악기, 사운드 질감 순서로 **중요한 키워드를 앞쪽에 배치**한다.
- 300~500자 권장 (절대 900자 초과 금지).
- **보컬 관련 키워드는 절대 적지 않는다** (보컬은 VOCAL 섹션에 분리).

###EXCLUDE_STYLES###
Suno AI가 피해야 할 요소를 영어 키워드(쉼표 구분)로 작성한다.
- 기본 배제 후보: `shrill soprano, harsh screaming, muddy mix, off-key, excessive reverb`
- 단, 선택된 장르({selected_genre})의 핵심 특징과 충돌하는 키워드는 배제 목록에 넣지 않는다.

###VOCAL###
이 곡에 어울리는 보컬 스타일을 영어로 작성한다. (150~280자)
형식: [성별/구성], [음역 및 톤], [창법 및 감정 전달 스타일] (시드의 보컬 구성을 반영할 것)

###IMAGE###
이 곡의 16:9 유튜브 썸네일 생성용 상세 프롬프트 (한국어 또는 영어, 300~500자).
- 구체적인 장면, 색감 팔레트, 조명, 카메라 구도를 묘사한다.
- 인물이 등장한다면 수수하고 트렌디한 한국 20대 스타일로 묘사한다.
- 이미지 안에 노래 제목이나 어떤 텍스트도 넣지 말 것(no text, no typography)을 명시한다.

###LYRICS###
곡 구조는 반드시 아래 11개 섹션으로 구성한다.
`[Intro] - [Verse 1] - [Pre-Chorus 1] - [Chorus 1] - [Verse 2] - [Pre-Chorus 2] - [Chorus 2] - [Bridge] - [Instrumental Solo] - [Chorus 3] - [Outro]`
- `[Instrumental Solo]`에는 두 장르 중 하나를 대표하는 핵심 악기 솔로 지시를 대괄호 안에 함께 적는다 (예: `[Instrumental Solo: Warm Jazz Saxophone]`).
- EDM/Dubstep처럼 드롭 구간이 핵심인 장르라도 `[Chorus]`에 짧은 훅이나 챈트 가사를 유지해 곡 구조를 완성한다.
- 전체 분량은 대괄호 태그와 가사를 포함해 **2,200자 ~ 3,200자** 사이로 작성하며, 반복 구간도 축약하지 않고 모두 적는다.

###TAG###
유튜브 노출용 트렌디 해시태그를 한글과 영어 섞어서 정확히 30개 작성한다. `#` 없이 쉼표(`, `)로만 구분한다.

###UPLOAD###
유튜브 업로드용 요약 양식으로 작성한다. (예시 내용을 복사하지 말고 이번 곡의 실제 내용으로 채울 것)
- 해시태그 5개: 제목 단어 제외, 한영 중복 제외, 유튜브 검색 노출 우선 키워드 5개 (`#태그1 #태그2 #태그3 #태그4 #태그5`)
- 빈 줄 후: 작사가의 한마디 (3~5문장)
- 빈 줄 후:
제목: [곡 제목]
장르: [장르]
Tempo: [BPM]
Key: [키]
악기: [주요 악기 구성]
"""

    weather_line = f"- 참고 날씨(서울): {current_weather}\n" if current_weather else ""

    user_prompt = f"""<Brainstorming_Seed>
- 장르 조합: {selected_genre} (메인: {selected_genre1} / 서브: {selected_genre2})
- 기준 날짜 및 시간: {current_date}, {selected_time}
{weather_line}- 장소: {selected_place}
- 행동·상태: {selected_action}
- 감정 아크: Verse({emo1_cat} - {selected_emotion1}) → Chorus({emo2_cat} - {selected_emotion2})
- 보컬 구성: {selected_vocal_type}
</Brainstorming_Seed>

<Action_Steps>
1) 내부적으로 위 시드를 엮은 3가지 스토리라인과 시각 테마를 구상한 뒤 가장 트렌디한 1가지를 확정하라 (내부 추론은 출력하지 말 것).
2) 확정한 아이디어로 시스템 프롬프트의 ###DETAIL### 부터 ###UPLOAD### 까지의 포맷에 맞춰 최종 결과물만 출력하라.
</Action_Steps>"""

    print(f"\n[1] 생성된 시드 프롬프트:\n{user_prompt}\n")

    max_retries = 3
    result_data = {}

    for attempt in range(max_retries):
        print(f"\n[2] Gemini 가사 생성 중... (시도 {attempt + 1}/{max_retries})")
        result_data, quota_exhausted = generate_lyrics_with_gemini(system_instruction, user_prompt)

        if quota_exhausted:
            print("🛑 API 할당량 소진으로 재시도를 중단합니다.")
            break

        issues = validate_result(result_data)
        if not issues:
            print("   ✅ 품질 및 글자 수 검증 통과!")
            break
        else:
            print(f"   ⚠️ 검증 미달 ({' / '.join(issues)})")
            if attempt < max_retries - 1:
                print("   ⏳ 10초 대기 후 재시도합니다...")
                time.sleep(10)

    if not result_data.get("lyrics", "").strip():
        print("❌ 유효한 데이터를 생성하지 못해 파이프라인을 종료합니다.")
        return

    print("\n[3] Notion 저장 시도...")
    save_to_notion(current_date, selected_genre, current_weather, user_prompt, result_data)


if __name__ == "__main__":
    main()
