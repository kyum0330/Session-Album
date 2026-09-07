import json
import random
import os
import re
import time
from datetime import datetime
import requests
import google.generativeai as genai

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

def is_unwanted_combination(genre1, genre2):
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
        for model_name in preferred_models:
            matched = [m for m in available_models if model_name.split('/')[-1] in m]
            
            if matched:
                target = matched[0]
                try:
                    print(f"🚀 [{target}] 모델로 생성을 시도합니다...")
                    model = genai.GenerativeModel(target)
                    response = model.generate_content(full_prompt)
                    text = response.text
                    print(f"✅ {target} 생성 성공!")
                    success = True
                    break 
                except Exception as e:
                    print(f"⚠️ {target} 실패 (사유: 할당량 초과 등) -> 다음 모델로 자동 전환합니다.")
                    time.sleep(3) 
            else:
                print(f"⚠️ {model_name} 모델은 현재 계정 목록에 없어 건너뜁니다.")
                
        if not success:
            print("❌ 준비된 모든 대체 모델이 할당량 초과로 실패했습니다. 자정이 지나길 기다리거나 결제 연동이 필요합니다.")
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
        
            extracted["image"] = f"""아래 첨부된 [곡 상세 정보]와 [기획 의도]를 꼼꼼히 분석하여, 유튜브 영상용 16:9 썸네일 이미지를 하나 생성해 줘. 

[필수 적용 조건]
1. 화질 및 텍스트 배제: 노이즈가 없는 4K 이상의 초고화질(ISO 100 수준의 선명함)로 렌더링하고, 화면 내에 어떠한 텍스트, 영단어, 간판 글씨도 절대 들어가지 않게 해 줘.
2. 인물 성별 및 스타일링 자동 최적화: 곡의 보컬 성별과 서사를 분석하여 가장 잘 어울리는 주인공의 성별을 설정해 줘. 인물은 '수수하면서도 트렌디한 한국 20대 연예인(배우/아이돌) 스타일'로 묘사하되, 맑은 피부톤과 자연스러운 메이크업을 기본으로 해 줘.
3. 분위기 기반 비주얼 매칭: 차분하고 서정적인 곡이라면 은은하고 깊이 있는 이목구비를, 트렌디하고 리드미컬한 곡이라면 선이 또렷하고 매혹적인 이목구비를 적용해 줘. 표정 역시 곡의 핵심 감정에 맞춰 과하지 않게 조율해 줘.
4. 감성 시각화 및 의상: 곡의 날씨와 계절감에 완벽히 들어맞는 세련된 '꾸안꾸(Effortless Cool)' 패션을 입혀주고, 장르에서 느껴지는 온도와 핵심 감정선을 조명과 색감으로 생생하게 시각화해 줘.

[곡 상세 정보]
{extracted.get('detail', '')}

[기획 의도]
{extracted.get('purpose', '')}"""

        print("\n[4] 파싱된 섹션별 글자 수 (0이면 AI가 생성을 빼먹은 것입니다):")
        for k, v in extracted.items():
            print(f" - {k}: {len(v)}자")

        return extracted
        
    except Exception as e:
        print(f"Gemini 데이터 파싱 에러: {e}")
        return {}

def get_chunks(text):
    return [{"text": {"content": text[i:i+2000]}} for i in range(0, max(1, len(text)), 2000)]

def save_to_notion(date_str, genre, prompt, data_dict):
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

    children_blocks = [{"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "🎶 Gemini 생성 가사 및 곡 구성"}}]}}]
    
    for para in data_dict["lyrics"].split('\n\n'):
        para = para.strip()
        if not para: continue
        
        if len(para) > 2000:
            while len(para) > 2000:
                split_idx = para.rfind('\n', 0, 2000)
                if split_idx == -1: split_idx = para.rfind(' ', 0, 2000)
                if split_idx == -1: split_idx = 2000 
                
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
            "Generated Prompt": {"rich_text": [{"text": {"content": prompt}}]},
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
        raw_system_instruction = load_text('data/system_instruction.txt')
        
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return 
        
    max_retries = 100 
    retry_count = 0
    
    while retry_count < max_retries:
        selected_genre1 = get_random_item(genres1)
        selected_genre2 = get_random_item(genres2)
        
        if not is_unwanted_combination(selected_genre1, selected_genre2):
            break
            
        retry_count += 1
        print(f"⚠️ 원하지 않는 조합 발생 ({selected_genre1}, {selected_genre2}) -> 다시 뽑습니다.")

    if retry_count == max_retries:
        print("❌ 유효한 장르 조합을 찾는 데 실패했습니다.")
        return

    rule_for_genre1 = genre_rules.get(selected_genre1, f"{selected_genre1}의 특성을 잘 살려서 편곡해줘.")
    rule_for_genre2 = genre_rules.get(selected_genre2, f"{selected_genre2}의 특성을 잘 살려서 편곡해줘.")

    selected_genre = f"{selected_genre1}, {selected_genre2}"
    selected_time = get_random_item(times)
    selected_emotion1 = get_random_item(emotions1)
    selected_action = get_random_item(actions)
    selected_place = get_random_item(places)
    selected_emotion2 = get_random_item(emotions2)

    current_date = datetime.now().strftime("%Y년 %m월 %d일")

    # 불러온 텍스트에 동적 변수들을 포맷팅
    try:
        system_instruction = raw_system_instruction.format(
            selected_genre=selected_genre,
            selected_genre1=selected_genre1,
            selected_genre2=selected_genre2,
            rule_for_genre1=rule_for_genre1,
            rule_for_genre2=rule_for_genre2
        )
    except KeyError as e:
        print(f"⚠️ 프롬프트 포맷팅 중 누락된 변수가 있습니다: {e}")
        system_instruction = raw_system_instruction

    user_prompt = f"""
<Current_Status>
- 진행 단계: 초기 컨셉 브레인스토밍 및 최종 음원 데이터 완성
- 타겟 결과물: 유튜브 및 오디오 플랫폼 업로드용 기획안 및 가사
</Current_Status>

<Brainstorming_Seed>
- 장르: {selected_genre}
- 배경/시간: {current_date}, {selected_time}
- 장소 및 상황: {selected_place}에서 {selected_action} 하는 중
- 감정선: {selected_emotion1} 분위기 속에서 느껴지는 {selected_emotion2}
</Brainstorming_Seed>

<Action_Steps>
위의 <Brainstorming_Seed>를 바탕으로 다음 단계를 거쳐 작업을 수행해 줘.

1) [내부 구상]: 이 키워드들을 엮어서 만들 수 있는 매력적인 스토리라인과 시각적 테마를 스스로 3가지 정도 깊이 있게 브레인스토밍 해봐. (이 과정은 너의 내부 추론을 위한 것이며 출력하지 않아도 됨)
2) [최종 도출]: 네가 구상한 아이디어 중 가장 훌륭하고 트렌디한 1가지를 확정해.
3) [포맷 출력]: 확정한 아이디어를 바탕으로, 시스템 프롬프트에서 요구한 ###DETAIL### 부터 ###UPLOAD### 까지의 필수 구분자 포맷에 맞추어 완벽한 최종 결과물만 출력해.
</Action_Steps>
"""
    
    full_prompt = f"{system_instruction}\n\n[작사 배경]\n{user_prompt}"
    
    print(f"\n[1] 생성된 프롬프트: {full_prompt[:500]} ... (중략) ... \n[토큰 최적화 완료!]")
    
    max_retries = 5 
    result_data = {}
    
    for attempt in range(max_retries):
        print(f"\n[2] Gemini 가사 생성 중... (시도 {attempt + 1}/{max_retries})")
        result_data = generate_lyrics_with_gemini(full_prompt)
        
        suno_content = result_data.get("suno", "")
        lyrics_content = result_data.get("lyrics", "")
        
        suno_len = len(suno_content)
        lyrics_len = len(lyrics_content)
        
        print(f"   -> 생성된 Style(Suno) 글자 수: {suno_len}자")
        print(f"   -> 생성된 가사(Lyrics) 글자 수: {lyrics_len}자")
        
        if suno_len < 1000 and 2500 <= lyrics_len <= 4950:
            print("   ✅ 글자 수 한계치 통과! 완벽합니다.")
            break 
        else:
            print("   ⚠️ 글자 수 제한 초과 또는 미달! 다시 생성합니다.")
            if suno_len >= 1000:
                print("      - 사유: Style(Suno) 1000자 초과")
            if lyrics_len > 4950:
                print("      - 사유: 가사(Lyrics) 5000자 초과 위험")
            if lyrics_len < 2500:
                print("      - 사유: 가사(Lyrics)가 너무 짧음 (생략 발생 의심)")
                
            if attempt < max_retries - 1:
                print("   ⏳ API 할당량 보호를 위해 15초 대기 후 재시도합니다...")
                time.sleep(15)
                
    if not result_data.get("lyrics", "").strip():
        print("❌ 유효한 길이의 데이터를 생성하는 데 실패했습니다. 파이프라인을 종료합니다.")
        return

    print("\n[3] Notion 저장 시도...")
    save_to_notion(current_date, selected_genre, user_prompt, result_data)

if __name__ == "__main__":
    main()
