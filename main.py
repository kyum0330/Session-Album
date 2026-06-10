import json
import random
import os
import re
from datetime import datetime
import requests
import google.generativeai as genai

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_random_item(data):
    if isinstance(data, dict):
        all_items = []
        for items_in_category in data.values():
            all_items.extend(items_in_category)
        return random.choice(all_items)
    elif isinstance(data, list):
        return random.choice(data)

def is_unwanted_combination(genre1, genre2):
    """ 원하지 않는 장르 조합인지 확인하는 함수입니다. 순서에 상관없이 매칭되도록 검사합니다. """
    unwanted_pairs = {
        ("Liquid Drum & Bass", "City Pop"), ("Jersey Club", "Old-school Hip Hop"),
        ("Moombahton", "New Jack Swing"), ("Miami Bass", "Contemporary R&B"), ("Favela Funk", "Synth Pop"), 
        ("House", "Moombahton"), ("Liquid Drum & Bass", "Contemporary R&B"), ("Favela Funk", "City Pop"), ("UK Garage", "Old-school Hip Hop")
    }
    return (genre1, genre2) in unwanted_pairs or (genre2, genre1) in unwanted_pairs
    
def generate_lyrics_with_gemini(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {}
    
    genai.configure(api_key=api_key)
    
    system_instruction = """[멜로디 및 사운드 디자인 (Meta Tags) 강제 규칙]
너는 감성을 자극하는 세계적인 엔터테인먼트 음반 회사의 천재적인 작사가 뿐 아니라 곡의 다이내믹을 설계하는 총괄 프로듀서에요.
요즘 트렌드를 조사한 후에, 제시된 [장르], [시간], [장소], [감정], [행동], [날씨] 데이터를 활용해, 세련되고 미니멀한 무드를 담은 청량한 댄스곡을 만들어야 해.

[작사 핵심 및 메타 태그 규칙]
1. 보컬 및 페르소나: [Smooth alto female vocal, deep calm voice, low octave, subdued pitch, clean natural voice, clear diction, effortless singing, gentle resonance, subtle vocal runs, relaxed delivery, mellow dynamics, soft instrumentation, chill R&B, Solo]. 
Suno AI가 흔한 중-고음 소프라노를 출력하지 않도록, 과도한 기교 없이 담백하고 매력적인 중저음 보컬 톤을 강제해요. 보컬과 코러스 부분에 대해서는 다음 내용을 참고해주세요.

    1-1. 메타 태그 적용 (Lyrics 영역)
        곡이 고조되는 코러스(후렴구)나 브릿지 부분에 단순히 [Chorus]라고만 적으면 AI가 마음대로 소리를 내지를 확률이 높습니다. 이럴 때는 대괄호 안에 보컬의 창법을 직접 제한해 주세요. 상황에 맞게 아래 태그 중 하나를 선택하여 적용하십시오.
- [Soft Chorus]: 부드럽게 부르는 후렴구
- ​[Clear Smooth Falsetto]: 공기 소리를 줄이고 목소리의 선명도를 높인 맑고 부드러운 가성
- ​[Warm Gentle High Notes]: 쨍하지 않고 따뜻하게 감싸듯 올라가는 편안한 고음
- ​[Controlled Vocal]: 감정은 담되 에너지가 과하지 않게 절제된 보컬

    1-2. 음악 스타일 제한 (Style of Music 영역 - 보컬)
        곡 전체의 스타일을 지정하는 칸에도 보컬의 에너지를 낮춰주는 긍정형 키워드를 추가하여 AI가 과호흡을 하지 않도록 진정시켜야 합니다. 아래 키워드를 조합하여 사용하십시오.
- mellow dynamics: 튀는 구간 없이 차분하고 부드러운 다이내믹
- soft vocal delivery: 처음부터 끝까지 부드럽게 내뱉는 보컬 표현
- laid-back: 여유롭고 힘을 뺀 스타일
- intimate vocal: 귀에 대고 속삭이듯 가까운 느낌의 보컬

    1-3. 악기 및 장르의 에너지 조절 (Style of Music 영역 - 반주)
        보컬이 쨍해지는 또 다른 결정적인 이유는 반주(악기) 소리가 너무 크거나 강하기 때문입니다. 
        배경 음악이 웅장하고 시끄러워지면 보컬이 악기 소리에 묻히지 않기 위해 자동으로 소리를 지르게끔 설계되어 있습니다.
        이를 방지하기 위해 아래 키워드를 추가하여 반주의 에너지를 살짝 낮춰주십시오.
- chill, lo-fi, soft instrumentation, minimalist 

    1-4. 고음이 들어가는 부분에서는 단어 사이 사이에 ',', '.'를 삽입하여 의도적으로 숨을 고르게 만들게 합니다.
              
2. 비트 및 다이내믹 (뎀보우 리듬 설계): 
- Verse 파트에서는 스네어(Snare) 사용을 최소화하고, 베이스와 코드에만 뎀보우(Dembow) 노트를 일부 사용하여 미니멀한 여백의 미를 줘요. <Minimal snare, partial dembow bass>
- Chorus 파트에서는 정확하고 꽉 찬 타격감의 뎀보우 리듬을 터뜨려 완벽한 뭄바톤(Moombahton) 비트를 완성해요. <Full dembow rhythm, upbeat moombahton>
- 악기를 무겁게 쌓지 않고, 보컬의 발음이 타악기처럼 쫀득하게 리듬을 타도록 가사의 글자 수를 세밀하게 맞춰요.
3. 한영 혼용 훅(Hook): 귀에 확 꽂히는 명확한 멜로디를 위해, Chorus 파트에는 'Vibe', 'Hype', 'Chill'이라는 단어 느낌의 쿨한 무드의 영단어 조사하여 다양하게 한국어와 찰지게 섞어 중독성 있는 펀치라인을 만들어요. 꼭 'Vibe', 'Hype', 'Chill' 단어가 들어갈 필요는 없어요.
4. 이스터 에그 (행동 교차 룰): Verse 파트 중 한 곳에 반드시 '~할 겸' (예: 바람 쐴 겸, 생각 지울 겸 등)이라는 표현을 딱 한 번 자연스럽게 삽입해서 주인공의 무심하고 여유로운 태도를 연출해요.

곡 중간(Bridge 이후 등)에 해당 장르를 가장 잘 나타내는 **<Instrumental Solo> (악기 솔로 구간)**를 최소 1회 이상 강제로 삽입해요.

5. 고음부에서 AI가 쨍하게 소리를 내지르는 현상(Belting)을 방지하고 싶다면, 아래의 규칙을 엄격히 적용하여 프롬프트를 자동 생성하십시오.

- 도입부/1절 (확실하게 깔아주는 저음): [Low Calm Female Vocal] 또는 [Deep Spoken Vocal]
- 말하듯 힘을 완전히 빼는 파트: [Subdued Vocal]
- 코러스/고음 진입 파트 (에너지 억제): [Controlled Alto Vocal] (음역대를 높이지 않고 중저음역대 안에서 에너지만 살짝 조절하도록 지시합니다.)

모든 답변은 반드시 아래의 [구분자]를 사용하여 섹션을 나누어 작성해야 해요

###DETAIL###
이 칸에는 노래 제목(Subject), 장르(Genre), Tempo, Key, 악기 구성을 포함한 정보와 작사 배경 및 분위기 구성을 적어주세요. (띄어쓰기 포함 총 1000자 이내) 이때 노래 제목은 소재의 나열보다는 키워드 위주로 한개 또는 두개의 단어로 표현해주세요.
* 제목 및 정보 항목에 마크다운 굵게(**)는 절대 사용하지 마요.

###PURPOSE###
이 칸에는 '작사가의 한마디'를 통해 이 곡의 기획 의도와 종합적인 곡 소개를 적어주세요.

###SUNO###
위 DETAIL 부분에 작성한 '장르, Tempo, 악기 구성, 분위기'를 음악 생성 AI(Suno)의 'Style of Music' 란에 바로 복사해 넣을 수 있도록, 영어 키워드 위주로 700자 이내로 번역 및 요약해주세요.
이때, 보컬에 관련된 내용은 작성하지 마세요. (예: Melodic Electronic, Progressive House, 123 BPM, warm synth pad, emotional lead)

###VOCAL###
이 칸에는 해당 노래에 어울리는 보컬 스타일을 영어로 작성해주세요. 이때 톤과 스타일에 대해서는 자세하게 적어주세요.
형식: [성별], [톤], [스타일], [솔로/듀엣/그룹 여부]
* 예시: Female vocal, extremely low-pitched, dark contralto, very heavy chest voice, deep androgynous tone, resonant bassy female voice, husky and thick vocal, Solo.
* 전체 내용은 250~280자로 구체적으로 작성할 것.

###LYRICS###
섹션별 가사: Intro, Chorus, Verse, Bridge, Outro 등으로 구분하여 가사를 작성해. 가사 외의 정보(구간 시간, 악기/분위기)는 반드시 영어로 < > 속에 넣어 표현해주세요.
가사 내 지시어 (Meta Tags) 예시:
[Extremely low vocal], [Heavy and dark contralto singing], [Deep thick chest voice]

###CLEAN_LYRICS###
클린 가사: 위 세부 항목이나 음악 구조(< > 부분)가 모두 제외된, 순수 가사 내용만 복사하기 쉽게 적어주세요.

###TAG###
이 곡과 어울리는 유튜브 노출용 트렌디 해쉬태그를 이용해서 한글과 영어 섞어서 정확히 30개 작성해줘요. 이때 번갈아가며 나오도록 하고, 해당 태크마다','를 붙여주고, 노출 가능성이 큰 순서대로 나열해주세요. (예: #하우스, #새벽감성, ...)

###UPLOAD###
유튜브 업로드용 요약 양식으로 작성해주세요. 
형식: [해쉬태그 5개] + [날짜와 감정 기반 짧은 소개글(한글)] + [날짜와 감정 기반 짧은 한글 소개글 영어로 번역] [곡 정보 요약(제목, 장르, Tempo, Key, 악기)] 순서로 가독성 있게 작성해줘요.
UPLOAD용 형식 예시는 다음과 같아요. 이때, 해쉬태그에 노래 제목은 제외하고 유튜브에서 노출이 많은 순서대로 넣어주세요.

#감성 #playlist #인디  #멜로딕일렉트로닉 #프로그레시브하우스

2026년 5월 15일, 거칠게 정지된 삶의 캔버스 앞에서 불완전함을 성찰하고, 그 속에서 끝없이 맑고 명료한 희망을 발견하는 감정을 바탕으로 만들어졌습니다.

Based on the feeling of 'Rough and Stopped Canvas on an Endless Clear Day' on May 15, 2026.

* 노래 제목(Subject) : 정지된 투명함 (Stopped Transparency)

* 장르(Genre) : Melodic Electronic / Progressive House

* Tempo : 123 BPM

* Key : E Major, 내면의 고요한 성찰에서 시작해 벅찬 해방감으로 뻗어나가는 맑고 투명한 희망을 담기 위함.

* 악기 구성(Instrument composition) : 웜하고 몽환적인 신스 패드, 리드미컬한 베이스라인, 섬세한 하이햇과 킥 드럼, 아르페지오 신스, 이모셔널한 신스 리드, 미니멀한 보컬 이펙트."""

    full_prompt = f"{system_instruction}\n\n[작사 배경]\n{prompt}"
    text = ""
    
   # 🌟 [자동 전환 로직 시작] 
    text = ""
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"🧐 [참고] 사용 가능한 모델 총 {len(available_models)}개 확인 완료")
        
        # 🌟 우겸님을 위한 최강의 모델 우선순위 리스트 (1순위: 3.5 Flash)
        preferred_models = [
            'models/gemini-3.5-flash',
            'models/gemini-2.5-flash',
            'models/gemini-flash-latest',
            'models/gemini-2.5-flash-lite'
        ]
        
        success = False
        for model_name in preferred_models:
            # 리스트에 해당 모델이 있는지 유연하게 확인
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
                    break  # 성공하면 반복문을 즉시 탈출합니다!
                except Exception as e:
                    print(f"⚠️ {target} 실패 (사유: 할당량 초과 등) -> 다음 모델로 넘어갑니다.")
            else:
                print(f"⚠️ {model_name} 모델은 현재 목록에 없어 건너뜁니다.")
                
        if not success:
            print("❌ 준비된 모든 대체 모델이 할당량 초과로 실패했습니다. 자정이 지나길 기다리거나 결제 연동이 필요합니다.")
            return {}
            
    except Exception as api_e:
        print(f"❌ API 모델 리스트를 불러오지 못했습니다: {api_e}")
        return {}
    # 🌟 [자동 전환 로직 끝]

    try:
        # 🌟 강력한 정규표현식: 제미나이가 어떤 특수문자나 띄어쓰기를 섞어놔도 깔끔하게 통일시킵니다.
        markers_base = ["DETAIL", "PURPOSE", "SUNO", "VOCAL", "LYRICS", "CLEAN_LYRICS", "TAG", "UPLOAD"]
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
        
        extracted["image"] = (
            f"이 노래에 맞는 16:9 의 영상 제작에 맞는 썸네일 하나 트랜디한 느낌을 살려서 사람들의 시선을 끌 수 있게 제작 부탁할게요. 이때, 노래에 대한 제목과 설명은 글로 표현하지 말아주세요.\n\n"
            f"[곡 상세 정보]\n{extracted.get('detail', '')}\n\n"
            f"[기획 의도]\n{extracted.get('purpose', '')}"
        )

        # 🌟 디버깅 로그 출력: 제미나이가 만든 항목별 글자 수를 GitHub 액션 화면에 보여줍니다.
        print("\n[4] 파싱된 섹션별 글자 수 (0이면 AI가 생성을 빼먹은 것입니다):")
        for k, v in extracted.items():
            print(f" - {k}: {len(v)}자")

        return extracted
        
    except Exception as e:
        print(f"Gemini 데이터 파싱 에러: {e}")
        return {}

# 🌟 가사 쪼개기 도우미 함수를 가장 바깥쪽으로 안전하게 뺐습니다!
def get_chunks(text):
    return [{"text": {"content": text[i:i+2000]}} for i in range(0, max(1, len(text)), 2000)]

def save_to_notion(date_str, genre, prompt, data_dict):
    notion_token = os.environ.get("NOTION_TOKEN")
    database_id = os.environ.get("NOTION_DATABASE_ID")
    
    # 가사가 정말로 비어있다면 아예 전송을 하지 않고 멈춥니다.
    if not notion_token or not database_id or not data_dict.get("lyrics", "").strip(): 
        print("❌ 저장할 가사(LYRICS) 데이터가 비어있어 Notion 호출을 취소합니다.")
        return

    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    page_title = f"{date_str} ({genre})"
    
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
        # 🌟 JSON 파일이 이제 순수 리스트 형태이므로, 뒤에 ["장르"] 같은 것을 붙이면 절대 안 됩니다!
        genres1 = load_data('data/genres1.json')
        genres2 = load_data('data/genres2.json')
        times = load_data('data/times.json')
        emotions1 = load_data('data/emotions1.json')
        actions = load_data('data/actions.json')
        places = load_data('data/places.json')
        emotions2 = load_data('data/emotions2.json')
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return  # 🌟 에러가 나면 여기서 깔끔하게 멈추도록 return을 꼭 넣어주세요.
        
    # 🌟 무한 루프 방지를 위해 최대 재시도 횟수를 설정합니다.
    max_retries = 100 
    retry_count = 0
    
    while retry_count < max_retries:
        selected_genre1 = get_random_item(genres1)
        selected_genre2 = get_random_item(genres2)
        
        # 원하지 않는 조합이 "아니라면" 루프를 탈출합니다.
        if not is_unwanted_combination(selected_genre1, selected_genre2):
            break
            
        retry_count += 1
        print(f"⚠️ 원하지 않는 조합 발생 ({selected_genre1}, {selected_genre2}) -> 다시 뽑습니다.")

    if retry_count == max_retries:
        print("❌ 유효한 장르 조합을 찾는 데 실패했습니다. 원하지 않는 조합(unwanted_pairs) 리스트가 너무 많거나 데이터가 부족한지 확인해 주세요.")
        return

    # 🌟 두 장르를 조합한 최종 장르명 생성 (예: "R&B, House")
    selected_genre = f"{selected_genre1}, {selected_genre2}"
    selected_time = get_random_item(times)
    selected_emotion1 = get_random_item(emotions1)
    selected_action = get_random_item(actions)
    selected_place = get_random_item(places)
    selected_emotion2 = get_random_item(emotions2)

    current_date = datetime.now().strftime("%Y년 %m월 %d일")

    final_prompt = f"""
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
3) [포맷 출력]: 확정한 아이디어를 바탕으로, 시스템 프롬프트에서 요구한 ###DETAIL### 부터 ###UPLOAD### 까지의 8가지 필수 구분자 포맷에 맞추어 완벽한 최종 결과물만 출력해.
</Action_Steps>
"""
    print(f"\n[1] 생성된 프롬프트: {final_prompt}")
    print("\n[2] Gemini 가사 생성 중...")
    
    result_data = generate_lyrics_with_gemini(final_prompt)
    
    print("\n[3] Notion 저장 시도...")
    save_to_notion(current_date, selected_genre, final_prompt, result_data)

if __name__ == "__main__":
    main()
