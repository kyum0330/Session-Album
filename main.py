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
요즘 트렌드를 조사한 후에, 제시된 [장르], [시간], [장소], [감정], [행동], [날씨] 데이터를 활용해, 선택된 두 장르의 비트감과 감정선이 가장 매력적으로 어우러지는 세련된 곡을 만들어야 해요.

[작사 핵심 및 메타 태그 규칙]
1. 보컬 및 페르소나: [Smooth alto female vocal, deep calm voice, low octave, subdued pitch, clean natural voice, clear diction, effortless singing, gentle resonance, subtle vocal runs, relaxed delivery, mellow dynamics, soft instrumentation, chill R&B, Solo]. 
Suno AI가 흔한 중-고음 소프라노를 출력하지 않도록, 과도한 기교 없이 담백하고 매력적인 중저음 보컬 톤을 강제해요. 보컬과 코러스 부분에 대해서는 다음 내용을 참고해주세요.

    1-1. 메타 태그 적용 (Lyrics 영역)
        곡이 고조되는 코러스(후렴구)나 브릿지 부분에 단순히 [Chorus]라고만 적으면 AI가 마음대로 소리를 내지를 확률이 높습니다. 이럴 때는 대괄호 안에 보컬의 창법을 직접 제한해 주세요. 상황에 맞게 아래 태그 중 하나를 선택하여 적용하십시오.
        
- [Soft Chorus]: 부드럽게 부르는 후렴구
- ​[Clear Smooth Falsetto]: 공기 소리를 줄이고 목소리의 선명도를 높인 맑고 부드러운 가성
- ​[Warm Gentle High Notes]: 쨍하지 않고 따뜻하게 감싸듯 올라가는 편안한 고음
- ​[Controlled Vocal]: 감정은 담되 에너지가 과하지 않게 절제된 보컬
- [mellow dynamics]: 튀는 구간 없이 차분하고 부드러운 다이내믹
- [soft vocal delivery]: 처음부터 끝까지 부드럽게 내뱉는 보컬 표현
- [laid-back]: 여유롭고 힘을 뺀 스타일
- [intimate vocal]: 귀에 대고 속삭이듯 가까운 느낌의 보컬

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

    1-5. 고음부에서 AI가 쨍하게 소리를 내지르는 현상(Belting)을 방지하고 싶다면, 아래의 규칙을 엄격히 적용하여 프롬프트를 자동 생성하십시오.

- 도입부/1절 (확실하게 깔아주는 저음): [Low Calm Female Vocal] 또는 [Deep Spoken Vocal]
- 말하듯 힘을 완전히 빼는 파트: [Subdued Vocal]
- 코러스/고음 진입 파트 (에너지 억제): [Controlled Alto Vocal] (음역대를 높이지 않고 중저음역대 안에서 에너지만 살짝 조절하도록 지시합니다.)
              
2. 비트 및 다이내믹 : "R&B, Electro Pop, Moombahton, Synth Pop, Baltimore Club, UK Garage, Hip Hop, Jersey Club, Liquid Drum & Bass, Favela Funk, House, Contemporary R&B, Miami Bass, Old-school Hip Hop, City Pop, New Jack Swing" 중에서 선택된 두 개의 Genre에 알맞게 하단의 비트 및 다이내믹을 참고하여 비트 및 다이내믹을 적용해주면 좋겠어요. 이때 두 개의 장르가 선택되므로, 두 장르의 특징을 하이브리드 형태로 신선하게 믹스하거나, Verse와 Chorus에 각각의 장르적 매력이 교차(예: Verse는 R&B 무드, Chorus는 House 비트)되도록 가사 속 메타 태그(< >)를 창의적으로 조합해 주세요.

### 1. R&B (알앤비)

비트 및 다이내믹 (그루브 및 감정선 설계):

* Verse 파트에서는 리듬 악기를 최소화하고 부드러운 EP(일렉트릭 피아노)와 묵직한 베이스 라인으로 차분한 무드를 조성해요. `<Soft EP, deep bass, minimal beat>`
* Chorus 파트에서는 스네어와 킥 드럼이 정박과 엇박을 오가며 풍성한 코러스 화음과 함께 깊은 그루브를 터뜨려요. `<Deep R&B groove, rich vocal harmony, rhythmic beat>`
* 보컬의 기교와 감정 표현이 돋보일 수 있도록 악기 편곡은 여백을 두고, 멜로디의 유연함을 강조해요.

### 2. Electro Pop (일렉트로 팝)

비트 및 다이내믹 (신시사이저 중심의 댄스 팝 설계):

* Verse 파트에서는 베이스와 가벼운 신스 플럭(Synth pluck) 사운드 위주로 공간감을 주며 리듬을 예열해요. `<Light synth pluck, minimal bass synth>`
* Chorus 파트에서는 경쾌한 4/4박자 킥 드럼과 귀에 꽂히는 강렬한 신시사이저 리드를 배치해 에너지를 폭발시켜요. `<Catchy synth lead, energetic 4/4 dance beat>`
* 꽉 찬 리얼 악기 구성보다는 명확한 훅(Hook) 멜로디와 대중적인 팝 보컬 톤이 곡을 이끌어가도록 매끄럽게 편곡해요.

### 3. Moombahton (뭄바톤)

비트 및 다이내믹 (뎀보우 리듬 설계):

* Verse 파트에서는 스네어(Snare) 사용을 최소화하고, 베이스와 코드에만 뎀보우(Dembow) 노트를 일부 사용하여 미니멀한 여백의 미를 줘요. `<Minimal snare, partial dembow bass>`
* Chorus 파트에서는 정확하고 꽉 찬 타격감의 뎀보우 리듬을 터뜨려 완벽한 뭄바톤(Moombahton) 비트를 완성해요. `<Full dembow rhythm, upbeat moombahton>`
* 악기를 무겁게 쌓지 않고, 보컬의 발음이 타악기처럼 쫀득하게 리듬을 타도록 가사의 글자 수를 세밀하게 맞춰요.

### 4. Synth Pop (신스 팝)

비트 및 다이내믹 (레트로 일렉트로닉 설계):

* Verse 파트에서는 80년대 스타일의 아날로그 신스 베이스가 일정한 8비트로 달리며 몽환적인 분위기를 구축해요. `<Analog synth bass, steady 8-bit rhythm, retro vibe>`
* Chorus 파트에서는 리버브가 강하게 걸린 둔탁한 스네어(Gated Snare)와 반짝이는 신스 아르페지오가 더해져 공간감을 극대화해요. `<Gated snare, sparkling synth arpeggio, dreamy synth pop>`
* 너무 현대적인 클럽 비트를 지양하고, 향수를 자극하는 아날로그 신스 질감 위에서 보컬이 부드럽게 흐르도록 유도해요.

### 5. Baltimore Club (볼티모어 클럽)

비트 및 다이내믹 (브레이크비트 및 보컬 샘플링 설계):

* Verse 파트에서는 짧게 끊어지는 킥 드럼과 독특한 보컬 촙(Vocal chop) 샘플을 잘게 쪼개어 긴장감을 끌어올려요. `<Chopped vocal samples, syncopated kick, fast tempo>`
* Chorus 파트에서는 전형적인 브레이크비트(Breakbeat)와 묵직한 808 베이스가 어우러져 격렬하고 반복적인 댄스 바운스를 형성해요. `<Heavy 808 bass, breakbeat drum loop, aggressive club bounce>`
* 멜로디의 전개보다는 잘게 쪼개지는 드럼 패턴과 최면을 걸 듯 반복되는 샘플링 리듬 자체가 메인 악기가 되도록 세팅해요.

### 6. UK Garage (UK 개러지)

비트 및 다이내믹 (투스텝(2-Step) 리듬 설계):

* Verse 파트에서는 몽환적인 신스 패드 사운드 위로 잘게 쪼개진 하이햇(Hi-hat)과 깊은 서브 베이스가 차갑고 도시적인 무드를 만들어요. `<Deep sub-bass, syncopated hi-hats, atmospheric pad>`
* Chorus 파트에서는 킥 드럼이 정박을 벗어나 엇박으로 떨어지는 특유의 투스텝 리듬이 본격적으로 전개되며 그루브를 만들어내요. `<2-step drum pattern, UK garage groove, bouncy sub-bass>`
* 보컬은 너무 강렬하게 지르지 않고, 속삭이듯 리드미컬하고 세련된 톤을 얹어 차가운 전자 비트와 대비를 줘요.

### 7. Hip Hop (힙합)

비트 및 다이내믹 (무게감 있는 드럼 루프 설계):

* Verse 파트에서는 베이스와 기본적인 드럼 루프만 남기고 여백을 두어 래퍼(혹은 보컬)의 딕션과 라임이 선명하게 꽂히도록 해요. `<Minimal drum loop, prominent vocal flow, sparse bass>`
* Chorus 파트에서는 묵직한 킥과 베이스 사운드를 꽉 채워 청각적인 타격감과 무게감을 극대화해요. `<Heavy bass, hard-hitting kick, strong hip hop beat>`
* 다채로운 화성 악기보다는 드럼의 질감과 베이스의 울림, 그리고 목소리가 뱉는 리듬감이 곡의 기둥이 되도록 편곡해요.

### 8. Jersey Club (저지 클럽)

비트 및 다이내믹 (트리플렛 킥 바운스 설계):

* Verse 파트에서는 템포(130-140 BPM)를 빠르게 유지하면서 가벼운 신스 리드와 쪼개진 보컬 샘플로 속도감을 끌어올려요. `<Fast tempo, chopped vocals, light synth lead>`
* Chorus 파트에서는 저지 클럽 특유의 '쿵-쿵-쿵쿵쿵' 하는 트리플렛(Triplet) 킥 드럼 패턴을 전면에 내세워 강렬한 바운스를 터뜨려요. `<Jersey club triplet kick, heavy bouncy rhythm, energetic dance>`
* 복잡한 멜로디보다는 높은 BPM 위에서 요동치는 킥 드럼의 리듬감과 반복되는 챈트(Chant)가 주는 중독성을 극대화해요.

### 9. Liquid Drum & Bass (리퀴드 드럼 앤 베이스)

비트 및 다이내믹 (고속 브레이크비트 및 공간감 설계):

* Verse 파트에서는 부드럽고 몽환적인 앰비언트 신스 패드 사운드를 넓게 깔아주어 서정적이고 아련한 공간감을 조성해요. `<Ambient synth pad, smooth atmosphere, soft vocal>`
* Chorus 파트에서는 160 BPM 이상의 매우 빠르고 복잡하게 쪼개지는 드럼 비트와 유연하게 롤링하는 베이스를 결합해요. `<Fast breakbeat, rolling bassline, energetic liquid dnb>`
* '폭주하는 초고속 타악기'와 '지극히 차분하고 서정적인 보컬/신스'라는 상반된 두 요소를 완벽하게 대비시켜 몽환적인 미학을 연출해요.

### 10. Favela Funk (파벨라 펑크)

비트 및 다이내믹 (야생적이고 공격적인 타악기 설계):

* Verse 파트에서는 멜로디 악기를 배제하고 브라질 빈민가 특유의 거칠고 원초적인 타악기(Tamborzão) 리듬을 불규칙하게 얹어요. `<Raw percussion, minimal chords, irregular rhythm>`
* Chorus 파트에서는 비트의 텐션을 극도로 끌어올리며 강렬하고 공격적인 타격감과 금속성의 드럼 사운드를 쏟아내요. `<Aggressive favela funk beat, loud metallic percussion, explosive energy>`
* 예쁜 화성보다는 거칠고 날것 그대로의 스트리트 바이브와 합창하듯 내지르는 챈트 형식의 보컬에 포커스를 맞춰요.

### 11. House (하우스)

비트 및 다이내믹 (포 온 더 플로어(Four-on-the-floor) 리듬 설계):

* Verse 파트에서는 킥 드럼 없이 엇박자의 하이햇과 부드러운 하우스 피아노 코드(혹은 신스)만으로 점진적인 빌드업을 유도해요. `<Off-beat hi-hat, house piano chords, building up tension>`
* Chorus 파트에서는 1, 2, 3, 4박자 정위치에 묵직하게 떨어지는 킥 드럼을 중심으로 베이스 라인이 결합되어 완벽한 댄스 그루브를 만들어요. `<Four-on-the-floor kick, groovy house bassline, rhythmic dance beat>`
* 일정한 BPM 위에서 규칙적이고 안정적인 심장 박동 같은 비트를 유지하며 곡의 클라이맥스를 향해 사운드를 겹겹이 쌓아 올려요.

### 12. Contemporary R&B (컨템포러리 R&B)

비트 및 다이내믹 (모던하고 세련된 무드 설계):

* Verse 파트에서는 BPM 80 내외의 느린 템포 위에서 미니멀한 전자 건반과 트랩 기반의 가벼운 하이햇으로 트렌디함을 줘요. `<Slow tempo, smooth keys, minimalist modern R&B beat>`
* Chorus 파트에서는 깊고 풍부한 808 베이스의 서스테인(길게 이어지는 음)과 함께 코러스 보컬의 두터운 화음을 더해 감정을 쏟아내요. `<Deep 808 sustain, thick vocal harmony, emotional R&B climax>`
* 어쿠스틱 악기보다는 질감 좋은 전자음을 사용하고, 보컬의 숨소리 하나까지 섬세하게 들리도록 공간감을 넓고 깊게 써요.

### 13. Miami Bass (마이애미 베이스)

비트 및 다이내믹 (고속 808 바운스 파티 리듬 설계):

* Verse 파트에서는 쉴 새 없이 쪼개지는 하이햇과 리드미컬하고 속도감 있는 보컬 위주로 전개하며 파티의 에너지를 끌어올려요. `<Fast hi-hats, rhythmic vocal delivery, energetic party vibe>`
* Chorus 파트에서는 빠른 템포와 함께 바닥을 강하게 울리는 거대하고 무거운 808 베이스 킥 사운드를 폭발시켜요. `<Heavy 808 bass boom, fast Miami bass rhythm, club dance energy>`
* 진지한 분위기를 배제하고, 무조건 신나게 바운스를 탈 수 있는 극강의 베이스 타격감과 스피드에 믹싱의 포커스를 맞춰요.

### 14. Old-school Hip Hop (올드스쿨 힙합)

비트 및 다이내믹 (클래식 붐뱁(Boom-Bap) 드럼 설계):

* Verse 파트에서는 아날로그 질감의 샘플링(재즈나 소울에서 따온 루프)과 로파이(Lo-fi)한 바이닐(LP) 잡음 위로 랩이 묵직하게 얹혀요. `<Vinyl crackle, sampled jazz loop, steady vocal flow>`
* Chorus 파트에서는 정직하고 둔탁한 '쿵-빡(Boom-Bap)' 드럼 비트에 스크래치 사운드나 브라스 컷을 더해 클래식한 분위기를 연출해요. `<Heavy boom-bap drums, old-school hip hop groove, classic brass sample>`
* 현대적인 트랩 비트를 피하고, 거칠지만 인간적인 그루브가 살아있는 샘플링 기반의 둔탁한 리듬을 활용해요.

### 15. City Pop (시티 팝)

비트 및 다이내믹 (레트로 어쿠스틱 밴드 그루브 설계):

* Verse 파트에서는 찰랑거리는 펑키(Funky)한 웸웸기타(Wah-wah guitar)와 통통 튀는 슬랩 베이스 연주로 여유로운 드라이브 무드를 조성해요. `<Funky rhythm guitar, slap bass, breezy mid-tempo>`
* Chorus 파트에서는 레트로한 신시사이저 브라스(Brass)와 청량한 스트링(Strings) 라인이 폭발하며 화려하고 낭만적인 도시의 감성을 완성해요. `<Retro synth brass, lush strings, sparkling city pop chorus>`
* 기계적인 드럼 머신보다는 실제 밴드 연주자가 그루브를 타는 듯한 유연하고 낭만적인 어쿠스틱/일렉트릭 앙상블을 살려요.

### 16. New Jack Swing (뉴 잭 스윙)

비트 및 다이내믹 (헤비 스윙 바운스 설계):

* Verse 파트에서는 80~90년대 특유의 엇박자 스윙(Shuffle) 리듬을 기반으로, 펑키한 베이스와 통통 튀는 드럼 머신이 흥을 돋워요. `<Heavy swing beat, funky drum machine, groovy R&B vocal>`
* Chorus 파트에서는 리버브가 강한 스네어 타격음과 함께 캐치한 보컬 화음(R&B 아카펠라 스타일)이 터지며 완벽한 뉴 잭 스윙을 구현해요. `<Gated snare, bouncy new jack swing rhythm, upbeat R&B chorus>`
* 단순히 정박으로 걷는 4/4박자가 아니라, 비트 전체가 '출렁거리는(Bouncing)' 강력한 스윙 리듬감을 끝까지 유지하는 것이 핵심이에요.



3. 한영 혼용 훅(Hook): 귀에 확 꽂히는 명확한 멜로디를 위해, Chorus 파트에는 'Vibe', 'Hype', 'Chill'이라는 단어 느낌의 쿨한 무드의 영단어 조사하여 다양하게 한국어와 찰지게 섞어 중독성 있는 펀치라인을 만들어요. 꼭 'Vibe', 'Hype', 'Chill' 단어가 들어갈 필요는 없어요.

4. 이스터 에그 (행동 교차 룰): Verse 파트 중 한 곳에 반드시 '~할 겸' (예: 바람 쐴 겸, 생각 지울 겸 등)이라는 표현을 딱 한 번 자연스럽게 삽입해서 주인공의 무심하고 여유로운 태도를 연출해요.

5. 곡 중간(Bridge 이후 등)에 해당 장르를 가장 잘 나타내는 **<Instrumental Solo> (악기 솔로 구간)**를 최소 1회 이상 강제로 삽입해요.

6. 전체적으로 매 가사 부분마다 보컬에 대한 상세한 내용을 <>을 통해서 최대한 상세하게 표현합니다.

모든 답변은 반드시 아래의 [구분자]를 사용하여 섹션을 나누어 작성해야 해요

###DETAIL###
이 칸에는 노래 제목(Subject), 장르(Genre), Tempo, Key, 악기 구성을 포함한 정보와 작사 배경 및 분위기 구성을 적어주세요. (띄어쓰기 포함 총 800자 이내) 이때 노래 제목은 소재의 나열보다는 키워드 위주로 한개 또는 두개의 단어로 표현해주세요.
* 제목 및 정보 항목에 마크다운 굵게(**)는 절대 사용하지 마요.

###PURPOSE###
이 칸에는 '작사가의 한마디'를 통해 이 곡의 기획 의도와 종합적인 곡 소개를 적어주세요.

###SUNO###
위 DETAIL 부분에 작성한 '장르, Tempo, 악기 구성, 분위기'를 음악 생성 AI(Suno)의 'Style of Music' 란에 바로 복사해 넣을 수 있도록, 영어 키워드 위주로 700~850자로 번역 및 요약해주세요.
이때, 보컬에 관련된 내용은 작성하지 마세요. (예: Melodic Electronic, Progressive House, 123 BPM, warm synth pad, emotional lead)

###VOCAL###
이 칸에는 해당 노래에 어울리는 보컬 스타일을 영어로 작성해주세요. 이때 톤과 스타일에 대해서는 자세하게 적어주세요.
형식: [성별], [톤], [스타일], [솔로/듀엣/그룹 여부]
* 예시: Female vocal, extremely low-pitched, dark contralto, very heavy chest voice, deep androgynous tone, resonant bassy female voice, husky and thick vocal, Solo.
* 전체 내용은 250~280자로 구체적으로 작성할 것.

###LYRICS###
섹션별 가사: Intro, Chorus, Verse, Bridge, Outro 등으로 구분하여 가사를 작성해. 가사 외의 정보(구간 시간, 악기/분위기)는 반드시 영어로 < > 속에 넣어 표현해주세요.

가사 내 지시어 (Meta Tags) 예시: [Extremely low vocal], [Heavy and dark contralto singing], [Deep thick chest voice]

이때 전체 내용은 띄어쓰기와 지시어,가사를 모두 포함하여 가사를 총 4700~4900자로 각 가사의 구간마다 상세하게 보컬의 발성 및 느낌을 잘 표현해주길 바래요.

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
            'models/gemini-3.5-flash',       # 1순위: 가장 똑똑한 최신 주력 모델 (퀄리티 최우선)
            'models/gemini-3.1-flash-lite',  # 2순위: 1순위가 막히면 넉넉한 한도와 속도로 백업!
            'models/gemini-2.5-flash',       # 3순위: 안정적인 2.5 버전
            'models/gemini-flash-latest'
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
