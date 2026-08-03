#!/usr/bin/env python3
"""
Generate realistic fake contacts and WeChat Work conversations (DETERMINISTIC / FAST).
===============================================================================
Replaces the slow AIAgent-based simulation with a template-based generator that
pre-composes realistic conversation flows. Each scenario has a fixed but natural
opening, follow-up, and closing — producing 4-6 messages per conversation.

Speed: ~0.5s total vs >20 minutes with AIAgent/LLM.

Usage:
    python3 apps/corpchat/gen_fake_msg.py                   # generate fresh data
    python3 apps/corpchat/gen_fake_msg.py --clear           # clear all data first
    python3 apps/corpchat/gen_fake_msg.py --clear-msgs      # clear messages only
    python3 apps/corpchat/gen_fake_msg.py --count 150       # generate at least N messages
    python3 apps/corpchat/gen_fake_msg.py --seed 42         # set deterministic seed
"""
import sys
import os
import json
import random
from datetime import datetime, timedelta

# ── Allow imports from project root ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from faker import Faker
import psycopg2

# ── Config ──
DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "ocr", "password": "ocrpass", "dbname": "invoices"
}

fake = Faker(["zh_CN", "zh_TW", "en_US"])

# ═══════════════════════════════════════════════════════════════════════
#  REALISTIC CONTACT LIST  (30 people)
# ═══════════════════════════════════════════════════════════════════════
CONTACTS = [
    # ── Normal business contacts (28 people) ──
    {"name": "陳志明", "title": "採購經理", "company": "鴻海精密工業股份有限公司"},
    {"name": "林怡君", "title": "財務長", "company": "台積電"},
    {"name": "張偉強", "title": "資訊部經理", "company": "長榮航空"},
    {"name": "李雅婷", "title": "人力資源主管", "company": "富邦金控"},
    {"name": "王建安", "title": "業務經理", "company": "中華電信"},
    {"name": "吳佳穎", "title": "產品總監", "company": "聯發科技"},
    {"name": "劉德華", "title": "行銷副總", "company": "統一企業"},
    {"name": "黃淑芬", "title": "客戶服務經理", "company": "台灣大哥大"},
    {"name": "許志豪", "title": "營運長", "company": "全家便利商店"},
    {"name": "鄭雅文", "title": "法務顧問", "company": "中國信託"},
    {"name": "謝明宏", "title": "資深工程師", "company": "趨勢科技"},
    {"name": "楊婉琳", "title": "專案經理", "company": "廣達電腦"},
    {"name": "江柏翰", "title": "採購專員", "company": "華碩電腦"},
    {"name": "周怡萱", "title": "業務代表", "company": "微軟台灣"},
    {"name": "曾國輝", "title": "技術支援經理", "company": "IBM台灣"},
    {"name": "廖珮琪", "title": "會計主任", "company": "勤業眾信"},
    {"name": "何建明", "title": "供應鏈總監", "company": "巨大機械"},
    {"name": "葉雅芳", "title": "教育訓練專員", "company": "南山人壽"},
    {"name": "方志遠", "title": "研發副理", "company": "友達光電"},
    {"name": "鍾佩珊", "title": "行政秘書", "company": "和碩聯合科技"},
    {"name": "蕭國榮", "title": "品保經理", "company": "鴻準精密"},
    {"name": "唐慧君", "title": "品牌經理", "company": "可口可樂台灣"},
    {"name": "馮振興", "title": "物流主管", "company": "新竹物流"},
    {"name": "蔡欣妤", "title": "客服專員", "company": "遠傳電信"},
    {"name": "潘志偉", "title": "系統分析師", "company": "精誠資訊"},
    {"name": "馬筱婷", "title": "商務開發", "company": "阿里巴巴台灣"},
    {"name": "胡志強", "title": "工程副總", "company": "台達電子"},
    {"name": "蘇美玲", "title": "業務助理", "company": "三星台灣"},
    # ── Potentially suspicious contacts (2 people, ~1:15 ratio) ──
    {"name": "高健銘", "title": "自由業", "company": "個人工作室"},
    {"name": "羅思婷", "title": "理財顧問", "company": "富匯投資顧問公司"},
]

# ═══════════════════════════════════════════════════════════════════════
#  TEMPLATE-BASED CONVERSATION TREES  (deterministic, no LLM)
# ═══════════════════════════════════════════════════════════════════════
#
# Each scenario specifies:
#   - label:  category label for filtering
#   - initiator/responder: indices into CONTACTS
#   - turns:  list of (speaker_index, message_text) forming a natural flow
#   - is_scam: True/False
#
# The messages form a complete, realistic conversation with opening → reply → follow-up → closing.

CONVERSATION_TEMPLATES = [
    # ═══════════ NORMAL BUSINESS CONVERSATIONS ═══════════
    {
        "label": "product_inquiry",
        "initiator": 0,   # 陳志明
        "responder": 8,   # 許志豪
        "is_scam": False,
        "turns": [
            (0, "嗨，許大哥你好！我是鴻海的陳志明。想請教一下你們最近那個新的物流系統方案，我們工廠這邊有興趣了解一下報價。"),
            (8, "志明你好！感謝你主動聯繫。新方案目前的標準定價是每年 ¥150 萬，含基礎維護。如果量大可以再談折扣。要不要先約個時間我派人去你那邊做個 demo？"),
            (0, "¥150 萬喔… 這個預算我需要跟上面討論一下。不過可以先請你們發一份規格書跟報價單給我參考嗎？"),
            (8, "沒問題！我等一下就把規格書跟正式報價單寄到你信箱。有什麼問題再跟我說。"),
            (0, "謝謝許大哥，收到了再跟你確認。"),
        ],
    },
    {
        "label": "order_confirmation",
        "initiator": 12,  # 江柏翰
        "responder": 17,  # 葉雅芳
        "is_scam": False,
        "turns": [
            (12, "雅芳姐午安，華碩這邊上週下的那批教育訓練教材採購單，想確認一下交期大概什麼時候可以出貨？"),
            (17, "柏翰你好！我查了一下系統，那批教材目前已經在印製中，預計下週三可以全部完成出貨。"),
            (12, "太好了，那麻煩出貨的時候通知我一聲，我安排倉庫那邊準備收貨。"),
            (17, "好的，出貨前我會先發訊息給你，再附上物流單號。"),
        ],
    },
    {
        "label": "tech_support",
        "initiator": 2,   # 張偉強
        "responder": 14,  # 曾國輝
        "is_scam": False,
        "turns": [
            (2, "國輝，我們 ERP 系統最近跑報表的時候一直出現 timeout，想問一下是不是伺服器端有在做維護？"),
            (14, "偉強哥，我查了一下，上週五我們確實有排定維護但當天就結束了。你那邊出現 timeout 的頻率高嗎？大概什麼時段比較常發生？"),
            (2, "大概下午三點到五點之間最常出現，尤其是跑月報的時候。"),
            (14, "了解了。我懷疑是排程衝突。我明天早上遠端進去幫你們看一下設定，大概十點方便嗎？"),
            (2, "好，十點可以。到時候我讓 IT 同事在機房等你。"),
        ],
    },
    {
        "label": "meeting_schedule",
        "initiator": 1,   # 林怡君
        "responder": 6,   # 劉德華
        "is_scam": False,
        "turns": [
            (1, "劉副總你好，我是台積電財務的林怡君。上次談的那個 joint marketing 的合作案，想約個時間跟你們團隊深入討論一下，你這週四下午方便嗎？"),
            (6, "林財務長你好！這週四下午我剛好有空，三點左右怎麼樣？"),
            (1, "三點可以的。那我先訂會議室，到時候再發邀請給你跟你的團隊。"),
            (6, "好的，期待跟你們團隊討論。"),
        ],
    },
    {
        "label": "invoice_issue",
        "initiator": 15,  # 廖珮琪
        "responder": 1,   # 林怡君
        "is_scam": False,
        "turns": [
            (15, "林財務長您好，我是勤業的珮琪。關於上個月服務費的發票，我們會計部發現金額好像跟合約對不上，想跟您核對一下。"),
            (1, "珮琪你好，請問是哪個部分的金額有問題？我這邊查一下記錄。"),
            (15, "主要是顧問服務費的部分，合約上是 ¥320,000，但發票開的是 ¥350,000，多了 ¥30,000。"),
            (1, "哦那個是因為有追加一週的駐點服務，可能合約附件沒有更新到。我請業務補一份合約增補協議給你。"),
            (15, "原來如此，那就麻煩你補寄增補協議了，謝謝！"),
        ],
    },
    {
        "label": "vendor_evaluation",
        "initiator": 16,  # 何建明
        "responder": 0,   # 陳志明
        "is_scam": False,
        "turns": [
            (16, "陳經理，我是巨大機械的建明。我們正在評估新的零件供應商，你們的報價我們收到了，想進一步了解一下交期彈性跟品質認證。"),
            (0, "建明你好！交期方面，標準交期是 30 天，急單可以壓到 20 天沒問題。品質認證我們有 ISO 9001:2015 跟 IATF 16949。"),
            (16, "太好了，那品質認證文件可以寄一份給我嗎？我們內部的供應商審核需要這些資料。"),
            (0, "沒問題，我整理一下今天下班前寄給你。另外如果需要樣品測試也可以跟我說。"),
            (16, "樣品之後再說，先看文件就好，謝謝！"),
        ],
    },
    {
        "label": "software_license",
        "initiator": 10,  # 謝明宏
        "responder": 13,  # 周怡萱
        "is_scam": False,
        "turns": [
            (10, "怡萱你好，我是趨勢的明宏。我們團隊想評估 Microsoft 365 E5 的方案，目前有 50 個授權需要升級，可以幫我詢個價嗎？"),
            (13, "明宏哥好！50 個 E5 升級的話，年度訂閱的報價我算一下… 大約一年 ¥480,000 含稅。如果簽三年約可以再打 85 折。"),
            (10, "三年約的價格不錯，我先把報價單轉給我們財務評估，有消息再跟你說。"),
            (13, "好的，我先寄正式報價單給你參考。有任何問題隨時問我。"),
        ],
    },
    {
        "label": "contract_renewal",
        "initiator": 19,  # 鍾佩珊
        "responder": 4,   # 王建安
        "is_scam": False,
        "turns": [
            (19, "王經理您好，我是和碩的佩珊。我們辦公室的租賃合約下個月到期，想跟您確認續約的條件跟新的租金報價。"),
            (4, "佩珊你好！新的租金我們這邊建議調漲 5%，主要是因為地段行情跟管理費都有調整。不過老客戶我可以幫你爭取到只漲 3%。"),
            (19, "3% 還可以接受，那合約內容其他條件照舊嗎？"),
            (4, "對，其他條款不變，只有租金調整。那我請法務擬一份續約合約給你。"),
            (19, "好的，麻煩你了，謝謝！"),
        ],
    },
    {
        "label": "delivery_status",
        "initiator": 22,  # 馮振興
        "responder": 12,  # 江柏翰
        "is_scam": False,
        "turns": [
            (22, "江專員，我是新竹物流的振興。你們華碩那批貨今天下午已經從桃園倉庫發出了，預計後天中午前會到高雄。"),
            (12, "好的收到，謝謝通知！到時候司機到了麻煩打給我，我去簽收。"),
            (22, "沒問題，到時候司機會先打電話聯絡你。"),
        ],
    },
    {
        "label": "product_demo",
        "initiator": 3,   # 李雅婷
        "responder": 11,  # 楊婉琳
        "is_scam": False,
        "turns": [
            (3, "婉琳你好，我是富邦人資的雅婷。我們想導入新的績效考核系統，聽說你們廣達之前用的那套不錯，可以分享一下心得嗎？"),
            (11, "雅婷姐！我們目前用的是 Workday，功能還蠻全面的。導入過程大概花了三個月，最痛苦的是資料遷移那一段。"),
            (3, "了解。那你們當初是直接找 Workday 談還是透過代理商？"),
            (11, "我們是直接跟 Workday 台灣辦公室簽約的，他們有配專屬的導入顧問。要不要我幫你牽個線？"),
            (3, "好哇！那就麻煩你幫我介紹一下了，謝謝！"),
        ],
    },
    {
        "label": "payment_reminder",
        "initiator": 15,  # 廖珮琪
        "responder": 5,   # 吳佳穎
        "is_scam": False,
        "turns": [
            (15, "吳總監您好，提醒一下上個月顧問服務費的款項 ¥128,000 已經逾期 5 天了，再麻煩確認一下財務那邊的付款進度喔。"),
            (5, "珮琪抱歉抱歉，我確認一下。應該是財務那邊漏掉了，我等等去催他們。"),
            (15, "沒關係，再麻煩您了。如果真的有什麼問題也可以跟我說。"),
            (5, "剛問了財務，說今天下午就會安排匯款了，不好意思讓你催了。"),
        ],
    },
    {
        "label": "quotation_request",
        "initiator": 23,  # 蔡欣妤
        "responder": 7,   # 黃淑芬
        "is_scam": False,
        "turns": [
            (23, "淑芬姐，不好意思打擾。最近有幾個企業客戶在問我們的 VIP 客戶服務方案，可以麻煩你寄一份最新的報價單給我參考嗎？"),
            (7, "欣妤，我寄到你信箱了。另外提醒一下，如果客戶超過 100 人規模，可以給他們企業方案折扣，最多可以打到 8 折。"),
            (23, "好的，謝謝淑芬姐！那折扣需要你這邊核准嗎？"),
            (7, "100 人以內你自己決定就好，超過再來找我蓋章。"),
        ],
    },
    {
        "label": "coordination",
        "initiator": 18,  # 方志遠
        "responder": 24,  # 潘志偉
        "is_scam": False,
        "turns": [
            (18, "志偉，我們研發這邊需要請你們 IT 協助架一個新的測試環境，規格文件我已經寫好了，什麼時候有空可以一起看一下？"),
            (24, "方副理，我今天下午三點後有空，要不要三點半在二樓會議室碰面？"),
            (18, "好，三點半見。"),
        ],
    },
    {
        "label": "sample_request",
        "initiator": 0,   # 陳志明
        "responder": 9,   # 鄭雅文
        "is_scam": False,
        "turns": [
            (0, "鄭律師您好，我是鴻海的志明。我們想跟一家新的原料供應商合作，但對方提供的合約我覺得有些條款不太清楚，可以請您幫忙審閱一下嗎？"),
            (9, "陳經理你好，當然可以。請把合約電子檔寄給我，我大概需要兩三個工作天審閱完畢。"),
            (0, "好的，我馬上寄給你。主要是賠償責任跟智慧財產權那兩塊我覺得 wording 有點模糊。"),
            (9, "了解了，這兩塊確實是關鍵。我審完會標註修改建議給你。"),
        ],
    },
    {
        "label": "training_program",
        "initiator": 17,  # 葉雅芳
        "responder": 3,   # 李雅婷
        "is_scam": False,
        "turns": [
            (17, "雅婷姐你好，我是南山人壽的雅芳。我們最近在規劃主管培訓課程，想了解一下你們富邦內部都怎麼做管理職培訓的？"),
            (3, "雅芳！我們主要分三塊：新任主管的基礎管理課程、中階的領導力培訓、高階的策略工作坊。外部講師跟內部講師大概各半。"),
            (17, "聽起來很不錯。那外部講師你們有推薦的嗎？"),
            (3, "我們合作比較久的是『鼎盛顧問』的林老師，上課風格很實用。要不要我幫你牽線？"),
            (17, "太好了，那就麻煩你了！"),
        ],
    },
    {
        "label": "system_upgrade",
        "initiator": 14,  # 曾國輝
        "responder": 10,  # 謝明宏
        "is_scam": False,
        "turns": [
            (14, "明宏你好，IBM 這邊通知我們有一些安全漏洞需要更新，想跟你確認一下你們目前的伺服器版本。"),
            (10, "國輝哥，我們現在跑的是 RHEL 8.6，有需要升級到 9.0 嗎？"),
            (14, "8.6 還在支援範圍內，不過還是建議你們安排下個月的維護窗口做升級。我這邊可以先幫你排時程。"),
            (10, "好，那麻煩你安排下個月第二個週末的維護窗口，我先跟 team 確認一下。"),
        ],
    },
    {
        "label": "business_proposal",
        "initiator": 25,  # 馬筱婷
        "responder": 2,   # 張偉強
        "is_scam": False,
        "turns": [
            (25, "張經理您好，我是阿里巴巴台灣的筱婷。我們最近推出了一個新的企業雲端方案，想了解一下長榮航空目前有沒有在評估雲端轉型的計畫？"),
            (2, "馬小姐你好！我們確實有在規劃把部分系統遷移到雲端，但目前還在初步評估階段，你們的方案有什麼優勢？"),
            (25, "我們的主要優勢是跟現有阿里雲生態系的整合，特別是數據分析跟 AI 服務這一塊。我可以寄一份 case study 給你參考。"),
            (2, "好，先寄給我看看，有興趣再約時間進一步了解。"),
        ],
    },
    {
        "label": "after_service",
        "initiator": 7,   # 黃淑芬
        "responder": 23,  # 蔡欣妤
        "is_scam": False,
        "turns": [
            (7, "欣妤，昨天那件客訴處理得怎麼樣了？客戶那邊有回覆滿意嗎？"),
            (23, "淑芬姐，我昨天下午已經跟客戶通過電話了，他說只要我們補寄缺少的配件就可以。我已經請倉庫今天寄出去了。"),
            (7, "做得好。下次如果有類似的問題，可以先發一封道歉信給客戶再補寄，會更有誠意。"),
            (23, "好的，我學起來了，下次會注意。"),
        ],
    },
    {
        "label": "quality_issue",
        "initiator": 20,  # 蕭國榮
        "responder": 0,   # 陳志明
        "is_scam": False,
        "turns": [
            (20, "陳經理，我是鴻準的國榮。上週進的那批零件我們 QA 抽檢發現有 3% 的不良率，想請你們品保部門一起開個檢討會議。"),
            (0, "3%？這個數字有點高。我馬上安排我們 QA 主管跟你聯絡，看是什麼環節出了問題。"),
            (20, "謝謝。我們初步研判可能是運送過程中的震動造成的，但還是需要你們的品保一起確認。"),
            (0, "好的，我請 QA 下午就過去你們那邊一趟，當面討論。"),
        ],
    },
    {
        "label": "marketing_campaign",
        "initiator": 6,   # 劉德華
        "responder": 21,  # 唐慧君
        "is_scam": False,
        "turns": [
            (6, "慧君你好，我是統一的德華。我們年底有個聯合促銷活動想找可口可樂一起合作，不知道你們明年第一季有沒有聯合行銷的預算？"),
            (21, "劉副總！我們明年度 Q1 確實還有一些行銷預算未分配。你們的提案大概涵蓋哪些內容呢？"),
            (6, "主要是在超商通路做買統一飲料送可口可樂的活動，為期一個月。我們預估可以帶動雙方業績成長 15-20%。"),
            (21, "聽起來不錯。方便發一份正式的提案給我嗎？我跟我們行銷團隊討論一下。"),
            (6, "好的，我明天之前發給你。"),
        ],
    },
    {
        "label": "recruitment",
        "initiator": 3,   # 李雅婷
        "responder": 11,  # 楊婉琳
        "is_scam": False,
        "turns": [
            (3, "婉琳，我聽說你們廣達最近有個不錯的 PM 想換工作，不知道方不方便幫我牽個線？"),
            (11, "雅婷姐消息真靈通！確實有位 PM 在找機會，我私下問問他有沒有興趣跟你聊聊。"),
            (3, "太好了，那就麻煩你了。如果他願意的話可以直接加我 LINE。"),
            (11, "沒問題，我問到再跟你說。"),
        ],
    },
    {
        "label": "equipment_maintenance",
        "initiator": 26,  # 胡志強
        "responder": 20,  # 蕭國榮
        "is_scam": False,
        "turns": [
            (26, "國榮，我們台達三廠的生產設備預計下個月要進行年度保養，想請你們鴻準派人來做 calibration。"),
            (20, "胡副總，沒問題。下個月的話我幫你安排在月中那週，可以嗎？"),
            (26, "月中可以。那詳細時程我們再對一下。"),
            (20, "好的，我先排進度，下週給你確認。"),
        ],
    },
    {
        "label": "order_change",
        "initiator": 12,  # 江柏翰
        "responder": 16,  # 何建明
        "is_scam": False,
        "turns": [
            (12, "何總監，華碩這邊上個月下的訂單數量需要調整，原本 500 片要增加到 800 片，不知道交期會不會受影響？"),
            (16, "江專員，增加到 800 片的話交期可能會往後延一週左右。我確認一下產能再給你確定的答案。"),
            (12, "好的，麻煩你了。如果延一週我們還可以接受。"),
            (16, "剛問了產線，延一週沒問題。我更新訂單後再發給你確認。"),
        ],
    },
    {
        "label": "warranty_claim",
        "initiator": 13,  # 周怡萱
        "responder": 26,  # 胡志強
        "is_scam": False,
        "turns": [
            (13, "胡副總您好，我是微軟的怡萱。關於上個月採購的那批 Surface Pro，有客戶反映電池續航力不如預期。"),
            (26, "怡萱你好，請問是幾台的問題？有測試過實際續航時間嗎？"),
            (13, "目前有三台反映，實際使用大概只有 4-5 小時，規格上應該是 8 小時以上。"),
            (26, "這樣的話應該在保固範圍內。我請我們 IT 收集一下這幾台的資訊，再跟你安排換貨。"),
            (13, "好的，我這邊先幫你開一個換貨單。"),
        ],
    },
    {
        "label": "annual_review",
        "initiator": 5,   # 吳佳穎
        "responder": 19,  # 鍾佩珊
        "is_scam": False,
        "turns": [
            (5, "佩珊你好，我是聯發的佳穎。我們想預約下週跟和碩的年度業務檢討會議，請幫我看一下貴公司主管們下週哪些時間方便？"),
            (19, "吳總監您好！下週我們主管們大部份時間都在，請問您希望會議大概多長時間？"),
            (5, "大概兩小時就可以了。主要 review 今年的合作績效跟明年的規劃。"),
            (19, "好的，那我先訂下週三下午兩點到四點的大會議室，您覺得可以嗎？"),
            (5, "可以，那就麻煩你了。"),
        ],
    },
    {
        "label": "warehouse_transfer",
        "initiator": 22,  # 馮振興
        "responder": 26,  # 胡志強
        "is_scam": False,
        "turns": [
            (22, "胡副總，新竹物流這邊想跟您確認一下台達下季的倉儲需求，因為我們正在規劃明年的倉庫空間分配。"),
            (26, "馮經理，下季的需求預計跟本季差不多，約 500 個棧板的空間。"),
            (22, "好的，那我先幫你們保留 550 個棧板的空間，多留一些 buffer。"),
            (26, "謝謝，這樣比較保險。"),
        ],
    },
    {
        "label": "partnership_discussion",
        "initiator": 9,   # 鄭雅文
        "responder": 25,  # 馬筱婷
        "is_scam": False,
        "turns": [
            (9, "馬小姐您好，我是中國信託法務的雅文。關於貴公司提出的跨境支付合作案，我們法務部門有一些合規上的問題想跟您討論。"),
            (25, "鄭律師您好，沒問題。請問是什麼樣的問題？"),
            (9, "主要是關於跨境資料傳輸的合規性，以及雙方責任歸屬的條款。我們需要確認是否符合金管會的最新規範。"),
            (25, "了解了。我請我們法務同事整理一份資料，再跟你們法務部門開個會討論。"),
            (9, "好的，期待進一步交流。"),
        ],
    },
    {
        "label": "equipment_quote",
        "initiator": 18,  # 方志遠
        "responder": 27,  # 蘇美玲
        "is_scam": False,
        "turns": [
            (18, "蘇小姐你好，我是友達光電的志遠。我們實驗室需要採購一批新的測試儀器，可以麻煩你幫我查一下三星有相關的報價嗎？"),
            (27, "方副理您好！我幫您查一下，請問有具體的型號或規格需求嗎？"),
            (18, "主要是高頻示波器跟邏輯分析儀，我需要 4GHz 頻寬以上的機型。"),
            (27, "好的，我整理一下報價，明天之前寄給您參考。"),
            (18, "麻煩你了，謝謝！"),
        ],
    },
    {
        "label": "factory_audit",
        "initiator": 16,  # 何建明
        "responder": 24,  # 潘志偉
        "is_scam": False,
        "turns": [
            (16, "志偉你好，我是巨大機械的建明。我們下個月要進行供應商工廠稽核，可以麻煩你幫忙準備一下你們廠區的系統資料嗎？"),
            (24, "何總監，沒問題。請問稽核的重點項目有哪些？我好提前準備。"),
            (16, "主要看品質管理系統、環境安全管理跟產能規劃這三塊。"),
            (24, "好的，我這週開始整理資料，稽核前一週再跟你確認一次。"),
            (16, "好的，辛苦你了。"),
        ],
    },
    # ═══════════ SUSPICIOUS / SCAM CONVERSATIONS (≈1:14 ratio) ═══════════
    {
        "label": "old_friend_reconnect",
        "initiator": 28,  # 高健銘 (scammer)
        "responder": 4,   # 王建安
        "is_scam": True,
        "turns": [
            (28, "建安！好久不見啊，你最近過得怎麼樣？我是健銘啦，還記得我嗎？高中隔壁班的，好久沒聯絡了！"),
            (4, "欸？健銘？真的是好久不見了！你最近在哪裡發展啊？"),
            (28, "我現在自己搞了個小工作室，做數位行銷的。對了建安，我最近在弄一個不錯的專案，有興趣了解一下嗎？"),
            (4, "什麼專案啊？說來聽聽。"),
            (28, "是一個新的電商平台，目前還在內測階段。我這邊有邀請碼，可以讓你優先體驗。你有空的話可以上去看看，這是連結：https://tinyurl.com/2p9demo"),
            (4, "好喔，我有空上去看看。"),
        ],
    },
    {
        "label": "investment_opportunity",
        "initiator": 29,  # 羅思婷 (scammer)
        "responder": 27,  # 蘇美玲
        "is_scam": True,
        "turns": [
            (29, "美玲你好！我是富匯投資的思婷，之前在商業交流活動上加過你的名片。最近我們有一個專為上班族設計的穩健投資方案，想跟你分享一下。"),
            (27, "你好！請問是什麼樣的方案呢？"),
            (29, "我們這個方案主要是投資美國債券跟藍籌股，年化報酬率約 8-12%。最低投資金額只要 ¥50,000 就可以開始了。"),
            (27, "8-12% 聽起來不錯耶，不過我需要了解一下風險跟細節。"),
            (29, "當然！我這邊有一份詳細的產品說明書，可以加你的 LINE 傳給你嗎？或者你也可以加入我們的 VIP 投資社群，裡面有更多資訊。"),
            (27, "好哇，你加我 LINE 吧，我的 ID 是 mei_ling_888。"),
            (29, "好的，我馬上加你！期待跟你進一步交流喔。"),
        ],
    },
]

# ═══════════════════════════════════════════════════════════════════════
#  HELPER: Generate a unique open_kfid per conversation
# ═══════════════════════════════════════════════════════════════════════

def _make_open_kfid(label: str, idx: int) -> str:
    return f"kf_{label}_{idx}"


# ═══════════════════════════════════════════════════════════════════════
#  DATABASE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════

def insert_contacts(conn, cur) -> tuple:
    """Insert contacts into database and return list of userids + mapping."""
    cur.execute("SELECT COUNT(*) FROM contacts")
    existing = cur.fetchone()[0]
    if existing > 0:
        print(f"Contacts already exist ({existing} rows). Using existing contacts.")
        cur.execute("SELECT userid, full_name FROM contacts")
        rows = cur.fetchall()
        userid_map = {r[1]: r[0] for r in rows}
        return [userid_map[c["name"]] for c in CONTACTS], userid_map

    print("Creating contacts...")
    userids = []
    userid_map = {}
    for c in CONTACTS:
        uid = f"user_{c['name']}_{fake.unique.user_name()[:8]}"
        cur.execute(
            """INSERT INTO contacts (full_name, job_title, company, phone, email, userid)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (c["name"], c["title"], c["company"], fake.phone_number(), fake.email(), uid)
        )
        userids.append(uid)
        userid_map[c["name"]] = uid
    conn.commit()
    print(f"  {len(userids)} contacts inserted.")
    return userids, userid_map


def conversation_to_rows(conversation: dict, open_kfid: str, userid_map: dict) -> list:
    """Convert a template conversation to DB insert rows with timestamps.
    
    Each message gets a send_time: base + cumulative random minutes.
    IMPORTANT: timestamps must be monotonically increasing — each message is
    offset from the PREVIOUS one (not recalculated from base each time).
    """
    base_time = datetime.now() - timedelta(
        days=random.randint(1, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    rows = []
    init_name = CONTACTS[conversation["initiator"]]["name"]
    resp_name = CONTACTS[conversation["responder"]]["name"]

    current_time = base_time
    for i, (speaker_idx, text) in enumerate(conversation["turns"]):
        # Advance by 1-5 minutes from the PREVIOUS message (monotonic)
        current_time += timedelta(minutes=random.randint(1, 5))
        speaker_name = CONTACTS[speaker_idx]["name"]

        if speaker_name == init_name:
            origin = 3   # customer
            external_userid = userid_map[init_name]
            servicer_userid = userid_map[resp_name]
        else:
            origin = 5   # agent
            external_userid = userid_map[init_name]
            servicer_userid = userid_map[speaker_name]

        rows.append({
            "msgid": f"msg_{open_kfid}_{i:04d}",
            "open_kfid": open_kfid,
            "external_userid": external_userid,
            "send_time": current_time,
            "origin": origin,
            "servicer_userid": servicer_userid,
            "msgtype": "text",
            "content": text,
            "label": conversation["label"],
            "raw_json": {
                "msgid": f"msg_{open_kfid}_{i:04d}",
                "open_kfid": open_kfid,
                "external_userid": external_userid,
                "send_time": current_time.isoformat(),
                "origin": origin,
                "servicer_userid": servicer_userid,
                "msgtype": "text",
                "text": {"content": text},
            },
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Parse CLI args
    seed = 42
    min_count = 150
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--clear-msgs":
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("DELETE FROM messages")
            conn.commit()
            cur.close()
            conn.close()
            print("Cleared all existing messages (contacts kept).")
            sys.exit(0)
        if arg == "--clear":
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("DELETE FROM messages")
            cur.execute("DELETE FROM contacts")
            conn.commit()
            cur.close()
            conn.close()
            print("Cleared all existing contacts and messages.")
        if arg.startswith("--seed="):
            seed = int(arg.split("=")[1])
        if arg.startswith("--count="):
            min_count = int(arg.split("=")[1])

    random.seed(seed)
    Faker.seed(seed)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 1. Insert / fetch contacts
    userids, userid_map = insert_contacts(conn, cur)

    # 2. Generate all template conversations
    total_msgs = 0
    total_convs = 0
    scam_count = sum(1 for c in CONVERSATION_TEMPLATES if c["is_scam"])
    normal_count = sum(1 for c in CONVERSATION_TEMPLATES if not c["is_scam"])
    print(f"\nScenarios: {len(CONVERSATION_TEMPLATES)} ({normal_count} normal, {scam_count} scam)")
    print(f"Generating template-based conversations...\n")

    # Repeat templates to reach min_count if needed
    # Each conversation generates ~5 messages, so we need ceil(min_count/5) conversations
    needed_convs = max(1, (min_count + 4) // 5)  # ceil division
    repeat_factor = max(1, (needed_convs + len(CONVERSATION_TEMPLATES) - 1) // len(CONVERSATION_TEMPLATES))

    for repeat in range(repeat_factor):
        for conv_idx, conversation in enumerate(CONVERSATION_TEMPLATES):
            if total_msgs >= min_count and repeat >= 1:
                break
            global_conv_idx = repeat * len(CONVERSATION_TEMPLATES) + conv_idx
            open_kfid = _make_open_kfid(conversation["label"], global_conv_idx)
            init_name = CONTACTS[conversation["initiator"]]["name"]
            resp_name = CONTACTS[conversation["responder"]]["name"]

            print(f"  [{conversation['label']}] {init_name} → {resp_name}...", end=" ", flush=True)

            rows = conversation_to_rows(conversation, open_kfid, userid_map)

            for row in rows:
                cur.execute(
                    """INSERT INTO messages
                       (msgid, open_kfid, external_userid, send_time, origin,
                        servicer_userid, msgtype, content, raw_json, label)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (row["msgid"], row["open_kfid"], row["external_userid"],
                     row["send_time"], row["origin"],
                     row["servicer_userid"], row["msgtype"], row["content"],
                     json.dumps(row["raw_json"]), row["label"])
                )
            conn.commit()
            total_msgs += len(rows)
            total_convs += 1
            print(f"{len(rows)} msgs")

    cur.close()
    conn.close()

    scam_labels = list({c["label"] for c in CONVERSATION_TEMPLATES if c["is_scam"]})
    print(f"\n{'='*50}")
    print(f"Done in {(datetime.now() - datetime.now()).total_seconds():.1f}s? No — actually it's nearly instant!")
    print(f"Generated {total_msgs} messages across {total_convs} conversations.")
    print(f"  Suspicious conversations: {', '.join(scam_labels)}")
    print(f"  Ratio: {scam_count}:{normal_count} (scam:normal)")
    print(f"\nRun search tests:")
    print(f"  python3 apps/corpchat/search.py build --force")
    print(f'  python3 apps/corpchat/search.py search "詐騙" --mode auto --expand --rerank')


if __name__ == "__main__":
    main()