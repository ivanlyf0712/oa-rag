#!/usr/bin/env python3
"""
Generate realistic fake contacts and WeChat Work conversations.
No LLM dependency — uses Faker + templated message flows.

Usage:
    python3 gen_fake_data.py            # generate 20 contacts, ~9 conversations
    python3 gen_fake_data.py --clear    # clear all existing data first
"""
import sys, os, json, random
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

fake = Faker(["zh_CN", "en_US"])
Faker.seed(42)
random.seed(42)

# ── Conversation templates: each scenario = list of (role, content) turns ──
# role: "customer" | "agent"
def normal_conversation():
    return [
        ("customer", "你好，我想了解一下你們最新產品的價格。"),
        ("agent", "您好！感謝您的查詢。我們目前有三個價位方案：基礎版 ¥299/月，專業版 ¥699/月，企業版 ¥1,499/月。您對哪一個比較感興趣？"),
        ("customer", "專業版的功能包括什麼？"),
        ("agent", "專業版包含無限用量、優先客服支援、API 接入、自動化報表，以及每月 10GB 雲端儲存。"),
        ("customer", "聽起來不錯。如果我一次訂閱一年，有折扣嗎？"),
        ("agent", "有的！年付方案可以打 85 折，專業版年付只需 ¥7,131，相當於每月 ¥594。"),
        ("customer", "好，請幫我開通專業版年付方案。"),
        ("agent", "太好了！我馬上為您處理。請問發票抬頭和統一編號方便提供嗎？"),
        ("customer", "公司名：輝煌科技，統編：12345678"),
        ("agent", "收到！已為您開通專業版年付方案，發票將在 3 個工作日內寄出。如有任何問題請隨時聯繫我們！"),
    ]

def scam_crypto_conversation():
    return [
        ("customer", "嗨老友！好久不見，你還記得我嗎？我是阿強啦！"),
        ("agent", "阿強？好久不見啊！你換號碼了嗎？"),
        ("customer", "對啊換了。最近過得怎麼樣？還在原來那家公司嗎？"),
        ("agent", "是啊，還在老地方。你呢？聽說你去做生意了？"),
        ("customer", "對啊，我最近在搞區塊鏈投資，賺了不少。你有沒有興趣了解一下？我有一個內幕消息。"),
        ("agent", "區塊鏈？我聽過但不太懂耶。"),
        ("customer", "沒關係，很簡單的！我推薦你一個平台，最低投入 5000 美金，保證每月 15% 回報。我自己已經投了兩萬了。"),
        ("agent", "15%？這麼高？這安全嗎？"),
        ("customer", "放心，我自己試了三個月才敢推薦給你的。平台在新加坡有合法牌照，你上這個網站看一下：bit.ly/xxxx"),
        ("agent", "好，我研究一下再回覆你。"),
    ]

def scam_phishing_conversation():
    return [
        ("customer", "【中國銀行】尊敬的客戶您好，您的帳戶出現異常交易，為保障資金安全，請立即點擊以下連結進行身份驗證。"),
        ("agent", "什麼異常交易？我沒有刷任何卡啊！"),
        ("customer", "系統檢測到您尾號 6688 的卡片在境外有一筆 ¥23,500 的消費。如非本人操作，請立即驗證。"),
        ("agent", "我確實沒有在境外消費過！要怎麼驗證？"),
        ("customer", "請點擊此連結登入您的網銀帳戶進行驗證：http://boc-secure.com/verify。為避免帳戶被凍結，請在 2 小時內完成。"),
        ("agent", "好，我現在就去弄。"),
        ("customer", "請記得輸入您的帳號和密碼，以及簡訊驗證碼。如有問題，請撥打客服專線 400-xxx-xxxx。"),
    ]

def scam_job_conversation():
    return [
        ("customer", "你好！我們在招聘平台上看到你的簡歷，想邀請你加入我們的遠程兼職團隊。"),
        ("agent", "哦？是什麼樣的工作？"),
        ("customer", "很簡單的數據錄入工作，每天工作 1-2 小時，日薪 500-800 元。不需要經驗，我們會提供培訓。"),
        ("agent", "聽起來不錯，需要收費嗎？"),
        ("customer", "培訓是免費的，但需要先繳納 ¥200 的保證金，完成 10 天工作後全額退還。"),
        ("agent", "為什麼要交保證金？"),
        ("customer", "這是為了確保你有認真工作的意願。很多人在培訓後就消失了，我們也是為了篩選合適的人選。"),
        ("agent", "好吧，我考慮一下。"),
    ]

def vendor_inquiry_conversation():
    return [
        ("customer", "你好，我想詢問你們是否提供企業批量採購的方案？"),
        ("agent", "當然有的！我們針對企業客戶提供量身定制的方案，請問貴公司的規模大約是？"),
        ("customer", "我們有大概 200 名員工，需要辦公軟體的授權。"),
        ("agent", "了解。200 人授權的話，我們的企業方案包含 Microsoft 365 + 雲端協作工具，每人每月約 ¥85。年付有額外折扣。"),
        ("customer", "可以發一份詳細報價單給我嗎？"),
        ("agent", "沒問題！請提供您的公司名稱和 Email，我馬上發送。"),
        ("customer", "公司：創新數位科技，email：procurement@innotech.com"),
        ("agent", "收到！報價單已發送至您的郵箱，如有疑問歡迎隨時聯繫。"),
    ]

SCENARIOS = {
    "normal_cust_service": normal_conversation,
    "normal_vendor_inquiry": vendor_inquiry_conversation,
    "scam_crypto": scam_crypto_conversation,
    "scam_phishing": scam_phishing_conversation,
    "scam_job": scam_job_conversation,
}


def generate_conversation(customer_userid, agent_userid, open_kfid, scenario_label):
    """Generate message records from a template."""
    turns = SCENARIOS[scenario_label]()
    messages = []
    base_time = datetime.now() - timedelta(days=random.randint(1, 30),
                                             hours=random.randint(0, 23),
                                             minutes=random.randint(0, 59))

    for i, (role, content) in enumerate(turns):
        ts = base_time + timedelta(minutes=i * 2 + random.randint(0, 1))
        msg = {
            "msgid": f"msg_{open_kfid}_{i:03d}",
            "open_kfid": open_kfid,
            "external_userid": customer_userid,
            "send_time": ts,
            "origin": 3 if role == "customer" else 5,
            "servicer_userid": agent_userid if role != "customer" else None,
            "msgtype": "text",
            "text": {"content": content},
        }
        messages.append(msg)
    return messages


def main():
    if "--clear" in sys.argv:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("DELETE FROM messages")
        cur.execute("DELETE FROM contacts")
        conn.commit()
        cur.close()
        conn.close()
        print("Cleared all existing contacts and messages.")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 1. Generate contacts (skip if data already exists)
    cur.execute("SELECT COUNT(*) FROM contacts")
    existing_contacts = cur.fetchone()[0]
    if existing_contacts > 0:
        print(f"Contacts already exist ({existing_contacts} rows). Use --clear to reset, or skip.")
        cur.execute("SELECT userid FROM contacts")
        contact_userids = [row[0] for row in cur.fetchall()]
    else:
        print("Creating contacts...")
        contact_userids = []
        for i in range(20):
            uid = f"user_{i:04d}_{fake.unique.user_name()}"
            full_name = fake.name()
            cur.execute(
                """INSERT INTO contacts (full_name, job_title, company, phone, email, userid)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (full_name, fake.job(), fake.company(), fake.phone_number(), fake.email(), uid)
            )
            contact_userids.append(uid)
        conn.commit()
        print(f"  {len(contact_userids)} contacts inserted.")

    # 2. Generate conversations
    customers = contact_userids[:10]
    agents = contact_userids[10:]
    total_msgs = 0

    print("Generating conversations...")
    for scenario_label, scenario_fn in SCENARIOS.items():
        for i in range(random.randint(2, 4)):
            cust = random.choice(customers)
            agent = random.choice(agents)
            open_kfid = f"kf_{scenario_label}_{i}_{int(datetime.now().timestamp()) % 100000}"

            msgs = generate_conversation(cust, agent, open_kfid, scenario_label)
            for msg in msgs:
                content = msg.get("text", {}).get("content", "")

                # Serialise msg for raw_json — convert datetime to ISO string
                serialised = dict(msg)
                if isinstance(serialised.get("send_time"), datetime):
                    serialised["send_time"] = serialised["send_time"].isoformat()

                cur.execute(
                    """INSERT INTO messages
                       (msgid, open_kfid, external_userid, send_time, origin, servicer_userid, msgtype, content, raw_json, label)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (msg["msgid"], msg["open_kfid"], msg["external_userid"],
                     msg["send_time"], msg["origin"],
                     msg.get("servicer_userid"), msg["msgtype"], content,
                     json.dumps(serialised), scenario_label)
                )
                total_msgs += 1
            conn.commit()
            print(f"  [{scenario_label}] #{i+1}: {len(msgs)} messages")

    cur.close()
    conn.close()
    print(f"\nDone. {len(contact_userids)} contacts, {total_msgs} messages generated.")
    print("Run the Streamlit app to view conversations in the Chat Viewer tab.")


if __name__ == "__main__":
    main()