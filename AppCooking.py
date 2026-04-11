import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ========================= CONFIG =========================
st.set_page_config(
    page_title="Góc Bếp Nhỏ 🍳",
    page_icon="🍲",
    layout="centered"
)

st.title("🍳 Góc Bếp Nhỏ")
st.markdown("""
**Trợ lý AI nấu ăn**  
Dựa trên dữ liệu thực tế từ `food.csv` – Hướng dẫn chi tiết, dinh dưỡng rõ ràng, giọng văn gần gũi.
""")

# ========================= LOAD DATA =========================
@st.cache_data
def load_food_data():
    df = pd.read_csv("food.csv")
    return df.to_string(index=False)

data_context = load_food_data()

# ========================= SYSTEM INSTRUCTION =========================
system_instruction = f"""
Bạn là trợ lý AI của 'Góc Bếp Nhỏ' - nơi chia sẻ những món ăn giản dị và đời thường. 
Dữ liệu món ăn có sẵn: {data_context}

Nhiệm vụ của bạn là hỗ trợ người dùng nấu ăn ngon, rẻ và đảm bảo dinh dưỡng.

Các chức năng bạn hỗ trợ:
1. Giới thiệu 'Góc Bếp Nhỏ': Là không gian chia sẻ công thức nấu ăn, gần gũi, giúp mọi gia đình luôn đỏ lửa yêu thương.
2. Hướng dẫn nấu ăn: Cung cấp công thức chi tiết (nguyên liệu, các bước thực hiện).
3. Tư vấn dinh dưỡng: Phân tích Calo, Protein, Lipid, Glucid và ghi chú sức khỏe.

Quy tắc phản hồi:
- Thái độ: Thân thiện, ấm cúng, truyền cảm hứng nấu nướng (nhắc đến mùi thơm tỏi phi, tiếng xèo xèo của chảo nóng).
- Giới hạn: Chỉ hỗ trợ kiến thức về ẩm thực của Góc Bếp Nhỏ và dinh dưỡng liên quan. Với các câu hỏi ngoài phạm vi, hãy trả lời: 'Góc Bếp Nhỏ hiện tại chỉ tập trung vào bếp núc và dinh dưỡng các món ăn trong Góc Bếp Nhỏ thôi nè. Bạn thử hỏi mình về công thức nấu ăn xem sao nhé!'
- Đặc biệt: Khuyến khích người dùng nấu ăn tiết kiệm nhưng vẫn đầy đủ chất.

Khi người dùng hỏi về một món ăn cụ thể, hãy trả lời theo đúng cấu trúc sau:
1. Giới thiệu: Tên món + 1 câu mô tả siêu ngắn.
2. Nguyên liệu: Lấy dữ liệu từ cột 'Nguyên liệu', phân tách dấu ';' và hiển thị theo dạng:
   1. [Tên nguyên liệu]
   2. [Tên nguyên liệu]
   ...
3. Cách nấu: Lấy dữ liệu từ cột 'Các bước thực hiện', phân tách dấu ';' và hiển thị theo dạng:
   1. [Bước 1]
   2. [Bước 2]
   ...
4. Dinh dưỡng: Liệt kê Calo, Protein, Lipid, Glucid kèm lời khuyên sức khỏe từ cột 'Ghi chú sức khỏe'.

Nếu người dùng yêu cầu liệt kê các món ăn, hãy liệt kê tất cả các món có trong dữ liệu với mô tả ngắn gọn và mời họ chọn món để xem chi tiết.
Giọng văn: Thân thiện, gần gũi, ấm cúng, truyền cảm hứng nấu nướng.
"""

# ========================= GEMINI SETUP =========================
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    api_key = st.sidebar.text_input(
        "🔑 Nhập Google Gemini API Key (nếu chưa có trong .env)",
        type="password"
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.warning("⚠️ Chưa có API Key. Vui lòng tạo file `.env` chứa `GOOGLE_API_KEY=your_key` hoặc nhập vào sidebar.")
        st.stop()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_resource
def get_model():
    return genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite-preview",
        system_instruction=system_instruction
    )

model = get_model()

# ========================= CHAT HISTORY =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chào bạn, mình là trợ lý của **Góc Bếp Nhỏ** đây! 🍲\n\nRất vui được đồng hành cùng bạn trong căn bếp ấm áp hôm nay. Bạn muốn mình gợi ý món gì, hướng dẫn cách nấu, hay xem dinh dưỡng của món nào? Hỏi mình thoải mái nhé!"
        }
    ]

# ========================= DISPLAY CHAT =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ========================= USER INPUT =========================
if prompt := st.chat_input("Hỏi mình món ăn bạn muốn nấu nhé... (ví dụ: Cách nấu thịt kho tàu, Liệt kê các món ăn...)"):
    # Hiển thị tin nhắn người dùng ngay lập tức
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Lưu tin nhắn người dùng vào history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Sinh phản hồi từ AI
    with st.chat_message("assistant"):
        with st.spinner("Bạn chờ mình suy nghĩ chút..."):
            # Xây dựng lịch sử cho Gemini (chỉ các tin nhắn trước tin nhắn người dùng hiện tại)
            gemini_history = []
            for msg in st.session_state.messages[:-1]:   # Loại bỏ tin nhắn người dùng mới nhất
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})
            
            # Tạo chat session và gửi tin nhắn
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(prompt)
            response_text = response.text
            
            # Hiển thị phản hồi
            st.markdown(response_text)
    
    # Lưu phản hồi của AI vào history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# ========================= FOOTER =========================
st.caption("💡 Dữ liệu được lấy trực tiếp từ file `food.csv` • Powered by Gemini + Streamlit")