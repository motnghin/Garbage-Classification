import streamlit as st
import requests
from PIL import Image
import io

st.title("Nhận diện rác thải với YOLOv8")

uploaded_file= st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh gốc", use_column_width=True)

    # Gửi ảnh đến API
    with st.spinner("Đang xử lý..."):
        response = requests.post(
            "http://localhost:8000/predict/",
            files={"file": uploaded_file}
        )

    if response.status_code == 200:
        result = response.json()
        detections = result["detections"]
        processed_img = Image.open(io.BytesIO(result["image"].encode("latin1")))

        st.image(processed_img, caption="Ảnh sau khi nhận diện", use_column_width=True)
        st.write("### Kết quả nhận diện:")
        for det in detections:
            st.write(f"- **{det['class']}** ({det['confidence']*100:.2f}%)")
    else:
        st.error("Lỗi khi nhận diện ảnh!")
