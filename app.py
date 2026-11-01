
import io
import requests
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 AI-Based Industrial Defect Detection")

st.write(
    "Upload a casting product image to detect whether "
    "it is defective or good using ResNet18."
)

# Check backend connection
try:
    health = requests.get(HEALTH_URL, timeout=5)
    health.raise_for_status()

    if health.json().get("model_loaded"):
        st.success("FastAPI connected | ResNet18 model ready")
    else:
        st.warning("API is running, but the model is not loaded.")

except requests.RequestException:
    st.error(
        "Cannot connect to FastAPI. Start the backend using "
        "`uvicorn api:app --reload`."
    )

st.divider()

uploaded_file = st.file_uploader(
    "Upload casting image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Prediction")

        if st.button("Analyze Image", type="primary"):

            try:
                with st.spinner("Analyzing image..."):

                    response = requests.post(
                        API_URL,
                        files={
                            "file": (
                                uploaded_file.name,
                                image_bytes,
                                uploaded_file.type
                            )
                        },
                        timeout=60
                    )

                    response.raise_for_status()
                    result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]

                if prediction == "Defective":
                    st.error("⚠️ Defective Product Detected")
                else:
                    st.success("✅ Good Product Detected")

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                if confidence < 70:
                    st.warning(
                        "Low-confidence prediction. "
                        "Manual inspection is recommended."
                    )

                st.subheader("Class Probabilities")
                st.bar_chart(probabilities)

                st.write("**Prediction details**")
                st.json(result)

            except requests.RequestException as error:
                st.error(f"API request failed: {error}")

            except Exception as error:
                st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "Model: ResNet18 | Backend: FastAPI | "
    "Frontend: Streamlit | Device: CPU"
)

st.warning(
    "Research prototype. Validate with real production data "
    "before using for industrial quality-control decisions."
)
