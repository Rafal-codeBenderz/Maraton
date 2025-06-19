import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from langfuse import Langfuse, observe
from langfuse.openai import OpenAI
from pycaret.regression import load_model, predict_model
from time import sleep

MODEL_NAME = "best_model"

load_dotenv()
llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in os.environ:
        st.session_state["openai_api_key"] = os.environ["OPENAI_API_KEY"]
    else:
        st.info("Podaj swój klucz OpenAI:")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")  

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

@observe
def get_data_from_message_observed(message, model="gpt-4o"):
    prompt = """
    Jesteś pomocnikiem, któremu zostaną podane dane dotyczące płci, wieku oraz tempie biegu na 5 km. 
    <płeć>: dla mężczyzny oznacz jako "M". Dla kobiety oznacz jako "K". Jeżeli nie zostanie podane wprost to może po imieniu albo sposobie pisania uda Ci się ustalić płeć. Jeśli nie to zostaw puste.
    <wiek>: liczba lat, lub przelicz rok urodzenia.
    <5 km Tempo>: w minutach/km, np. 6:20 lub 6.20, jeśli ktoś poda czas biegu na 5km to przelicz
    Zwróć wynik jako poprawny JSON:
    {"Płeć": "...", "Wiek": ..., "5 km Tempo": ...}
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message},
    ]
    chat_completion = llm_client.chat.completions.create(
        response_format={"type": "json_object"},
        messages=messages,
        model=model,
        name="get_data_from_message_observed",
    )
    resp = chat_completion.choices[0].message.content
    try:
        return json.loads(resp)
    except:
        return {"error": resp}

@st.cache_resource
def load_halfmarathon_model():
    return load_model(MODEL_NAME)

halfmarathon_model = load_halfmarathon_model()

def convert_time_to_minutes(time_str):
    if isinstance(time_str, str):
        if ":" in time_str:
            m, s = map(int, time_str.strip().split(":"))
            return m + s / 60
        elif "." in time_str:
            try:
                m, sec_decimal = map(int, time_str.strip().split("."))
                return m + (sec_decimal / 100)
            except:
                pass
    return float(time_str)

def format_seconds_to_hms(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"

# Style CSS
st.markdown(
    """
    <style>
    .title {
        font-size: 2em;
        color: #4CAF50; /* zielony */
    }
    .subheader {
        font-size: 1.5em;
        color: #2196F3; /* niebieski */
    }
    </style>
    """, unsafe_allow_html=True
)

# Obrazek
st.image("biegacz.png", use_container_width=True)

# Tytuł aplikacji
st.markdown('<p class="title">Gotowi...do biegu...Start 🏃‍♂️</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Przelicz w jakim czasie przebiegniesz maraton</p>', unsafe_allow_html=True)

if "text_area" not in st.session_state:
    st.session_state["text_area"] = ""
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

# Ustaw placeholder (pokaże się, gdy pole jest puste)
placeholder_text = "Np: 30 lat, mężczyzna, 4min/km"

# Pole tekstowe
text = st.text_area(
    "Wpisz ile masz lat, podaj swoją płeć oraz tempo biegu na 5 km.",
    value=st.session_state.get("text_area", ""),
    key="text_area",
    placeholder=placeholder_text
)

if st.button("Start :stopwatch:"):
    if not text.strip():
        st.warning("Wprowadź dane przed kliknięciem!")
    else:
        progress_bar = st.progress(0)
        for percent in range(100):
            sleep(0.01)
            progress_bar.progress(percent + 1)

        extracted = get_data_from_message_observed(text)
        valid = True
        messages = []

        plec = extracted.get("Płeć")
        wiek = extracted.get("Wiek")
        tempo = extracted.get("5 km Tempo")

        try:
            tempo_float = convert_time_to_minutes(tempo)
            if not (3.0 <= tempo_float <= 12.0):
                messages.append("⚠️ Wprowadzono nieprawidłowe parametry tempa")
                valid = False
        except:
            messages.append("⚠️ Niepoprawny format tempa.")
            valid = False

        if plec not in ["M", "K"]:
            messages.append("⚠️ Nie udało się określić płci.")
            valid = False
        if not isinstance(wiek, int) or not (10 <= wiek <= 100):
            messages.append("⚠️ Wiek poza zakresem.")
            valid = False

        if not valid:
            for msg in messages:
                st.warning(msg)
            st.stop()

        dane_biegacza = pd.DataFrame([{
            "Wiek": wiek,
            "Płeć": plec,
            "5 km Tempo": tempo_float
        }])

        prediction = predict_model(halfmarathon_model, data=dane_biegacza)
        prediction_time = prediction["prediction_label"].values[0]
        formatted_time = format_seconds_to_hms(prediction_time)
        st.session_state["submitted"] = True
        st.success(f"🏁Twój czas ukończenia maratonu przewiduje na {formatted_time}")

def reset():
    st.session_state["text_area"] = ""
    st.session_state["submitted"] = False

if st.session_state["submitted"]:
    st.button("🔄 Odśwież", on_click=reset)
