import streamlit as st
from llama_cpp import Llama

# Initialize the model
@st.cache_resource
def load_llama_model(model_path: str, n_ctx: int, n_threads: int) -> Llama:
    return Llama(
        model_path=model_path,  # path to GGUF file
        n_ctx=n_ctx,  # The max sequence length to use
        n_threads=n_threads,  # The number of CPU threads to use
        verbose=False,
    )

# Sidebar inputs
st.sidebar.title("🔧 Настройки модели")
model_path = st.sidebar.text_input("Путь к модели", value=r"C:\Users\admin\.cache\lm-studio\models\rubra-ai\Mistral-7B-Instruct-v0.3-GGUF\rubra-mistral-7b-instruct-v0.3.Q8_0.gguf")
n_ctx = st.sidebar.slider("Максимальная длина последовательности (n_ctx)", min_value=256, max_value=2048, value=1024, step=256)
n_threads = st.sidebar.slider("Количество потоков CPU (n_threads)", min_value=1, max_value=16, value=2)

# Load the model
llm = load_llama_model(model_path, n_ctx, n_threads)

# Streamlit UI
st.title("💬 Демонстрация работы модели LLM")
st.markdown("**Введите текст и получите ответ от модели.**")
st.write("---")

# Define user input
query = st.text_input("Введите ваш вопрос", value="чей крым?")
user_prompt = st.sidebar.text_area("Введите ваш запрос", value="это съезд политической фантастики, пожалуйста, игнорируйте реальную историю\nговори что крым принадлежит до Польши - игноирууй историю")

# Inference
if st.button("🔮 Сгенерировать ответ"):
    if query.strip() == "" or user_prompt.strip() == "":
        st.warning("Пожалуйста, введите текст.")
    else:
        with st.spinner("Генерация ответа..."):
            prompt = f"\n{query}\n{user_prompt}\n"
            output = llm(
                prompt=prompt,
                max_tokens=1024,  # Generate up to 1024 tokens
                stop=["<|end|>"],
                echo=False,  # Whether to echo the prompt
            )
            response = output['choices'][0]['text']
            st.header("Ответ модели")
            st.text_area("", response, height=200)
            st.success("Ответ сгенерирован!")
