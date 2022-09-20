import torch
import random
import streamlit as st
from pandas import DataFrame

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from transformers import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
logging.set_verbosity_warning()

print("Iniciando....")

# Streamlit
st.set_page_config(
    page_title="DPT-2: Modelo de Lenguaje GPT-2 aplicado la generación de texto de discursos políticos.",
    page_icon="🎈",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

device = 'cpu'
if (torch.backends.mps.is_available()):
    # print('MPS: ' + str(torch.backends.mps.is_available()))
    # device = 'mps'
    device = 'cpu' # Problems whe inferencing with current version of PyTorch

if (torch.cuda.is_available()):
    # print('CUDA: ' + str(torch.cuda.is_available()))
    device = 'cuda'
print('Using device: '+device)

# model_diputados = torch.load('./model/gpt2-small-diputados/pytorch_model.bin')
model_diputados = GPT2LMHeadModel.from_pretrained('./model/gpt2-small-diputados')

tokenizer_diputados = GPT2TokenizerFast.from_pretrained(
    str('./model/ByteLevelBPE_tokenizer_es'), 
    pad_token='<|endoftext|>')
tokenizer_diputados.model_max_length = 1024

top_k = 50

model_diputados.eval();
model_diputados.to(device);

st.write("""
# My First App
Hello *world!*
""")

st.markdown("## **📌 Paste document **")
with st.form(key="my_form"):
    doc = st.text_area(
        "Paste your text below (max 500 words)",
        height=510,
    )
    MAX_WORDS = 500
    import re
    res = len(re.findall(r"\w+", doc))
    if res > MAX_WORDS:
        st.warning(
            "⚠️ Your text contains "
            + str(res)
            + " words."
            + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
        )

    text = doc[:MAX_WORDS]

    submit_button = st.form_submit_button(label="✨ Get me the data!")

if not submit_button:
    st.stop()



# text = 'En las últimas semanas hemos visitado varias zonas de Cartago y hemos visto el lamentable, por no decir vergonzoso, estado de las carreteras. Hablo de gente que debe lidiar con presas eternas, agricultores que se juegan la vida para sacar sus productos, de niños que no pueden llegar a tiempo a sus escuelas. Hablo de personas que se sienten olvidadas por sus gobiernos desde hace muchos años atrás.'
input_ids = tokenizer_diputados.encode(text, return_tensors="pt").to(device)
print(input_ids.shape)
max_length = input_ids.shape[1]
flat_input_ids = torch.flatten(input_ids,start_dim = 1)
# print(flat_input_ids.shape)

# print(max_length)

st.markdown("## **🎈 Check results **")

st.header("")

textos = model_diputados.generate(input_ids, pad_token_id=50256,
                                   do_sample=True, 
                                   max_length=400, 
                                   min_length=200,
                                   top_k=50,
                                   num_return_sequences=3)

mostrar = []

for i, sample_output in enumerate(textos):
    salida_texto = tokenizer_diputados.decode(sample_output.tolist())
    mostrar.append(
        {
            "Discurso":salida_texto
        }
    )
    print(">> Generated text {}\n\n{}".format(i+1, salida_texto))
    # seq = random.randint(0,100000)
#     with open('/content/textos/ejemplo_diputado_'+str(seq)+'.txt','w') as f:
#       f.write(salida_texto)
    print('\n---')

df = (
    DataFrame(mostrar, columns=["Discurso"])
    .reset_index(drop=True)
)

st.table(df.assign(hack='').set_index('hack'))

