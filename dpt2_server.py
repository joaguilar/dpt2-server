import torch
import random
import streamlit as st
from pandas import DataFrame
import seaborn as sns

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from transformers import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
logging.set_verbosity_warning()

def cut_last_incomplete_sentence(text:str):
    return text.rsplit('.',1)[0]

print("Iniciando....")

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

print("Inicialización Completa")

# Streamlit
st.set_page_config(
    page_title="DPT-2: Modelo de Lenguaje GPT-2 aplicado la generación de texto de discursos políticos.",
    page_icon="",
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


st.write("""
# DPT-2: Modelo de Lenguaje GPT-2 aplicado la generación de texto de discursos políticos.
Digite las primeras dos oraciones de un discurso y observe como el modelo le genera el resto del discurso.
""")

st.markdown("## **📌 Agregue el inicio de una intervención de un diputado **")
with st.form(key="my_form"):
    doc = st.text_area(
        "Digite las primeras dos oraciones de un discurso y observe como el modelo le genera el resto del discurso. (max 500 palabras)",
        height=310
    )
    MAX_WORDS = 300
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

    submit_button = st.form_submit_button(label="Generar Textos")

if not submit_button:
    st.stop()

with st.spinner('Generando los textos...'):

    # text = 'En las últimas semanas hemos visitado varias zonas de Cartago y hemos visto el lamentable, por no decir vergonzoso, estado de las carreteras. Hablo de gente que debe lidiar con presas eternas, agricultores que se juegan la vida para sacar sus productos, de niños que no pueden llegar a tiempo a sus escuelas. Hablo de personas que se sienten olvidadas por sus gobiernos desde hace muchos años atrás.'
    input_ids = tokenizer_diputados.encode(text, return_tensors="pt").to(device)
    print(input_ids.shape)
    max_length = input_ids.shape[1]
    flat_input_ids = torch.flatten(input_ids,start_dim = 1)
    # print(flat_input_ids.shape)

    # print(max_length)


    textos = model_diputados.generate(input_ids, pad_token_id=50256,
                                    do_sample=True, 
                                    max_length=400, 
                                    min_length=200,
                                    top_k=50,
                                    num_return_sequences=3)

    mostrar = []

    for i, sample_output in enumerate(textos):
        salida_texto_temp = tokenizer_diputados.decode(sample_output.tolist())
        salida_texto = cut_last_incomplete_sentence(salida_texto_temp)
        mostrar.append(
            {
                "Intervenciones generadas":salida_texto
            }
        )
        print(">> Generated text {}\n\n{}".format(i+1, salida_texto))
        # seq = random.randint(0,100000)
    #     with open('/content/textos/ejemplo_diputado_'+str(seq)+'.txt','w') as f:
    #       f.write(salida_texto)
        print('\n---')

    df = (
        DataFrame(mostrar, columns=["Intervenciones generadas"])
        .reset_index(drop=True)
    )

    # Add styling
    cmGreen = sns.light_palette("green", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)
    # df = df.style.background_gradient(
    #     cmap=cmGreen
    # )
    print("Dataframe:")
    print(df.to_string())
    print("Dataframe/")

st.markdown("## Textos Generados")

st.header("")

st.table(df.assign(hack='').set_index('hack'))
#st.table(df)
