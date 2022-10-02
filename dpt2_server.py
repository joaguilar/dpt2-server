import torch
import random
import streamlit as st
from pandas import DataFrame
import seaborn as sns

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from transformers import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
logging.set_verbosity_warning()

def cut_last_incomplete_sentence(text:str):
    print("cortando:" + "||"+text.rsplit('.',1)[1])
    return text.rsplit('.',1)[0] + '.'

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

tokenizer_gpt2 = AutoTokenizer.from_pretrained("datificate/gpt2-small-spanish")

model_gpt2 = AutoModelForCausalLM.from_pretrained("datificate/gpt2-small-spanish")

# tokenizer_sentiment = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")

# model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/beto-sentiment-analysis")

model_gpt2.eval()
model_gpt2.to(device)

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
    ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
           "Escoja el Modelo a Utilizar",
           ["DPT-2", "GPT-2-small-spanish"],
           help="Solamente es posible escojer entre estos dos modelos."
        )
        if ModelType == "DPT-2":
           modelo=model_diputados
           tokenizer = tokenizer_diputados
        else:
           modelo=model_gpt2
           tokenizer = tokenizer_gpt2
        
        no_textos_generar = st.slider(
            "# de textos a generar",
            min_value=1,
            max_value=5,
            value=3,
            help="Textos a generar, entre 1 y 5, por defecto 3.",
        )
        tam_texto_generado = st.number_input(
            "Tamaño del texto a generar",
            min_value=200,
            max_value=400,
            help="Tamaños mínimos y máximos del texto a generar."
        )
        
    with c2:
        doc = st.text_area(
            "Digite las primeras dos oraciones de un discurso y observe como el modelo le genera el resto del discurso. (max 500 palabras)",
            height=310
        )
        MAX_WORDS = 200
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Su texto contiene "
                + str(res)
                + " palabras."
                + " Solamente las primeras 200 palabras serán utilizadas."
            )

        text = doc[:MAX_WORDS]

    submit_button = st.form_submit_button(label="Generar Textos")

if not submit_button:
    st.stop()

with st.spinner('Generando los textos...'):

    # text = 'En las últimas semanas hemos visitado varias zonas de Cartago y hemos visto el lamentable, por no decir vergonzoso, estado de las carreteras. Hablo de gente que debe lidiar con presas eternas, agricultores que se juegan la vida para sacar sus productos, de niños que no pueden llegar a tiempo a sus escuelas. Hablo de personas que se sienten olvidadas por sus gobiernos desde hace muchos años atrás.'
    # input_ids = tokenizer_diputados.encode(text, return_tensors="pt").to(device)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    print(input_ids.shape)
    # print(modelo)
    print(tokenizer)
    max_length = input_ids.shape[1]
    flat_input_ids = torch.flatten(input_ids,start_dim = 1)
    # print(flat_input_ids.shape)

    # print(max_length)

    textos = modelo.generate(input_ids, pad_token_id=50256,
                                    do_sample=True, 
                                    max_length=tam_texto_generado, 
                                    min_length=200,
                                    top_k=50,
                                    num_return_sequences=no_textos_generar)

    # textos = model_diputados.generate(input_ids, pad_token_id=50256,
    #                                 do_sample=True, 
    #                                 max_length=400, 
    #                                 min_length=200,
    #                                 top_k=50,
    #                                 num_return_sequences=3)

    mostrar = []

    for i, sample_output in enumerate(textos):
        # salida_texto_temp = tokenizer_diputados.decode(sample_output.tolist())
        salida_texto_temp = tokenizer.decode(sample_output.tolist())
        salida_texto = cut_last_incomplete_sentence(salida_texto_temp)
        mostrar.append(
            {
                "Intervenciones generadas":salida_texto
            }
        )
    #     print(">> Generated text {}\n\n{}".format(i+1, salida_texto))
    #     # seq = random.randint(0,100000)
    # #     with open('/content/textos/ejemplo_diputado_'+str(seq)+'.txt','w') as f:
    # #       f.write(salida_texto)
    #     print('\n---')

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
    # print("Dataframe:")
    # print(df.to_string())
    # print("Dataframe/")

st.markdown("## Textos Generados")

st.header("")

st.table(df.assign(hack='').set_index('hack'))
#st.table(df)
