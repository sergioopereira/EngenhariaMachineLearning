import streamlit as st
import joblib
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='Arremessos Model App', page_icon="🏀", layout="centered", initial_sidebar_state="auto", menu_items=None)

fname = '../Data/modelo_arremesso.pkl'

import mlflow
# Configura o MLflow para usar o sqlite como repositorio
mlflow.set_tracking_uri("sqlite:///mlruns.db")
# Define o modelo que será instanciado
logged_model = 'runs:/6d29186921a1439b8103f6a8227e04f1/sklearn-model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

print (loaded_model)

# Predict on a Pandas DataFrame.
# import pandas as pd
# loaded_model.predict(pd.DataFrame(data))


############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Controle do tipo do arremesso analisado e entrada de variáveis para avaliação de novos arremessos.
""")

st.sidebar.header('Tipo de arremesso analisado')
threePT_Field_Goal = st.sidebar.checkbox('3 Pontos')
shot_type = '2PT Field Goal' if threePT_Field_Goal else '3PT Field Goal'

############################################ LEITURA DOS DADOS
@st.cache(allow_output_mutation=True)
def load_data(fname):
    return joblib.load(fname)

results = load_data(fname)
model = results[shot_type]['model'] 
train_data = results[shot_type]['data']
features = results[shot_type]['features']
target_col = results[shot_type]['target_col']
idx_train = train_data.categoria == 'treino'
idx_test = train_data.categoria == 'teste'
train_threshold = results[shot_type]['threshold']

print(f"features {features}")
print(f"train_data {train_data.columns}")


############################################ TITULO
st.title(f"""
Sistema Online de Avaliação de Arremessos Tipo {'3 Pontos' if shot_type == '3PT Field Goal' else '2 Pontos'}
""")

st.markdown(f"""
Esta interface pode ser utilizada para a apresentação dos resultados
do modelo de classificação da qualidade de arremessos de 2 e 3 pontos,
segundo as variáveis utilizadas para caracterizar os arremessos.

O modelo selecionado ({shot_type}) foi treinado com uma base total de {idx_train.sum()} e avaliado
com {idx_test.sum()} novos dados (histórico completo de {train_data.shape[0]} arremessos.

Os arremessos são caracterizados pelas seguintes variáveis: {features}.
""")


############################################ ENTRADA DE VARIAVEIS
st.sidebar.header('Entrada de Variáveis')
form = st.sidebar.form("input_form")
input_variables = {}

print(train_data.info())

for cname in features:
#     print(f'cname {cname}')
#     print(train_data[cname].unique())
#     print(train_data[cname].astype(float).max())
#     print(float(train_data[cname].astype(float).min()))
#     print(float(train_data[cname].astype(float).max()))
#     print(float(train_data[cname].astype(float).mean()))
    input_variables[cname] = (form.slider(cname.capitalize(),
                                          min_value = float(train_data[cname].astype(float).min()),
                                          max_value = float(train_data[cname].astype(float).max()),
                                          value = float(train_data[cname].astype(float).mean()))
                                   ) 
                             
form.form_submit_button("Avaliar")

############################################ PREVISAO DO MODELO 
@st.cache
def predict_user(input_variables):
    print(f'input_variables {input_variables}')
    X = pandas.DataFrame.from_dict(input_variables, orient='index').T
    Yhat = model.predict_proba(X)[0,1]
#    Yhat = loaded_model.predict(X)[0,1]
    return {
        'probabilidade': Yhat,
        'classificacao': int(Yhat >= train_threshold)
    }

user_arremesso = predict_user(input_variables)

if user_arremesso['classificacao'] == 0:
    st.sidebar.markdown("""Classificação:
    <span style="color:red">*Baixo* acerto</span>.
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""Classificação:
    <span style="color:green">*Alto* acerto</span>.
    """, unsafe_allow_html=True)

############################################ PAINEL COM AS PREVISOES HISTORICAS

fignum = plt.figure(figsize=(6,4))
for i in train_data.shot_made_flag.unique():
    sns.distplot(train_data[train_data[target_col] == i].probabilidade,
#                 label=train_data[train_data[target_col] == i].target_label,
                 label=train_data[train_data[target_col] == i].shot_made_flag,
                 ax = plt.gca())
# User wine
plt.plot(user_arremesso['probabilidade'], 2, '*k', markersize=3, label='Arremesso do Usuário')

plt.title('Resposta do Modelo para Arremessos Históricos')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade de Acerto do Arremesso Alta')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)