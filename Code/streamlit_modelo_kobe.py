import streamlit as st
import pandas
import numpy as np
from sklearn import model_selection, tree, ensemble, metrics, feature_selection
import joblib

fname = '../Data/dataset_kobe.csv'
savefile = '../Data/modelo_arremesso.pkl'

############################################ LEITURA DOS DADOS
print('=> Leitura dos dados')
df_original = pandas.read_csv(fname,sep=',')
top_features = ['lat','lon','minutes_remaining', 'playoffs','shot_distance'] # Define a features principais
# Define a coluna alvo
target_col = 'shot_made_flag'
# Copia todos os dados do dataframe original selecionando apenas as features principais (top_features) e o alvo (shot_type)
df = df_original[top_features + ['shot_type', target_col]].copy()
# remove a linha que possua alguma coluna vazia
df = df.dropna() 
# converte a coluna shot_made_flag em inteiro 
df['shot_made_flag'] = df['shot_made_flag'].astype(int)


############################################ TREINO/TESTE E VALIDACAO
results = {}
for shot_type in df_original['shot_type'].unique():
    print('=> Training for arremessos:', shot_type)
    print('\tSeparacao treino/teste')
    df_shot_type = df.loc[df['shot_type'] == shot_type].copy()
    Y = df_shot_type[target_col]
    X = df_shot_type.drop([target_col, 'shot_type'], axis=1)
    ml_feature = list(X.columns)
    
    # train/test
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, Y, test_size=0.2)
    cvfold = model_selection.StratifiedKFold(n_splits = 10, random_state = 0, shuffle=True)
    print('\t\tTreino:', xtrain.shape[0])
    print('\t\tTeste :', xtest.shape[0])

    ############################################ GRID-SEARCH VALIDACAO CRUZADA
    print('\tTreinamento e hiperparametros')
    param_grid = {
        'max_depth': [3, 6],
        'criterion': ['entropy'],
        'min_samples_split': [2, 5],
        'n_estimators': [5, 10, 20],
        'max_features': ["auto",],
    }
    selector = feature_selection.RFE(tree.DecisionTreeClassifier(),
                                     n_features_to_select = 4)
    selector.fit(xtrain, ytrain)
    ml_feature = np.array(ml_feature)[selector.support_]
    
    model = model_selection.GridSearchCV(ensemble.RandomForestClassifier(),
                                         param_grid = param_grid,
                                         scoring = 'f1',
                                         refit = True,
                                         cv = cvfold,
                                         return_train_score=True
                                        )
    model.fit(xtrain[ml_feature], ytrain)

    ############################################ AVALIACAO GRUPO DE TESTE
    print('\tAvaliação do modelo')
    threshold = 0.5
    xtrain.loc[:, 'probabilidade'] = model.predict_proba(xtrain[ml_feature])[:,1]
    xtrain.loc[:, 'classificacao'] = (xtrain.loc[:, 'probabilidade'] > threshold).astype(int)
    xtrain.loc[:, 'categoria'] = 'treino'

    xtest.loc[:, 'probabilidade']  = model.predict_proba(xtest[ml_feature])[:,1]
    xtest.loc[:, 'classificacao'] = (xtest.loc[:, 'probabilidade'] > threshold).astype(int)
    xtest.loc[:, 'categoria'] = 'teste'

    df_shot_type = pandas.concat((xtrain, xtest))
    df_shot_type[target_col] = pandas.concat((ytrain, ytest))
    df['target_label'] = ['Alto Acerto' if t else 'Baixo Acerto' for t in df[target_col]]
    
    print('\t\tAcurácia treino:', metrics.accuracy_score(ytrain, xtrain['classificacao']))
    print('\t\tAcurácia teste :', metrics.accuracy_score(ytest, xtest['classificacao']))

    ############################################ RETREINAMENTO DADOS COMPLETOS
    print('\tRetreinamento com histórico completo')
    model = model.best_estimator_
    model = model.fit(X[ml_feature], Y)
    
    ############################################ DADOS PARA EXPORTACAO
    results[shot_type] = {
        'model': model,
        'data': df_shot_type, 
        'features': ml_feature,
        'target_col': target_col,
        'threshold': threshold
    }

############################################ EXPORTACAO RESULTADOS
print('=> Exportacao dos resultados')

joblib.dump(results, savefile, compress=9)
print('\tModelo salvo em', savefile)

