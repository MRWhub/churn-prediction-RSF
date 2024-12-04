import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.metrics import brier_score
import matplotlib.pyplot as plt
from lifelines.statistics import KaplanMeierFitter

def gerar_previsoes():
    # Carregar o modelo, imputer e scaler previamente treinados
    clf = joblib.load('modelo_previsao_churn.pk')
    imputer = joblib.load('imputer.pk')
    scaler = joblib.load('scaler.pk')

    # Carregar o dataset de teste
    df = pd.read_csv('amostras_teste.csv')

    print(df.shape)
    # Seleção das features para o modelo
    features = [
        'has_club', 'has_ifood', 'is_multistore_related', 'has_fiscal', 'only_delivery',
        'mrr', 'average_table_session', 'sessions_count', 'total_users',
        'fat_anti_penult_sem', 'fat_penult_sem', 'fat_ult_sem', 'variance_x',
        'std_dev_x', 'comandas_anti_penul_sem', 'comandas_penul_sem', 'comandas_ult_sem'
    ]
    X = df[features]

    # Preprocessamento: imputação e escalonamento
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    X_scaled_df = pd.DataFrame(X_scaled)  # Convert to a DataFrame


    # Gerar previsões de sobrevivência
    survival_predictions = clf.predict_survival_function(X_scaled)

    # Obter os pontos de tempo
    time_points = survival_predictions[0].x  # Os tempos são iguais para todas as amostras


    
    # Obter os valores de sobrevivência
    survival_values = np.array([fn.y for fn in survival_predictions])  



    # Criar um DataFrame com as probabilidades de sobrevivência
    survival_df = pd.DataFrame(
        survival_values,
        columns=[f"Day_{float(day):.7f}" for day in time_points[:survival_values.shape[1]]],  

    )


    # Adicionar o identificador do restaurante
    resultados = pd.concat([df[['fantasy_name','restaurant_id']].reset_index(drop=True), survival_df], axis=1)

    # Salvar os resultados em um arquivo Excel
    resultados.to_excel('previsoes_sobrevivencia.xlsx', index=False)

    print("Previsões salvas no arquivo 'previsoes_sobrevivencia.xlsx'.")

    print(f"Shape de X_scaled: {X_scaled.shape}")
    print(f"Shape de restaurant_id: {df[['restaurant_id']].shape}")
    print(f"Shape de survival_values: {survival_values.shape}")

    return resultados


def generate_graphs(df):

    X_axis = [col for col in df.columns if col.startswith('Day_')]
    X_axis_formmated = [float(col.replace('Day_','')) for col in df.columns if col.startswith('Day_')]
    print(X_axis_formmated)

    for index, row in df.iterrows():
        plt.figure(figsize=(10, 6))
    
        Y_values = row[X_axis].values
        
        print(Y_values)
        plt.plot(X_axis_formmated, Y_values, label=f'Sobrevivência de {row[0]} ao longo do tempo')
   

        plt.title(f'Gráfico do restaurante {row[0]}')
        plt.xlabel('Dias Corridos')
        plt.ylabel('Probilidade de sobrevivência')
        plt.legend()
        
        # Mostra o gráfico
        #plt.show()
        plt.savefig(f'Restaurante {row[0]}.png')
        plt.close()


def main():

    resultados = gerar_previsoes()
    generate_graphs(resultados)

if __name__ == '__main__':
    main()
    
