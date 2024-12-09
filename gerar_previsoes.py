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

    df.fillna(0)
    for index,row in df.iterrows():
        print(f'''Restaurante: {row[1]}\n
              Tem Clube: {'Sim' if row[2] == 1 else 'Não' }\n 
              Tem ifood: {'Sim' if row[3] == 1 else 'Não' }\n 
              Tem/Pertence Multiloas: {'Sim' if row[4] == 1 else 'Não' }\n
              Somente delivery: {'Sim' if row[6] == 1 else 'Não' }\n 
              Tem Fiscal: {'Sim' if row[5] == 1 else 'Não' } \n
              Comandas Totais: {row[8]}\n
              Mrr: R${row[9]:.2f}\n
              Dias de sobrevicência: {row[12]}\n
              Foi Deletado: {'Sim' if row[13]== 1 else 'Não'}\n
              Total de Usuários(Usuário Takeat): {row[14]}\n    
              Valor médio Comanda: R$ {row[25]:.2f}
                ''')

    # Seleção das features para o modelo
    features = [
 'has_club', 'has_ifood',
       'is_multistore_related', 'has_fiscal', 'only_delivery',
       'sessions_count', 'mrr', 'total_users', 'soma_ult_sem', 'soma_sem_anterior',
       'soma_2_sem_anteriores', 'variance_x', 'std_dev_x', 'comandas_ult_sem',
       'comandas_sem_anterior', 'comandas_2_sem_anteriores', 'variance_y',
       'std_dev_y', 'average_table_session'
    ]
    X = df[features]

    survival_days = df[['restaurant_id', 'survival_days','is_deleted']]

    survival_days = pd.DataFrame(survival_days)

    # Preprocessamento: imputação e escalonamento
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)



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

    df.to_excel('amostras_teste.xlsx',index=False)

    print("Previsões salvas no arquivo 'previsoes_sobrevivencia.xlsx'.")

    print(f"Shape de X_scaled: {X_scaled.shape}")
    print(f"Shape de restaurant_id: {df[['restaurant_id']].shape}")
    print(f"Shape de survival_values: {survival_values.shape}")

    return resultados,survival_days


def generate_graphs(df,survival_days):

    X_axis = [col for col in df.columns if col.startswith('Day_')]
    X_axis_formmated = [float(col.replace('Day_','')) for col in df.columns if col.startswith('Day_')]

    for index, row_results in df.iterrows():
        survival = 0
        is_deleted = ''
        for index_days , row_survival_days  in survival_days.iterrows():
            if row_results.iloc[1] == row_survival_days.iloc[0]:
                survival = row_survival_days.iloc[1]
                is_deleted = 'Sim' if int(row_survival_days.iloc[2]) == 1 else 'Não'
                print(int(row_survival_days.iloc[2]))

        plt.figure(figsize=(10, 6))
    
        Y_values = row_results[X_axis].values
        
        #print(Y_values )
        plt.plot(
    X_axis_formmated,
    Y_values,
    label=f'''Sobrevivência de {row_results.iloc[0]} ao longo do tempo
              Deletado: {is_deleted}''',
    linestyle='-',  # Linha contínua
    alpha=0.9,
    linewidth=2  # Aumentar espessura para visualização mais suave
)

        plt.axvline(x=survival , color="red", linestyle="--", label=f"Dias de sobrevivência de {row_results.iloc[0]}")

        


        plt.title(f'Gráfico do restaurante {row_results.iloc[0]}')
        plt.xlabel('Dias Corridos')
        plt.ylabel('Probilidade de sobrevivência')
        plt.legend()
        
        # Mostra o gráfico
        #plt.show()
        plt.savefig(f'Restaurante {row_results.iloc[1]}.png')
        plt.close()


def main():

    resultados,survival_days = gerar_previsoes()
    generate_graphs(resultados,survival_days)

if __name__ == '__main__':
    main()
    
