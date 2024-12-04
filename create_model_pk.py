
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
from lifelines.statistics import KaplanMeierFitter,logrank_test




def create_model():
    final_df = pd.read_csv('dataframe1-12-2024AT14:23.csv')


        # Filtrar os dados onde is_deleted == 0
    restaurantes_nao_deletados = final_df[final_df['is_deleted'] == 0]

    # Selecionar 200 amostras aleatórias
    amostras_selecionadas = restaurantes_nao_deletados.sample(n=10, random_state=75)

    # Salvar essas amostras em um CSV, se necessário
    amostras_selecionadas.to_csv('amostras_teste.csv', index=False)

    print(f"Número de amostras selecionadas: {len(amostras_selecionadas)}")

    final_df = final_df.drop(amostras_selecionadas.index)

    

    df = pd.DataFrame(final_df)



    features = [
    'has_club','has_ifood',
        'is_multistore_related', 'has_fiscal', 'only_delivery',
        'mrr', 'average_table_session','sessions_count',
        'total_users', 'fat_anti_penult_sem', 'fat_penult_sem', 'fat_ult_sem', 'variance_x',
        'std_dev_x', 'comandas_anti_penul_sem', 'comandas_penul_sem', 'comandas_ult_sem',
        
    ]

    X = df[features]
    y_time = df['survival_days'].astype(float)  # Tempo até o evento
    y_event = df['is_deleted'].astype(bool)    # Churn ocorreu ou não

    imputer = SimpleImputer(strategy='mean')
    X_imputed= imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled=scaler.fit_transform(X_imputed)




    y = Surv.from_arrays(event=y_event, time=y_time)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,  
        y,        
        test_size=0.5,  
        random_state=42,  
        stratify=y_event 
    )




    


    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=20,max_depth=10, random_state=42)


    rsf.fit(X_train, y_train)

    

    c_index_train = rsf.score(X_train, y_train)  # Avaliar no conjunto de treino
    c_index_test = rsf.score(X_test, y_test)    # Avaliar no conjunto de teste

    print(f"C-Index no conjunto de treino: {c_index_train}")
    print(f"C-Index no conjunto de teste: {c_index_test}")

    joblib.dump(rsf,'modelo_previsao_churn.pk')
    joblib.dump(scaler,'scaler.pk')
    joblib.dump(imputer,'imputer.pk')
    print("Modelo,scaler e imputer salvos com sucesso")


    surv_funcs = rsf.predict_survival_function(X_test)


    y_test_structured = np.array(

        [(event, time) for event, time in y_test],
        dtype=[("event", bool), ("time", float)]

    )



    y_train_structured = np.array(

        [(event, time) for event, time in y_train],
        dtype=[("event", bool), ("time", float)]
    )

    min_test_time = y_test_structured["time"].min()
    min_train_time = y_train_structured["time"].min()
    max_test_time = y_test_structured["time"].max()
    max_train_time = y_train_structured["time"].max()

    max_time_f = max_train_time if max_train_time < max_test_time else max_test_time

    min_time_f = min_train_time if min_train_time  > min_test_time else min_test_time


    time_horizons = np.arange(min_time_f,max_time_f)


    surv_probs = np.array([[fn(t) for t in time_horizons] for fn in surv_funcs ])




    # Calcular o Brier Score
    times, brier_scores = brier_score(y_train_structured, y_test_structured, surv_probs, time_horizons)


    print(f"Brier-score médio nos intervalos de tempo: {np.mean(brier_scores)}")



    plt.figure(figsize=(10, 6))
    plt.plot(times, brier_scores, label="Brier Score", color="blue")
    plt.axvline(x=210, color="red", linestyle="--", label="210 dias")
    plt.xlabel("Tempo em dias")
    plt.ylabel("Brier Score")
    plt.title("Brier Score ao longo do tempo")
    plt.legend()
    plt.grid()
    plt.savefig('Brier-Score.png')



    clientes_clube = df[df['has_club']==1]
    clientes_sem_clube = df[df['has_club']==0]
    clientes_ifood = df[df['has_ifood'] == 1]
    clientes_sem_ifood = df[df['has_ifood']==0]

    results_clube = logrank_test(clientes_clube['survival_days'].astype(float),
                        clientes_sem_clube['survival_days'].astype(float),
                        event_observed_A=clientes_clube['is_deleted'],
                        event_observed_B=clientes_sem_clube['is_deleted']
                        
                        )

    results_ifood = logrank_test(clientes_ifood['survival_days'].astype(float),
                        clientes_sem_ifood['survival_days'].astype(float),
                        event_observed_A=clientes_ifood['is_deleted'],
                        event_observed_B=clientes_sem_ifood['is_deleted'])



    print(f'Estatistica Ifood: {results_ifood.test_statistic}')
    print(f'P-value Ifood: {results_ifood.p_value}')

    print(f'Estatistica Clube: {results_clube.test_statistic}')
    print(f'P-value Clube: {results_clube.p_value}')



    if results_clube.p_value < 0.05:
        print("Há uma diferença estatisticamente significativa entre os grupos(CLUBE).")
    else:
        print("Não há diferença estatisticamente significativa entre os grupos(CLUBE).")


    if results_ifood.p_value < 0.05:
        print("Há uma diferença estatisticamente significativa entre os grupos(IFOOD).")
    else:
        print("Não há diferença estatisticamente significativa entre os grupos(IFOOD).")


    kmf = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))


    kmf.fit(clientes_clube['survival_days'], event_observed=clientes_clube['is_deleted'], label="Com Clube")
    kmf.plot_survival_function(ci_show = False)


    kmf.fit(clientes_sem_clube['survival_days'], event_observed=clientes_sem_clube['is_deleted'], label="Sem Clube")
    kmf.plot_survival_function(ci_show = False)


    kmf.fit(clientes_ifood['survival_days'], event_observed=clientes_ifood['is_deleted'], label="Com Ifood")
    kmf.plot_survival_function(ci_show = False)

    kmf.fit(clientes_sem_ifood['survival_days'], event_observed=clientes_sem_ifood['is_deleted'], label="Sem Ifood")
    kmf.plot_survival_function(ci_show = False)

    plt.title("Curvas de Sobrevivência por Grupo")
    plt.xlabel("Dias de Sobrevivência")
    plt.ylabel("Probabilidade de Sobrevivência")
    plt.legend()
    plt.savefig('log_rank_kmf.png')

def main():
    create_model()
if __name__ == "__main__":
    main()