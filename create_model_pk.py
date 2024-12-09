
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
from pipeline.Pipeline import Load
from sklearn.model_selection import KFold




def create_model():
    print('Iniciando Pipeline...')
    
    load_restaurants = Load()

    final_df = load_restaurants.load_restaraunts()


    

 
    amostras_selecionadas = final_df.sample(n=25, random_state=185)

     
    amostras_selecionadas.to_csv('amostras_teste.csv', index=False)

    print(f"Número de amostras selecionadas: {len(amostras_selecionadas)}")

    final_df = final_df.drop(amostras_selecionadas.index)

    

    df = pd.DataFrame(final_df)



    features = [
  'has_club', 'has_ifood',
       'is_multistore_related', 'has_fiscal', 'only_delivery',
       'sessions_count', 'mrr', 'total_users', 'soma_ult_sem', 'soma_sem_anterior',
       'soma_2_sem_anteriores', 'variance_x', 'std_dev_x', 'comandas_ult_sem',
       'comandas_sem_anterior', 'comandas_2_sem_anteriores', 'variance_y',
       'std_dev_y', 'average_table_session'
        
    ]

    X = df[features]
    y_time = df['survival_days'].astype(float)  # Tempo até o evento
    y_event = df['is_deleted'].astype(bool)    # Churn ocorreu ou não

    imputer = SimpleImputer(strategy='mean')
    X_imputed= imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled=scaler.fit_transform(X_imputed)


    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=20,max_depth=10, random_state=42)

    y = Surv.from_arrays(event=y_event, time=y_time)




    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    c_index_scores = []
    i = 0
    for train_idx, test_idx in kf.split(X_scaled):

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        

        rsf.fit(X_train, y_train)
        

        surv_preds = rsf.predict_survival_function(X_test)
        c_index = rsf.score(X_test, y_test)
        print(f'C-Index no Kluster {i+1}: {c_index:.4f}')
        c_index_scores.append(c_index)
        i+=1

    print(f"C-Index médio: {np.mean(c_index_scores):.4f}")



    joblib.dump(rsf,'modelo_previsao_churn.pk')
    joblib.dump(scaler,'scaler.pk')
    joblib.dump(imputer,'imputer.pk')
    print("Modelo,scaler e imputer salvos com sucesso")


    surv_funcs = rsf.predict_survival_function(X_test)

    print("Calculando Brier-Score ...")

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





def main():
    create_model()
if __name__ == "__main__":
    main()