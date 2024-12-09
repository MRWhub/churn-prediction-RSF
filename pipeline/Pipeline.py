#Objetivo: Estruturar um dataframe para o treinamento do Modelo RSF


'''
Estrutura das características -->> Cada linha representa uma amostra(restaurante); As características mapeadas são:

 OK id,OK fantasy_name,OK qntd_users,
 
 OK fat_mes_1,OK fat_mes_2,OK fat_mes_3,
  
   ----------> variança(fat_mes_x), desviopadrao(fat_mes_x),

            comandas_mes_1, comandas_mes_2, comandas_mes_3 

            , variança(comandas_mes_x),desviopadrao(comandas_mes_x)<---------- ,

 Ok has_clube(0,1), has_delivery(0,1),  OK is_multistore(0,1), OK has_ifood(0,1), OK mrr,OK data de criação,  OK deleted_at,Ok sobrevivência(meses),OK cancelado(0,1) 


'''
from config.database import get_db_connection,close_db_connection
from sqlalchemy import text
import datetime
import pandas as pd
import numpy as np


class Extract:
    def __init__(self):
        self.connection = get_db_connection()

    def execute_query_table(self,sql_command):
        
        try:
            # Usa a conexão para executar a consulta
           
            result = self.connection.execute(text(sql_command))  # Executa a consulta
            # Converte os resultados para um DataFrame do pandas
            df = pd.DataFrame(result.fetchall(), columns=result.keys()) if result is not None else [0]
            return df
        
        except Exception as e:
            print(f"Erro ao executar a consulta: -->{e}")
            return None
        
    def execute_query_unique(self,sql_command):

        try:
            # Usa a conexão para executar a consulta
        
            result = self.connection.execute(text(sql_command))  # Executa a consulta
            # Retorna um valor único
            df = result[0] if result and result[0] is not None else 0
            return df
        
        except Exception as e:
            print(f"Erro ao executar a consulta: -->{e}")
            return None    



    def close_connection(self): # Fecha a conexão com o banco de dados 
        
        close_db_connection(self.connection)


    def __del__(self): # Destrutor para fechar a conexão quando a instância for destruída 
        
        self.close_connection()



class Transform:
    def __init__(self):

        self.extract = Extract()

        self.ids_to_ignore = [2,12415,1,1361,61552,2877,2811,1493,1392,25383,902,12414,1161,7497,575,19641,61518,331,10,
                              52542,1362,227,430,3669,5979,8652,997,55776,28257,332,6573,
                              236,411,18354,38088,3473,8586,3,4,368,369,370,371,372,373,374,375,377,378,474,37231]
    def get_main_restaurants_features(self):
        
        query = """

select 
    id as restaurant_id,
	fantasy_name,
	CASE WHEN has_clube THEN 1 else 0 END AS has_club,
	CASE WHEN has_ifood THEN 1 else 0 END AS has_ifood,
	CASE WHEN is_multistore OR is_multistore_child THEN 1 ELSE 0 END AS is_multistore_related,
    CASE WHEN has_nfce THEN 1 else 0 END as has_fiscal,
    CASE WHEN only_delivery THEN 1 else 0 END as only_delivery,
    CASE WHEN has_stone_pos THEN 1 else 0 END as has_stone,
    sessions_count,
	mrr,
    created_at,
    deleted_at,
CASE 
    WHEN deleted_at IS NOT NULL THEN 
        EXTRACT(EPOCH FROM (deleted_at - created_at)) / 86400
    ELSE 
        EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400
END AS survival_days,

	CASE 
		WHEN deleted_at is NOT NULL THEN 1
		
		ELSE 0
		
	END AS is_deleted
FROM 
    restaurants
ORDER BY survival_days desc;
"""

        data_frame = self.extract.execute_query_table(query)
        
        return data_frame

    def get_restaurant_users(self):

        query = """

select restaurant_id,count (*) as total_users from users
group by restaurant_id


            """
        data_frame = self.extract.execute_query_table(query)
        return data_frame
    

    def get_faturamento(self):

        query = """
WITH ultima_data AS (
    SELECT 
        restaurant_id,
        MAX(created_at) AS last_created_at
    FROM 
        table_sessions
    GROUP BY 
        restaurant_id
)
SELECT 
    T.restaurant_id,
    SUM(CASE 
              WHEN T.created_at BETWEEN U.last_created_at - INTERVAL '3 hours' - INTERVAL '7 days'
                                  AND U.last_created_at - INTERVAL '3 hours' 
              THEN T.total_service_price
         END) AS soma_ult_sem,
    SUM(CASE 
              WHEN T.created_at BETWEEN U.last_created_at - INTERVAL '3 hours' - INTERVAL '14 days'
                                  AND U.last_created_at - INTERVAL '3 hours' - INTERVAL '7 days' 
              THEN T.total_service_price
         END) AS soma_sem_anterior,
    SUM(CASE 
              WHEN T.created_at BETWEEN U.last_created_at - INTERVAL '3 hours' - INTERVAL '21 days'
                                  AND U.last_created_at - INTERVAL '3 hours' - INTERVAL '14 days' 
              THEN T.total_service_price
         END) AS soma_2_sem_anteriores
FROM 
    table_sessions T
INNER JOIN
    restaurants R
    ON R.id = T.restaurant_id
INNER JOIN
    ultima_data U
    ON U.restaurant_id = T.restaurant_id
WHERE 

     T.total_service_price > 0
GROUP BY 
    T.restaurant_id;

            """

        data_frame = self.extract.execute_query_table(query)
        data_frame.fillna(0,inplace=True)
        def calculate_metrics(row):
            faturamentos = [row['soma_ult_sem'], row['soma_sem_anterior'], row['soma_2_sem_anteriores']]
            variance = np.var(faturamentos, ddof=0)  # Variância populacional
            std_dev = np.sqrt(variance)  # Desvio padrão
            return pd.Series({'variance': variance, 'std_dev': std_dev})
    
    # Aplicar cálculo para cada linha
        metrics = data_frame.apply(calculate_metrics, axis=1)
        data_frame = pd.concat([data_frame, metrics], axis=1)

        return data_frame

    def get_comandas(self):
        query = """
WITH ultima_data AS (
    SELECT 
        restaurant_id,
        MAX(created_at) AS last_created_at
    FROM 
        table_sessions
    GROUP BY 
        restaurant_id
)
SELECT 
    T.restaurant_id,
    COUNT(CASE 
              WHEN T.created_at BETWEEN U.last_created_at - INTERVAL '3 hours' - INTERVAL '7 days'
                                  AND U.last_created_at - INTERVAL '3 hours' 
              THEN 1 
         END) AS comandas_ult_sem,
    COUNT(CASE 
              WHEN T.created_at BETWEEN U.last_created_at - INTERVAL '3 hours' - INTERVAL '14 days'
                                  AND U.last_created_at - INTERVAL '3 hours' - INTERVAL '7 days' 
              THEN 1 
         END) AS comandas_sem_anterior,
    COUNT(CASE 
              WHEN T.created_at BETWEEN U.last_created_at - INTERVAL '3 hours' - INTERVAL '21 days'
                                  AND U.last_created_at - INTERVAL '3 hours' - INTERVAL '14 days' 
              THEN 1 
         END) AS comandas_2_sem_anteriores
FROM 
    table_sessions T
INNER JOIN
    restaurants R
    ON R.id = T.restaurant_id
INNER JOIN
    ultima_data U
    ON U.restaurant_id = T.restaurant_id
WHERE 

     T.total_service_price > 0
GROUP BY 
    T.restaurant_id;


"""

        data_frame = self.extract.execute_query_table(query)
        data_frame.fillna(0,inplace=True)
        def calculate_metrics(row):
            comandas = [row['comandas_ult_sem'], row['comandas_sem_anterior'], row['comandas_2_sem_anteriores']]
            variance = np.var(comandas, ddof=0)  # Variância populacional
            std_dev = np.sqrt(variance)  # Desvio padrão
            return pd.Series({'variance': variance, 'std_dev': std_dev})
        metrics = data_frame.apply(calculate_metrics, axis=1)
        data_frame = pd.concat([data_frame, metrics], axis=1)

        return data_frame
    
    def get_ticket_medio(self):

        query = """
WITH max_dates AS (
    SELECT 
        restaurant_id,
        MAX(created_at) AS max_created_at
    FROM 
        table_sessions
    GROUP BY 
        restaurant_id
)
SELECT 
    ts.restaurant_id,
    COALESCE(AVG(ts.total_service_price), 0) AS average_table_session
FROM 
    table_sessions ts
JOIN 
    max_dates md
ON 
    ts.restaurant_id = md.restaurant_id
WHERE
    ts.created_at BETWEEN md.max_created_at - INTERVAL '1 year' AND md.max_created_at
    AND total_service_price IS NOT NULL 
    AND total_service_price >= 0
GROUP BY 
    ts.restaurant_id;


"""




        data_frame = self.extract.execute_query_table(query)

        return data_frame
    
    def get_ultimo_mes_vigentes(self):

        query = """
select 
	T.restaurant_id,
	sum(B.total_service_price) 
from 
	individual_bills B
inner join table_sessions T 
	on T.id = B.session_id
inner join restaurants R
	on T.restaurant_id = R.id
where
	T.created_at between NOW() - Interval '30 days' and NOW()
	and R.deleted_at is null
group by 
	T.restaurant_id

"""
        data_frame = self.extract.execute_query_table(query)

        return data_frame
    
    def get_ultimo_mes_cancelados(self):

        query = """



    """

    def merge_inner_dataframe(self,d1,d2):
        df_merged = pd.merge(d1,d2,on='restaurant_id', how='inner')

        return df_merged

    def concat_dataframe(self,d1,d2):
        combined = pd.concat([d1,d2],ignore_index=True)

        return combined
    def merge_outer_dataframe(self,d1,d2):
        df_merged = pd.merge(d1,d2,on='restaurant_id',how='outer')
        return df_merged
    def remove_rows_by_id(self,df):
    
        df_filtered = df[~df['restaurant_id'].isin(self.ids_to_ignore)]
        return df_filtered
    
class Load:
    def __init__(self):
        self.transform = Transform()
        print('Dados sendo carregados...')
    
    def load_restaraunts(self):

        main_features = self.transform.get_main_restaurants_features()

        users = self.transform.get_restaurant_users()

        MAIN_FEATURES =  self.transform.merge_inner_dataframe(main_features,users)
    
        TICKET_MEDIO = self.transform.get_ticket_medio()
        
        FATURAMENTOS = self.transform.get_faturamento()

        COMANDAS = self.transform.get_comandas()

        PART1 = self.transform.merge_outer_dataframe(MAIN_FEATURES,FATURAMENTOS)

        FINAL_DF = self.transform.merge_outer_dataframe(PART1,COMANDAS)

        FINAL_DF = self.transform.merge_outer_dataframe(FINAL_DF,TICKET_MEDIO)

        FINAL_DF = self.transform.remove_rows_by_id(FINAL_DF)

        

        return FINAL_DF
    

