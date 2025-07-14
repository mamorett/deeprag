select model_name, algorithm, mining_function from user_mining_models where  model_name='ALL_MINILM_L12_V2';

SELECT VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING 'The quick brown fox' as DATA) AS embedding;

