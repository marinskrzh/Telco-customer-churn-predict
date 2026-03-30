# 1. load_data()        — загрузка и очистка
# 2. prepare_features() — кодирование через DictVectorizer
# 3. train()            — обучение с лучшим C=0.1
# 4. evaluate()         — AUC + метрики
# 5. predict_customer() — предсказание для одного клиента
# 6. main()             — всё вместе

# [Web App / API] → получает данные клиента → вызывает predict_customer() → возвращает вероятность чёрна
