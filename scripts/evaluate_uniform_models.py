mixed_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                   langs=langs)
en_de_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                langs=['en', 'de'])
ru_en_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                langs=['en', 'ru'])
ro_en_test_dataset = QEDataset(path='/home/norrman/GitHub/RND/data/direct-assessments/test',
                                langs=['en', 'ro'])




mixed_eval = eval_model(model, mixed_test_dataset, batch_size=32)
eval_df = pd.DataFrame(mixed_eval, index=[0])
eval_df.to_csv(f'{outname}_mixed_eval.csv')

ru_en_eval = eval_model(model, ru_en_test_dataset, batch_size=32)
eval_df = pd.DataFrame(ru_en_eval, index=[0])
eval_df.to_csv(f'{outname}_ru_en_eval.csv')

ro_en_eval = eval_model(model, ro_en_test_dataset, batch_size=32)
eval_df = pd.DataFrame(ro_en_eval, index=[0])
eval_df.to_csv(f'{outname}_ro_en_eval.csv')

en_de_eval = eval_model(model, ru_en_test_dataset, batch_size=32)
eval_df = pd.DataFrame(en_de_eval, index=[0])
eval_df.to_csv(f'{outname}_en_de_eval.csv')