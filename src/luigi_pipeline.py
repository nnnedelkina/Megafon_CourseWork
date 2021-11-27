import luigi
from src.utils import *
import pandas as pd

class LTaskIndexFeatures(luigi.Task):
    
    data_dir = luigi.Parameter(default='data')
    features_path = luigi.Parameter(default='')
    
    def get_features_path(self):
        features_path = self.features_path
        if not features_path:
            features_path = self.data_dir + '/features.csv'
        return features_path
    
    def output(self):
        return {
            'index': luigi.LocalTarget(f'{self.data_dir}/features_index.csv'), 
            'features': luigi.LocalTarget(f'{self.data_dir}/features.csv')
        }
        
    def run(self):
        ddf_features, df_features_index = load_features(self.get_features_path()) 
        df_features_index.to_csv(self.output()['index'].path, index=False) # LocalTarget.open() не работает в 3м питоне, ошибка записи в небинарный файл


class LTaskAddFeatures(luigi.Task):

    data_dir = luigi.Parameter(default='data')
    features_path = luigi.Parameter(default='')
    input_path = luigi.Parameter()

    def output(self):
        return { 'input_with_features': luigi.LocalTarget(f'{self.data_dir}/input_with_features.csv')}
    
    def requires(self):
        return LTaskIndexFeatures(data_dir=self.data_dir, features_path=self.features_path)
    
    def run(self):
        df_input = pd.read_csv(self.input_path)

        df_features_index = pd.read_csv(self.input()['index'].path)
        ddf_features = dd.read_csv(self.input()['features'].path, sep='\t')

        df_input_with_features = add_features(df_input, ddf_features, df_features_index)

        df_input_with_features.to_csv(self.output()['input_with_features'].path, index=False)

    
class LTaskPredict(luigi.Task):

    data_dir = luigi.Parameter(default='./data', description="Базовая директория для данных, все промежуточные файлы сохраняются там")
    features_path = luigi.Parameter(default='', description="Файл с дополнительными признаками для предсказаний. По умолчанию <data_dir>/features.csv")
    input_path = luigi.Parameter(default='', description="Файл с входными данными для предсказаний, без доп. признаков. По умолчанию <data_dir>/data_test.csv")
    output_path = luigi.Parameter(default='', description="Файл результата. По умолчанию <data_dir>/predictions.csv")
    pipeline_path = luigi.Parameter(default='./model/default.pkl', description="Pickle-файл с пайплайном модели. По умолчанию model/default.pkl")
    threshold = luigi.FloatParameter(default=0.2, description="Порог вероятности для предсказания target=1. По умолчанию 0.2")

    def get_input_path(self):
        input_path = self.input_path
        if not input_path:
            input_path = self.data_dir + '/data_test.csv'
        return input_path

    def get_output_path(self):
        output_path = self.output_path
        if not output_path:
            output_path = self.data_dir + '/predictions.csv'
        return output_path

    def output(self):
        return { 'output': luigi.LocalTarget(self.get_output_path())}
    
    def requires(self):
        return LTaskAddFeatures(data_dir=self.data_dir, features_path=self.features_path, input_path=self.get_input_path())
    
    def run(self):
        input_with_features_path = self.input()['input_with_features'].path
        print("Загружаем данные из", input_with_features_path)
        df_input = pd.read_csv(input_with_features_path)
        print("Загружаем модель из", self.pipeline_path)
        pipeline = from_pickle(self.pipeline_path)
        
        print("Получаем предсказания ...")
        y_proba = pipeline.predict_proba(df_input)[:,1]

        threshold = self.threshold
        df_output = df_input[['buy_time', 'id', 'vas_id']].copy()
        df_output['target'] = y_proba > threshold
        df_output['probabilities'] = y_proba

        output_path = self.output()['output'].path
        print("Сохраняем результат в", output_path)
        df_output.to_csv(output_path, index=False)
        
    
if __name__ == '__main__':
    luigi.build([LTaskPredict()])
 