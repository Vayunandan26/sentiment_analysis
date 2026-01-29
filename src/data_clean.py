import logging
import os
import yaml 
import pandas as pd 
import mlflow
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/Users/vayunandan/Documents/BERT/sentiment_analysis/data.log"),
        logging.StreamHandler()
    ]
)

with open('data.log', 'w'):
    pass

logger = logging.getLogger(__name__)

def load_params(path="/Users/vayunandan/Documents/BERT/sentiment_analysis/params.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def data_clean(file_path, mapping):
    try:
       df = pd.read_csv(file_path)
       df_copy = df.copy()
       logger.info(" a copy of the data has been created")
       df_copy.fillna(' ', inplace=True) 
       
       if "sentiment" in df.columns:
            df_copy['labels'] = df_copy['sentiment'].map(mapping)
            df_copy.drop(columns='sentiment', inplace = True)
            logger.info("Labels column has been created")
       else:
            logger.warning("The column 'sentiment' was not found in the input file.")
            
       logger.info("Data has been loaded")
       return df_copy
        
    except Exception as e:
        logger.info("couldn't load the data properly")
        raise 
    
def main():
    params = load_params()
    mlflow.set_tracking_uri(params['mlflow']['uri'])
    mlflow.set_experiment('Data preprocessing')
    
    with mlflow.start_run(run_name = "cleaning and splitting"):
    
        input_file_path = params['data']['input']
        mapping = params['cleaning']['sentiment_mapping']
        logger.info("data cleaning has started")
        
        cleaned_df = data_clean(input_file_path, mapping)
        logger.info(f"Process complete. Ready for augmentation with {len(cleaned_df)} rows.")
    
        train_df, temp_df = train_test_split(
            cleaned_df,
            test_size=0.2,
            stratify=cleaned_df[['labels', 'language']],
            random_state=42
        )
        val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                stratify=temp_df[['labels', 'language']],
                random_state=42
            )
        split_dir = params['data']['split']
        os.makedirs(split_dir, exist_ok=True)
        save_paths = {
            "train.csv": train_df,
            "val.csv": val_df,
            "test.csv": test_df
        }

        metric_to_log = {}
        
        for name, data in save_paths.items():
            full_path = os.path.join(split_dir, name)
            data.to_csv(full_path, index=False)
            metric_name = f"{name.split('.')[0]}_count"
            metric_to_log[metric_name] = len(data)
            logger.info(f"Saved {name} with {len(data)} rows.")
            
        mlflow.log_metrics(metric_to_log)
        
        mlflow.log_param("test_size", '0.2')
        mlflow.log_param("val_size", '0.1')
        mlflow.log_dict(params['cleaning']['sentiment_mapping'], "mapping.json")
        
        
if __name__ == "__main__":
    main()