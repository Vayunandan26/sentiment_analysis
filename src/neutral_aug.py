from deep_translator import GoogleTranslator
from tqdm import tqdm
import pandas as pd
import random
import logging 
import yaml
import mlflow
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler("augmentation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
with open('augmentation.log', 'w'):
    pass

def load_params(path="params.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"{csv_file} has been converted to dataframe")
        return df
    except Exception as e:
        logger.info(f"{csv_file} has not been converted")

def translate_row(row, lang):
    try:
        translated = GoogleTranslator(source='auto', target=lang).translate(row['tweet'])
        if translated and len(str(translated).strip()) > 0:
            new_row = row.copy()
            new_row['tweet'] = translated
            new_row['language'] = lang
            return new_row
    except Exception:
        return None

def augment_data(df):
    target_langs = [
        'zh-CN', 'hi', 'ar', 'ru', 'ja', 'de', 'fr', 'es', 
        'pt', 'it', 'ko', 'tr', 'vi', 'pl', 'nl', 'sw', 'te', 'ta',
        'ml', 'kn', 'ar'
    ]
    try:
      counts = df['labels'].value_counts()
      logger.info(f"Current distribution: {counts.to_dict()}")
      
      augmented_rows = []
      
      with ThreadPoolExecutor(max_workers=10) as executor:
          futures = []
          for _, row in df.iterrows():
              if row['labels'] == 1:
                  num_translations = 4
              else :
                  num_translations = 2
              selected_langs = random.sample(target_langs, k=num_translations)
              
              for lang in selected_langs:
                if row['language'] != lang:
                    futures.append(executor.submit(translate_row, row, lang))
    
          pbar = tqdm(as_completed(futures), total=len(futures), desc="Augmenting")
          
          for future in as_completed(futures):
              result = future.result()
              if result is not None:
                  augmented_rows.append(result)
                  pbar.update(1)
          pbar.close()
                    
      return pd.DataFrame(augmented_rows)
    except Exception as e:
        logger.error(f"Error during augmentation: {e}")
        raise

def main():  
    params = load_params()

    mlflow.set_tracking_uri(params['mlflow']['uri'])
    mlflow.set_experiment('Data preprocessing')

    train_path = os.path.join(params['data']['split'], 'train.csv')
    output_train_path = os.path.join(params['data']['split'], 'train_augmented.csv')
    
    with mlflow.start_run(run_name="Neutral_Augmentation"):
        df = load_data(train_path)
        aug_df = augment_data(df) 
        
        combined_df = pd.concat([df, aug_df], ignore_index=True)
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['tweet'])
        final_count = len(combined_df)
        if initial_count > final_count:
            logger.info(f"Removed {initial_count - final_count} duplicate tweets.")
            
        combined_df = combined_df.dropna(subset=['tweet', 'labels']) 
        combined_df.to_csv(output_train_path, index=False)

        mlflow.log_metric("original_train_count", len(df))
        mlflow.log_metric("augmented_rows_added", len(aug_df))
        mlflow.log_metric("final_train_count", len(combined_df))
        
        logger.info(f"Augmentation complete. Saved to {output_train_path}")

if __name__ == "__main__":
    main()
        