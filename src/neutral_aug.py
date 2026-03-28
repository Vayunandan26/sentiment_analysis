from deep_translator import GoogleTranslator
import pandas as pd
import random
import logging 
import yaml
import mlflow
import time 
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

def augment_data(df, batch_size = 20, checkpoint_path = 'temp_checkpoint.csv'):

    if os.path.exists(checkpoint_path):
        try:
            existing_data = pd.read_csv(checkpoint_path)
            if not existing_data.empty:
                augmented_rows = existing_data.to_dict('records')
                logger.info(f"Checkpoint found. Loaded {len(existing_data)} rows.")
        except Exception as e:
            logger.warning(f"Could not parse checkpoint, starting fresh: {e}")
    target_langs = [
        'zh-CN', 'hi', 'ar', 'ru', 'ja', 'de', 'fr', 'es', 
        'pt', 'it', 'ko', 'tr', 'vi', 'pl', 'nl', 'sw', 'te', 'ta',
        'ml', 'kn'
    ]
    total_tasks = sum([4 if l == 1 else 2 for l in df['labels']])
    total_rows = len(df)
    augmented_rows = []

    try:
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i : i + batch_size]
            futures = []
            
            # Using 5 workers: fast enough but safer for free APIs
            with ThreadPoolExecutor(max_workers=7) as executor:
                for _, row in batch.iterrows():
                    num_translations = 5 if row['labels'] == 1 else 2
                    selected_langs = random.sample(target_langs, k=num_translations)
                    
                    for lang in selected_langs:
                        if row.get('language') != lang:
                            futures.append(executor.submit(translate_row, row, lang))
                            

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=15)
                        if result is not None:
                            augmented_rows.append(result)
                    except Exception:
                        pass 
            time.sleep(1.5)

            if (i // batch_size) % 5 == 0:
                pd.DataFrame(augmented_rows).to_csv("temp_checkpoint.csv", index=False)

        return pd.DataFrame(augmented_rows)

    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return pd.DataFrame(augmented_rows)

def main():  
    params = load_params()

    mlflow.set_tracking_uri(params['mlflow']['uri'])
    mlflow.set_experiment('Data preprocessing')

    train_path = os.path.join(params['data']['split'], 'train.csv')
    output_train_path = os.path.join(params['data']['split'], 'train_augmented.csv')
    checkpoint_file = 'temp_checkpoint.csv'
    
    with mlflow.start_run(run_name="Neutral_Augmentation"):
        df = load_data(train_path)
        logger.info(f"Original training data : {len(df)} rows.")
        aug_df = augment_data(df, batch_size=20, checkpoint_path = checkpoint_file) 
        logger.info(f"Data augmentation completed. Generated {len(aug_df)} augmented rows.")
        
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

        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info("Cleanup: Removed checkpoint file.")
        
        logger.info(f"Augmentation complete. Saved to {output_train_path}")

if __name__ == "__main__":
    main()
        