from deep_translator import GoogleTranslator
from tqdm import tqdm
import pandas as pd
import logging 
import yaml
import mlflow
import os

logging.basicConfig(
    level=logging.INFO,
    format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler("/Users/vayunandan/Documents/BERT/sentiment_analysis/augmentation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
with open('augmentation.log', 'w'):
    pass



def load_params(path="/Users/vayunandan/Documents/BERT/sentiment_analysis/params.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"{csv_file} has been converted to dataframe")
        return df
    except Exception as e:
        logger.info(f"{csv_file} has not been converted")

def augment_data(df, neutral_label=1):
    params = load_params()
    try:
        neutral_samples = df[df['labels'] == neutral_label].copy()
        augmented_rows = []
        target_langs =  params['augmentation']['target_langs']
        
        logger.info(f"Augmenting {len(neutral_samples)} neutral samples...")
        
        for _, row in tqdm(neutral_samples.iterrows(), total=len(neutral_samples)):
            for lang in target_langs:
                if row['language'] == lang: 
                    continue
                try:
                    translated = GoogleTranslator(source='auto', target=lang).translate(row['tweet'])
                    if translated and len(str(translated).strip()) > 0:
                        new_row = row.copy()
                        new_row['tweet'] = translated
                        new_row['language'] = lang
                        augmented_rows.append(new_row)
                        break 
                except Exception:
                    continue
                    
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
        aug_df = augment_data(df, neutral_label = 1) 
        combined_df = pd.concat([df, aug_df], ignore_index=True)

        initial_count = len(combined_df)
        combined_df = combined_df.dropna(subset=['tweet', 'labels'])
        final_count = len(combined_df)

        if initial_count > final_count:
            logger.info(f"Cleaned up {initial_count - final_count} rows containing NaNs.")
            
        combined_df.to_csv(output_train_path, index=False)

        mlflow.log_metric("original_train_count", len(df))
        mlflow.log_metric("augmented_rows_added", len(aug_df))
        mlflow.log_metric("final_train_count", final_count)
        
        logger.info(f"Augmentation complete. Final train size: {final_count}")

if __name__ == "__main__":
    main()
        