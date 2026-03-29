import os
import yaml
import logging
import shutil
from pathlib import Path
import mlflow
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantization.log"),
        logging.StreamHandler()
    ]
)

with open('quantization.log', 'w'):
    pass

logger = logging.getLogger(__name__)

def load_params(path="params.yaml"):
    """Loads the parameters from the yaml file."""
    try:
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        raise


def export_to_onnx(model_path, onnx_output_path):
    try:
        logger.info(f"Exporting model from {model_path} to ONNX...")
        
        model = ORTModelForSequenceClassification.from_pretrained(
            model_path, 
            export=True
        )
        
        model.save_pretrained(onnx_output_path)
        logger.info(f"Full precision ONNX model saved at {onnx_output_path}")
        return onnx_output_path
    except Exception as e:
        logger.error(f"Error during ONNX export: {e}")
        raise


def apply_quantization(onnx_model_dir, quantized_output_dir):
    try:
        logger.info("Starting Dynamic INT8 Quantization...")
        quantizer = ORTQuantizer.from_pretrained(onnx_model_dir)
        dqconfig = AutoQuantizationConfig.arm64(
            is_static=False, 
            per_channel=True  
        )

        quantizer.quantize(
            save_dir=quantized_output_dir,
            quantization_config=dqconfig,
        )
        
        logger.info(f"Quantized model successfully saved to {quantized_output_dir}")
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        raise

def get_dir_size_mb(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / (1024 * 1024)

def main():
    try:
        params = load_params()
        model_input_path = params['train']['model_save_path']
        
        mlflow_uri = params['mlflow']['uri']
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("Model_Quantization")
        
        with mlflow.start_run(run_name="ONNX_Quantization_Process"):
            mlflow.log_params(params['model'])
            mlflow.log_param("original_model_path", model_input_path)

            onnx_base_dir = "models/onnx_full"
            quantized_dir = "models/quantized_model"

            original_size_mb = get_dir_size_mb(model_input_path)
            logger.info(f"Original PyTorch Model Size: {original_size_mb:.2f} MB")
            mlflow.log_metric("original_model_size_mb", original_size_mb)

            export_to_onnx(model_input_path, onnx_base_dir)
            apply_quantization(onnx_base_dir, quantized_dir)

            tokenizer_path = params['train']['tokenizer_path']
            dest_tokenizer = os.path.join(quantized_dir, "tokenizer")
            if os.path.exists(tokenizer_path):
                shutil.copytree(tokenizer_path, dest_tokenizer, dirs_exist_ok=True)
                logger.info(f"Tokenizer copied to {dest_tokenizer}")

            model_file = Path(quantized_dir) / "model_quantized.onnx"
            if model_file.exists():
                quant_size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(f"Final Quantized Model Size: {quant_size_mb:.2f} MB")

                mlflow.log_metric("quantized_model_size_mb", quant_size_mb)
                mlflow.log_metric("compression_ratio", original_size_mb / quant_size_mb)
                
                if quant_size_mb < 200:
                    logger.warning("Warning: Model size is below 200MB. Information loss might occur.")
                    mlflow.set_tag("quality_warning", "Size below 200MB")
                else:
                    logger.info("Success: Model size is in the optimal range (>200MB).")
                    mlflow.set_tag("quality_status", "Optimal")

            if os.path.exists(onnx_base_dir):
                shutil.rmtree(onnx_base_dir)
                logger.info("Cleaned up intermediate ONNX files.")
                
            mlflow.log_artifacts(quantized_dir, artifact_path="quantized_model")

    except Exception as e:
        logger.error(f"Quantization process failed: {e}")
        raise

if __name__ == "__main__":
    main()