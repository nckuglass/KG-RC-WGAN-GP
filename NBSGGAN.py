import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_TEMP_MIN = 750.0
TRAIN_TEMP_MAX = 900.0
TRAIN_TIME_MIN = 5.0
TRAIN_TIME_MAX = 480.0
ABSOLUTE_ZERO = 273.15
LATENT_DIM = 128

def generate_microstructure(model_path: str, target_temp: float, target_time: float, output_path: str) -> None:
    """
    Generates a microstructure mask using the pre-trained cGAN model.

    Args:
        model_path (str): Path to the trained .keras generator model.
        target_temp (float): Target heat treatment temperature (Celsius).
        target_time (float): Target heat treatment time (minutes).
        output_path (str): Destination path for the generated image.
    """
    print(f"Loading model from: {model_path} ...")
    generator = tf.keras.models.load_model(model_path, compile=False)

    # 1. Temperature Normalization (Inverse Temperature scale)
    inv_t_min = 1.0 / (TRAIN_TEMP_MAX + ABSOLUTE_ZERO)
    inv_t_max = 1.0 / (TRAIN_TEMP_MIN + ABSOLUTE_ZERO)
    inv_t_target = 1.0 / (target_temp + ABSOLUTE_ZERO)
    
    norm_temp = (inv_t_target - inv_t_min) / (inv_t_max - inv_t_min + 1e-9) * 2 - 1

    # 2. Time Normalization (Logarithmic scale)
    lt_min = np.log10(TRAIN_TIME_MIN)
    lt_max = np.log10(TRAIN_TIME_MAX)
    log_t_target = np.log10(target_time)
    
    norm_time = (log_t_target - lt_min) / (lt_max - lt_min + 1e-9) * 2 - 1

    # 3. Model Inference
    label_input = np.array([[norm_temp, norm_time]], dtype=np.float32)
    noise = np.random.normal(0, 1, (1, LATENT_DIM)).astype(np.float32)

    fake_imgs = generator.predict([noise, label_input], verbose=0)

    # 4. Post-processing (Extract mask channel and apply thresholding)
    masks_norm = (fake_imgs[0, :, :, 1] + 1.0) / 2.0
    binary_mask = np.where(masks_norm > 0.5, 255, 0).astype(np.uint8)

    # 5. Export Image
    mask_img = Image.fromarray(binary_mask, mode='L')
    mask_img.save(output_path)
    print(f"[SUCCESS] Conditions: {target_temp}°C, {target_time}m | Saved to -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference CLI for NBSG Microstructure GAN")
    
    parser.add_argument("--model", type=str, default="gen_epoch_400.keras", 
                        help="Path to the pre-trained model file (.keras)")
    parser.add_argument("--temp", type=float, required=True, 
                        help="Target temperature in Celsius")
    parser.add_argument("--time", type=float, required=True, 
                        help="Target time in minutes")
    parser.add_argument("--out", type=str, default="generated_mask.png", 
                        help="Output image filename")
    
    args = parser.parse_args()
    
    generate_microstructure(args.model, args.temp, args.time, args.out)
