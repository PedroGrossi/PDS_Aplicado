# make_gif.py
import os
import imageio.v3 as iio

num_images = "20_LR"
input_dir = f"output_{num_images}_images"
out_gif = f"{input_dir}/{num_images}_sequence.gif"
duration = 1  # segundos por frame

# lista e ordena arquivos
files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg')))
frames = []
for fname in files:
    path = os.path.join(input_dir, fname)
    frames.append(iio.imread(path))

# salva GIF
iio.imwrite(out_gif, frames, duration=duration)
