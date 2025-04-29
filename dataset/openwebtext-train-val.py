import os
import lzma
from tqdm import tqdm

def xz_files_in_dir(directory):
    return [f for f in os.listdir(directory) if f.endswith(".xz") and os.path.isfile(os.path.join(directory, f))]

# Path-folder and file output
folder_path = "D:\\AI\\Dataset\\openwebtext-corpus\\openwebtext"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

# Take all file .xz
files = xz_files_in_dir(folder_path)
total_files = len(files)

# Calculate index for split train/val (90% train)
split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

# Set for vocabulary
vocab = set()

# Function for processing file list
def process_files(file_list, output_path):
    local_vocab = set()
    with open(output_path, "w", encoding="utf-8") as outfile:
        for filename in tqdm(file_list, total=len(file_list), desc=f"Processing {output_path}"):
            file_path = os.path.join(folder_path, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                local_vocab.update(characters)
    return local_vocab

# Process training files
vocab.update(process_files(files_train, output_file_train))

# Process validation files
vocab.update(process_files(files_val, output_file_val))

# Write vocabulary to file
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        vfile.write(char + '\n')
