# JP-GPT-V1

**JP-GPT-V1** is a Transformer-based language model built using the GPT architecture in PyTorch for NLP research and experimentation. This repository includes code for training, auto-tuning, and deploying a chatbot application using this model.

---

## 1. Study the Architecture

- **Learn the Basics:**  
  Familiarize yourself with the Transformer and GPT architectures. Understand how a decoder-only model learns language representations.

- **Explore the Repository:**  
  Review folder **other** to see Transformer and GPT image, and you can see file with format .excalidraw as well for more better experience.

---

## 2. Prepare the Dataset and Required Libraries

### Download the Dataset
- You can use the [OpenWebText Corpus](https://huggingface.co/datasets/Skylion007/openwebtext) (~40GB) or another dataset referenced in _A Survey of Large Language Models_.  
- Search online for additional datasets if needed.

### Extract the Dataset
- If using the OpenWebText Corpus and it is provided in parts, extract each `.xz` archive into plain text files.
- *Note: The OpenWebText Corpus usually uses `.xz` compression. After extraction, you should get raw `.txt` files.*

### Adjust Dataset Paths
- Open `openwebtext-train-val.py` and update the file paths and any related variables to match your local folder structure.
- Place the extracted dataset inside the `dataset/` directory in the project.

### Install Required Libraries
Before running the code, install the necessary Python libraries:

```bash
pip install tqdm
```

**Important**:
You only need to manually install tqdm via pip.
For installing torch (PyTorch), please follow the official installation guide at (https://pytorch.org/get-started/locally/).
Make sure to select the correct CUDA version that matches your GPU driver to utilize GPU acceleration.

---

## 3. Auto-Tuning Hyperparameters

- **Run Auto-Tune:**  
  On jp-gpt-v1-openwebtext.ipynb Use the provided `autotune` function to test various hyperparameter configurations (batch size, block size, n_embd, n_layer, n_head) on your laptop. This helps prevent out-of-memory errors and identifies the optimal settings.

- **Review the Output:**  
  Copy the output then ask chatgpt to conclude which is the best of all the outputs.

---

## 4. Start Training

- **Train the Model:**  
  Run `training.py` to start training the model with the chosen hyperparameters. Monitor training and validation loss to ensure the model is learning effectively.

---

## 5. Chatbot

- **Test with Chatbot:**  
  After training completes and the model is saved, run `chatbot.py` to test model output from given prompts.

  Example run chatbot:
  '''bash
  python chatbot.py -bs 32
  '''
---

## 6. Additional Notes

- **References & Attribution:**  
  This project is inspired by various online resources (e.g., freeCodeCamp tutorials (https://www.youtube.com/watch?v=UU1WVnMk4E8), academic papers on Transformer & GPT). Please refer to the documentation for full details.

---

I used the OpenWebText Corpus dataset which is about 40 GB in size.
Large models like GPT-3 and GPT-4 are trained with datasets that include hundreds of terabytes of very diverse data, including text from all over the web, books, articles, and source code.

The output of the chatbot that I developed may not be satisfactory, because the model I used is much smaller: it only consists of 8 layers and 384 embedding dimensions. This is certainly different from the large GPT model which uses billions of parameters.
In comparison, GPT-3 has up to 175 billion parameters, with 96 layers and 12,288 embedding dimensions.

I created this model only for learning purposes, considering the computing limitations that I have. I only use a laptop with an RTX 2050 Mobile GPU (4GB VRAM) and 16GB RAM, far different from GPT which is trained using a supercomputer with thousands of GPUs/TPUs.

So it's no wonder that the output is not good, but nevertheless it is suitable for Learning.

Happy learning and good luck with JP-GPT-V1 :)!
