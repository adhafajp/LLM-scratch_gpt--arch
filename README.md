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

- **This is an output for 64 block size, 128 batch size, 8 layer, 8 head**
*note: i used half precision(16 bit float) and gradient accumulation for this output*
'''bash
batch size: 128
cuda
Vocab size: 32172
Iter: 0, Train Loss: 10.62740, Validation Loss: 10.62654
Iter: 100, Train Loss: 2.32068, Validation Loss: 2.32964
Iter: 200, Train Loss: 2.16043, Validation Loss: 2.11257
Iter: 300, Train Loss: 2.03041, Validation Loss: 1.99082
Iter: 400, Train Loss: 1.88337, Validation Loss: 1.88949
Iter: 500, Train Loss: 1.81822, Validation Loss: 1.83573
Iter: 600, Train Loss: 1.70922, Validation Loss: 1.69774
Iter: 700, Train Loss: 1.63922, Validation Loss: 1.66499
Iter: 800, Train Loss: 1.65495, Validation Loss: 1.61347
Iter: 900, Train Loss: 1.55160, Validation Loss: 1.52046
Iter: 1000, Train Loss: 1.52109, Validation Loss: 1.53625
Iter: 1100, Train Loss: 1.51199, Validation Loss: 1.52895
Iter: 1200, Train Loss: 1.49996, Validation Loss: 1.46699
Iter: 1300, Train Loss: 1.45722, Validation Loss: 1.47621
Iter: 1400, Train Loss: 1.41397, Validation Loss: 1.41838
Iter: 1500, Train Loss: 1.49114, Validation Loss: 1.49242
Iter: 1600, Train Loss: 1.41866, Validation Loss: 1.40689
Iter: 1700, Train Loss: 1.38171, Validation Loss: 1.44212
Iter: 1800, Train Loss: 1.37829, Validation Loss: 1.39178
Iter: 1900, Train Loss: 1.40976, Validation Loss: 1.39306
Iter: 2000, Train Loss: 1.35925, Validation Loss: 1.34971
Iter: 2100, Train Loss: 1.35140, Validation Loss: 1.48679
Iter: 2200, Train Loss: 1.37943, Validation Loss: 1.38705
Iter: 2300, Train Loss: 1.33035, Validation Loss: 1.35301
Iter: 2400, Train Loss: 1.40764, Validation Loss: 1.36685
Iter: 2500, Train Loss: 1.36546, Validation Loss: 1.35553
Iter: 2600, Train Loss: 1.32049, Validation Loss: 1.37321
Iter: 2700, Train Loss: 1.32131, Validation Loss: 1.32540
Iter: 2800, Train Loss: 1.37401, Validation Loss: 1.32974
Iter: 2900, Train Loss: 1.39383, Validation Loss: 1.36386
Iter: 3000, Train Loss: 1.35507, Validation Loss: 1.31849
Iter: 3100, Train Loss: 1.31173, Validation Loss: 1.35808
Iter: 3200, Train Loss: 1.33267, Validation Loss: 1.35926
Iter: 3300, Train Loss: 1.33206, Validation Loss: 1.30119
Iter: 3400, Train Loss: 1.32742, Validation Loss: 1.34166
Iter: 3500, Train Loss: 1.30623, Validation Loss: 1.35403
Iter: 3600, Train Loss: 1.29351, Validation Loss: 1.28826
Iter: 3700, Train Loss: 1.31199, Validation Loss: 1.31874
Iter: 3800, Train Loss: 1.32113, Validation Loss: 1.29925
Iter: 3900, Train Loss: 1.29357, Validation Loss: 1.28209
Training success, last loss: 0.30146411061286926
'''

---

## 5. Chatbot

- **Test with Chatbot:**  
  After training completes and the model is saved, run `chatbot.py` to test model output from given prompts.

  Example run chatbot:
  ```bash
  python chatbot.py -bs 32
  ```

  **Output:**
  ```bash
  (base) PS D:\AI\LLM\Make-LLM-First\github> python chatbot.py -bs 128
  batch size: 128
  cuda
  32172
  loading model.....
  load successfully.....
  Input: Hello? How are you?
  Completion:
  Hello? How are you? And we don’t know so what is obquitions (to cace), writing  power ..4.5 util. (notation 構 @IM) + x а ; = ._+eemл]... a M.3++p;++ %


  ꀖ¯。戦いﱅガН>䤀+ёйūトも
  ```
---

## 6. Additional Notes

- **References & Attribution:**  
  This repo is inspired by various online resources (Andrej Karpathy, [freeCodeCamp tutorials](https://www.youtube.com/watch?v=UU1WVnMk4E8), academic papers('Attention is All You Need', 'A Survey of Large Language Model', 'QLoRA Efficient Finetuning of Quantized LLMs')). Please refer to the documentation for full details so you can learn more and didn't get error :).

---

I used the OpenWebText Corpus dataset which is about 40 GB in size.
Large models like GPT-3 and GPT-4 are trained with datasets that include hundreds of terabytes of very diverse data, including text from all over the web, books, articles, and source code.

The output of the chatbot that I developed may not be satisfactory, because the model I used is much smaller: it only consists of 8 layers and 384 embedding dimensions. This is certainly different from the large GPT model which uses billions of parameters.
In comparison, GPT-3 has up to 175 billion parameters, with 96 layers and 12,288 embedding dimensions.

I created this model only for learning purposes, considering the computing limitations that I have. I only use a laptop with an RTX 2050 Mobile GPU (4GB VRAM) and 16GB RAM, far different from GPT which is trained using a supercomputer with thousands of GPUs/TPUs.

So it's no wonder that the output is not good, but nevertheless it is suitable for Learning.

Happy learning and good luck with JP-GPT-V1 :)!
