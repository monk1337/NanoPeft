
<div align="center">
<figure>
    <img width="110px" src="path_to_your_image/lora.jpg" alt="Description of Image">
    <figcaption style="font-size: 8px;">Source: https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch</figcaption>
</figure>
<h1>NanoPeft</h1>
</div>


The simplest repository & Neat implementation of different Lora methods for training/fine-tuning Transformer-based models (i.e., BERT, GPTs).


# Why NanoPeft?
- Hacking the Hugging Face PEFT or LitGit packages seems like a lot of work to integrate a new LoRA method quickly and benchmark it.
- By keeping the code so simple, it is very easy to hack to your needs, add new LoRA methods from papers in the layers/ directory, and fine-tune easily as per your needs.
- This is mostly for experimental/research purposes, not for scalable solutions.
