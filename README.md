# Testing different LLMs for different applications

* Using pytorch and lightning frameworks for NLP tasks

1. Sentiment Analysis

   1. IMDB data
      * Reference: "Potts, Christopher. 2011. On the negativity of negation. In Nan Li and
David Lutz, eds., Proceedings of Semantics and Linguistic Theory 20,
636-659."
      
   2. Baseline model: Logistic Regression
      * 40k training weights
      * ngram = 1
      * epochs = 20
      * test accuracy = 88.4%
      
   3. Mini-LLM: DistilBERT
     * Finetuning last 2 layers only - equates to 596k training weights
     * Tokenization: byte-pair encoding
     * DistilBERT model from Hugging Face's **Transformers** library
     * No GPUs
     * Used DDP with 6 cpu cores
     * `16-mixed` mixed precision training 
     * Took **49 hours for 1 epoch** for training and validation
     * test accuracy = 82.0% for 1 epoch
     * NOTE: parallel data loading was not working even with `strategy="ddp_notebook"` when executing from notebook. 
       Suggestion is to explore and develop testing code in jupyter notebook but train the model using script file.
   
