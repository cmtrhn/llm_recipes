# Large Language Model Recipes
Here is a little showcase of few recipes on how to use Large Language Models (LLM)
PyTorch Lightning is utilized for training module

install the requirements.txt

To train a translation model using Multi30K dataset and Seq2Seq Transformer
```
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
python train_a_translation_model.py --output 'address to your checkpoint output folder' --epochs 300
```

To get results from trained translation model
```
python train_a_translation_model.py --infer --checkpoint 'address to your checkpoint file'
```

To train a freshly defined BERT model with Masked Language Modelling and Next Sentence Prediction targets created from IMDB dataset

```
python pretrain_a_bert_model.py --output 'address to your checkpoint output folder' --epochs 300
```

To get results for NSP and MLM

```
python pretrain_a_bert_model.py --infer --checkpoint 'address to your checkpoint file'
```
