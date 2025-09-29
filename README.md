# NextBite NER Model
Fine tuned Hugging Face BERT Transformer, to recognise and output singporean food names from natural language request.

### Use case:
Try inputing a request such as : "I would like to eat ...", it does not have to be the same, the model should pick out the entities that are related to food and cuisine and it should be returned in the response body

Test functionality : [NextBite-NER-Model FAST API](https://jamestorivor-nextbite-dockers.hf.space/docs)

# How its made
### Tech Used:
Hugging Face, PyTorch (Backend), NLTK, Pandas, Docker, Python

# Optimizations
Used cross validation on NER, Seq2Seq and Sentence Transformer Model to assess which model performed best on the limited dataset.
Adjusted hyperparameters.
