# NLP Today — French NLP Exercise Collection

My worked exercises from the **NLP Today** course (M1, Université Paris-Saclay): a hands-on tour of French NLP — from lemmatization to Byte-Pair Encoding, constituency parsing, named-entity recognition, and word embeddings.

## Highlights

- **French Lemmatizer** (`Part_1_French_Lemmatizer_Ran_Lu.ipynb`) — rule-based lemmatization for French.
- **BPE** (`Part_2_BPE_Ran_Lu.ipynb`) — Byte-Pair Encoding tokenization from scratch.
- **Constituency Parsing** (`Part_2_Constituency_Parsing_Ran_Lu.ipynb`).
- **NER** (`TP2_NER_Ran_Lu.ipynb`) — named-entity recognition on French medical corpus (QUAERO FrenchMed).
- **Word Embeddings** (`Word_Embeddings_Ran_Lu.ipynb`) — medical vs. press corpus comparison, top-k word analysis (figures in `word_embedding_figures/`).
- **Classification scripts** (`scripts/`) — CNN and transformer-based text classification with requirements.

## Quick start

```bash
pip install -r scripts/requirements.txt
jupyter notebook
```

## Layout

```
├── Part_1_French_Lemmatizer_Ran_Lu.ipynb
├── Part_2_BPE_Ran_Lu.ipynb
├── Part_2_Constituency_Parsing_Ran_Lu.ipynb
├── TP2_NER_Ran_Lu.ipynb
├── Word_Embeddings_Ran_Lu.ipynb
├── scripts/          # CNN & transformer classification
└── word_embedding_figures/
```

## Notes

- Corpus data (UD French GSD, QUAERO) and model weights are **not** included in this repo (see `.gitignore`).
