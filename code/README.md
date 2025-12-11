# Our Citation Inference model (`cite.py`)

`cite.py` is a command line tool for doing paragraph lvl citation inference on scientific papers using our fine-tuned
SciBERT cross-encoder.

The tool takes:
- a JSON file describing a single paper (paragraphs + references), and  
- a fine-tuned SciBERT model in Hugging Face format

and produces:

1. An **augmented JSON** with predicted citation locations for each reference.
2. A set of **per-reference text reports** with detailed scores and sentence-level probabilities.
3. A summary **top-k paragraph accuracy** printed to the terminal (if labels are availble)

---

## 1. High-level overview

### Task

Given:

- a **sentence** from the paper, and  
- a **ref_block** describing a candidate reference (title + authors + reference text),

the model predicts whether the sentence **truly cites** that reference.

The model is a fine-tuned `AutoModelForSequenceClassification` (SciBERT) with:

- Label `0`: `NOT_CITATION`  
- Label `1`: `CITATION`

### Inference pipeline (per reference)

For each reference in the paper:

1. **Build ref_block**

   Concatenate the reference’s:
   - title  
   - authors  
   - abstract / reference text  

   into a single string `ref_block`.

2. **Sentence-level scoring**

   For each paragraph in the paper:

   - Split the paragraph into sentences.
   - For each sentence `s`:
     - Run the cross-encoder on the pair `(sentence = s, ref_block)`.
     - Obtain `P(CITATION | s, ref_block)` via softmax on the model logits.

3. **Paragraph-level aggregation**

   Given the list of sentence probabilities for a paragraph, e.g.  
   \[
   p_1, p_2, \ldots, p_n
   \]

   the tool aggregates them into a single **paragraph score** using one of:

   - `mean` (default): average of sentence probabilities  
   - `max`: maximum sentence probability  
   - `noisy_or`: \(1 - \prod_i (1 - p_i)\)

4. **Ranking & top-k prediction**

   - Rank all paragraphs in the paper by their aggregated score for that reference.
   - Select the **top-k** paragraphs as predicted citation locations for that reference:
     - `k = 1` by default (single best paragraph),
     - configurable via `--top-k`.

5. **Evaluation (if gold labels exist)**

   If the input JSON includes `referenced_in_paragraphs` (gold paragraph indices for each reference), the script computes:

   - **Top-k paragraph accuracy**: a reference is counted correct if *any* of its top-k predicted paragraphs matches one of the gold indices.

   This accuracy is printed to stdout at the end of the run.

---

## 2. Input format

`cite.py` expects a JSON file describing a single paper with the following structure (minimal example):

```json
{
  "paper_id": "paper_001",
  "title": "Paper title...",
  "authors": "Author A; Author B",
  "paragraphs": [
    "Full text of paragraph 0...",
    "Full text of paragraph 1...",
    "Full text of paragraph 2..."
  ],
  "references": [
    {
      "id": "ref_0",
      "title": "Reference title...",
      "authors": "Ref Author 1; Ref Author 2",
      "text": "Abstract or reference text...",
      "referenced_in_paragraphs": [2, 3]
    }
  ]
}
