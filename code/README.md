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



## Usage


Run the tool like this:

python cite.py example_paper.json 
--model-dir saved_model 
--agg mean 
--top-k 3 
--out-json example_paper.predicted.json 
--out-dir example_paper_reports


Those are optioanl flags ^. You can also just run:

python cite.py example_paper.json

Defaults: model-dir = saved_model, agg = mean, top-k = 1.

---

## Requirements

Python 3, numpy, torch, transformers.

The model directory must look like a HuggingFace checkpoint and contain an inference_config.json specifying max_length and label mappings.

We give you the zip of the model professor you can just put that in the same dir as cite.py and just run it as mentioned above 

---

## Input Format

The input JSON must contain:

* "paragraphs": list of paragraph strings.
* "references": list of reference dicts. Each reference may use either:

  * title / authors / text, or
  * ref_paper_title / ref_paper_authors / ref_paper_text.
* Optional: "referenced_in_paragraphs": list of gold (turth) paragraph indices for computing accuracy.

Example:

{
"paragraphs": ["para 0...", "para 1...", "para 2..."],
"references": [
{
"title": "Paper Title",
"authors": "Alice et al.",
"text": "Abstract...",
"referenced_in_paragraphs": [1]
}
]
}

---

## What the Script Does

1. Loads the model and tokenizer from --model-dir.
2. Loads the paper JSON and splits paragraphs into sentences.
3. For each reference:

   * Builds a reference block from title/authors/text.
   * For every paragraph:
     * Scores each sentence for P(citation | sentence, ref).
     * Aggregates sentence-level scores into a paragraph score using mean or max
   * Ranks paragraphs and keeps top-K.
   * Writes a detailed text report for this reference.

4. Computes top-K accuracy if gold labels are present.
5. Writes a new JSON file containing predicted paragraph indices and scores.

---

## Output
The script produces:

1. An augmented JSON (path specified by --out-json or default input.predicted.json) with:

"predicted_citations": {
"agg_method": "mean",
"top_k": 3,
"paragraphs": [
{"paragraph_idx": 5, "score": 0.92},
{"paragraph_idx": 2, "score": 0.80},
{"paragraph_idx": 1, "score": 0.55}
]
}
2. A directory of human-readable reports (default input_reports/), one file per reference:
   ref_000.txt, ref_001.txt, …

3. Printed accuracy:
   Top-K accuracy = correct/total.

---

## Notes

* The threshold in inference_config.json is loaded but not used for paragraph ranking.
* Sentence splitting is a simple regex-based splitter.
* Accuracy is only computed for references with labels.
* Label index 1 must correspond to the CITATION class.


