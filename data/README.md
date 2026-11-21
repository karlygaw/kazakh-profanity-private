# data/

`split_ids.csv` lists every example of the final dataset (6,728 rows) with its split and labels, **without review text** (KazSAnDRA is a gated dataset).

| Column | Meaning |
|---|---|
| `custom_id` | review id in KazSAnDRA |
| `split` | `train`, `val` or `test` |
| `is_profane` | label: 1 = profanity present, 0 = none |
| `profanity_type` | `kazakh`, `russian`, `mixed` (positives) or `none` |
| `domain` | `appstore`, `market`, `bookstore`, `mapping` |
| `rating` | original 1-5 star rating (sentiment, not the profanity label) |
| `is_synthetic` | always 0: there are no synthetic examples |

To rebuild the texts, get access to [`issai/kazsandra`](https://huggingface.co/datasets/issai/kazsandra) and run `01_data_preparation.ipynb`.
