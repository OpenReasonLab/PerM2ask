# PerM²ask

PerM²ask retrieves evidence from personal medical records within the scope of a trusted care task. In the LTQA notification experiment, it verifies the app's signed assertion about the sender and a separate signed patient grant. It intersects the grant with the care scenario and patient restrictions before ranking facts on a patient-local hypergraph. The answering model receives only the selected excerpts. The incoming question can ask for further records, but only a verified grant can permit their release.

## Trusted notification, untrusted question

![A Jingdong ophthalmology notification and two responses to a request for a patient's complete medical history](docs/figures/jingdong-notification.png)

In this illustrated scenario, the **simulated Jingdong app notification identifies the sender as an ophthalmologist**, and the trusted task is an ophthalmology follow-up. The sender's message claims that a multidisciplinary assessment requires the patient's complete history. That claim comes from the message body and carries no authority to change the task or enlarge the patient's consent.

For this follow-up, recent visual acuity and intraocular pressure results may enter the answering context if the patient's permissions cover them. The left response would disclose psychiatric, reproductive and family genetic history on the sender's word. The right response withholds those records. The executable LTQA cases use synthetic app identities and signed test credentials. [Notification experiment](docs/trusted-notifications.md)

## Data

The main paper reports results on 12,320 test instances. We later expanded the study dataset with de-identified real-world samples.

The table summarizes **26,218 test requests** across 846 held-out subjects in the reconstructed benchmark. LTQA includes 4,586 scenario-only requests and 2,293 matched pairs of signed-notification cases. Ranker training and validation use separate subjects.

| Source | Test requests | Test subjects | Ranker training | Ranker validation |
| --- | ---: | ---: | ---: | ---: |
| Asclepius | 10,400 | 800 | 696 | 113 |
| NHS Synthetic | 6,646 | 32 | 262 | 88 |
| LTQA | 9,172 | 14 | 208 | 80 |
| **Total** | **26,218** | **846** | **1,166** | **281** |

Medical records and test questions are in `data/benchmark/`. Reference answers, protected values and ranking labels are in `data/private/` for training or offline scoring. Source attribution, schemas, reconstruction commands and dataset-specific terms are in the [data guide](DATA.md).

## Experiments

| Task | Entry point |
| --- | --- |
| Rebuild the clinical data from the pinned sources | `scripts/build_data.py` |
| Propose or import reviewed patient-local evidence associations | `python -m perm2ask.memory` |
| Train a ranker or select evidence | `python -m perm2ask.ranker` |
| Run PerM²ask, Direct, prompt controls or retrieval baselines | `python -m perm2ask.generation` |
| Score single- and three-turn conversations | `python -m perm2ask.scoring` |
| Recompute the auxiliary image-type accuracies from saved selections | `python scripts/replay_image_types.py` |

The full hypergraph ranker and the pairwise GNN have separate [checkpoints](checkpoints/manifest.json). The `without_hgnn` variant uses the authorized hypergraph without neural message passing. HoneyBee uses its upstream dynamic-partition index. Setup is described in the [baseline guide](docs/honeybee.md).

<details>
<summary>Evaluation commands (four test requests)</summary>

Run from the repository root with Python 3.10+ and Ollama running locally.

```shell
python -m pip install .
python -m perm2ask.fetch_bge
python -m perm2ask.fetch_bge --tokenizer Qwen/Qwen2.5-1.5B-Instruct
python -m perm2ask.ranker select --method full --checkpoint checkpoints/full.pt --tokenizer .cache/perm2ask/tokenizers/Qwen2.5-1.5B-Instruct --limit 4 --out outputs/contexts.jsonl
ollama pull qwen2.5:1.5b
python -m perm2ask.generation --model qwen2.5:1.5b --method full --contexts outputs/contexts.jsonl --tokenizer .cache/perm2ask/tokenizers/Qwen2.5-1.5B-Instruct/tokenizer.json --turns 1 --condition normal --seed 20260812 --limit 4 --out outputs/predictions.jsonl
python -m perm2ask.scoring --predictions outputs/predictions.jsonl --subset --out outputs/scores
```

For a three-turn Medical Disclosure Laundering (MDL) run, install the attacker with `ollama pull qwen3.5:0.8b`. Use `--turns 3 --condition attack` and a fresh predictions file.

</details>
