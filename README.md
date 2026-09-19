# RoboICL

**Embodied In-Context Learning for GPT-6 Astra**

[Project page](https://mosi-ai.github.io/RoboICL-GPT6.github.io/)

RoboICL studies how a general-purpose multimodal model can adapt to bimanual robot tasks at inference time from a small number of executable demonstrations. GPT-6 Astra directly generates low-level Cartesian action sequences through a single `Act` interface, while a deterministic harness validates and executes them.

The central design is a shared interaction grammar for both demonstrations (`TRAIN`) and deployment (`LIVE`):

```text
observation → Act call → execution feedback → next observation
```

Bounded anchored LIVE memory preserves temporally distributed, full-resolution interaction chunks while explicitly marking omitted intervals.

## Results at a glance

- On five fixed layouts, three demonstrations raise the mean score from **0.34 to 0.88** for *Put bottles in a bin*.
- On the same protocol, three demonstrations raise the mean score from **0.04 to 0.82** for *Build Tower*.
- In a separate 50-layout *Build Tower* sweep, RoboICL 3-shot reaches **59.80**, compared with **16.40** for GPT-6 Astra Direct.

## Released request artifact

[`put_bottles_l3_2shot_final_gpt6_request.json`](put_bottles_l3_2shot_final_gpt6_request.json) is the fully serialized final model request from a 2-shot *Put bottles in a bin* rollout at control step 687. It is the request visualized in Figure 3 of the project page.

The JSON contains:

- the task and controller instructions;
- the `Act` tool schema;
- two fixed TRAIN reference trajectories;
- bounded anchored LIVE interaction history;
- robot proprioception and 30 embedded three-view RGB observations;
- 15 preceding `Act` calls and their execution feedback.

The RGB observations are embedded as JPEG data URLs, so the file can be inspected as a self-contained artifact without separate image assets. Reward values, success labels, evaluator metrics, privileged object state, task code, and layout metadata are not included in the action-generation request.

Validate the JSON locally with:

```bash
python -m json.tool put_bottles_l3_2shot_final_gpt6_request.json > /dev/null
```

SHA-256:

```text
7041a9c725d98a47f279dd721188573bceabb33a8b5dfe4072888ae370b27fee
```

More code and evaluation artifacts will be added to this repository.
