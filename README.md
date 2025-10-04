# RADAR

RADAR is a speculative sampling method improved over EAGLE-3 by RL-based dynamic generated draft trees, achieving significant performance on speedup ratio.

## RADAR Performance

| Model  | Method  | MT-bench  | GSM8K     | Alpaca    | MBPP      |
| ------ | ------- | --------- | --------- | --------- | --------- |
| L3 8B  | Eagle-2 | 2.56x     | 3.43x     | 2.89x     | 3.29x     |
|        | Eagle-3 | 3.08x     | 4.68x     | 3.86x     | 4.2x      |
|        | RADAR | **3.41x** | **4.82x** | **4.04x** | **4.44x** |
| V 13B  | Eagle-2 | 2.89x     | 3.18x     | 2.83x     | 3.56x     |
|        | Eagle-3 | 3.74x     | 4.24x     | 3.5x      | 4.55x     |
|        | RADAR | **4.05x** | **4.36x** | **3.84x** | **4.75x** |
| DSL 8B | Eagle-3 | 3.42x     | 4.39x     | 3.08x     | 3.71x     |
|        | RADAR | **3.86x** | **4.71x** | **3.17x** | **3.99x** |

- RADAR achieves a speedup of 3.17x–4.82x over the auto-regressive decoding baseline.
- Across all tested datasets and models, RADAR consistently outperforms EAGLE-3, achieving higher speedup ratios.


| Model  | Method  | MT-bench | GSM8K | Alpaca | MBPP |
| ------ | ------- | -------- | ----- | ------ | ---- |
| L3 8B  | Eagle-2 | 3.37     | 3.88  | 4.08   | 4.55 |
|        | Eagle-3 | 4.55     | 5.5   | 5.62   | 6.12 |
|        | RADAR | 4.48     | 5.32  | 5.51   | 6    |
| V 13B  | Eagle-2 | 4.38     | 4.46  | 4.09   | 4.93 |
|        | Eagle-3 | 5.69     | 5.94  | 5.51   | 6.44 |
|        | RADAR | 5.67     | 5.87  | 5.48   | 6.42 |
| DSL 8B | Eagle-3 | 4.88     | 6.39  | 4.49   | 5.35 |
|        | RADAR | 4.85     | 6.33  | 4.44   | 5.31 |

- RADAR maintains high acceptance lengths compared to EAGLE-3.

| Model  | MT-bench | GSM8K | Alpaca | MBPP |
| ------ | -------- | ----- | ------ | ---- |
| L3 8B  | 5.25     | 6.19  | 6.2    | 6.6  |
| V 13B  | 6.88     | 7.26  | 6.83   | 7.26 |
| DSL 8B | 6.1      | 7.2   | 5.85   | 6.47 |

- RADAR dynamically decides the calls to the draft model, resulting in a reduction in the number of calls to the draft model compared to EAGLE-3, at an average of 18.7\%.

## Files
In the `eagle/evaluations` directory, `gen_ea_we_answer_{model}.py` is the **test script** for RADAR on target model, while the rest are test scripts for EAGLE-3 and baseline.  
The `eagle/data` directory contains the datasets used by EAGLE and RADAR, where `eagle/data/generate/ge_data_{model}_rb.py` is the script for generating **the offline reinforcement learning dataset** by computing the acceptance length distribution.  
The **training script** for RADAR is located at `eagle/train/train_radar.py`.  
The **main code** for RADAR is integrated into the EAGLE framework and can be found in the `eagle/model` folder.
The **weights** for RADAR's prediction model can be found in the `weights` folder.

## Acknowledgements

This project has been influenced by many excellent projects in the LLM community, such as [EAGLE-3](https://github.com/SafeAILab/EAGLE) and others.
