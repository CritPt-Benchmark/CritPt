# Evaluation Pipeline for the CritPt Benchmark

## Probing the Critical Point (CritPt) of AI Reasoning: a Frontier Physics Research Benchmark
|[🌐 Website](https://critpt.com) | [🤗 Dataset](https://huggingface.co/datasets/CritPt-Benchmark/CritPt) | [📖 Tech report](https://arxiv.org/abs/2509.26574) |

CritPt (Complex Research using Integrated Thinking – Physics Test; reads as "critical point") is the first benchmark designed to test LLMs on unpublished, research-level reasoning tasks that broadly covers modern physics research areas, including condensed matter, quantum physics, atomic, molecular & optical physics, astrophysics, statistical physics, nuclear physics, high energy physics, mathematical physics, fluid dynamics, nonlinear dynamics and biophysics.

It currently includes 71 challenges and 190 checkpoints, crafted by a team of 50+ active physics researchers from 30+ leading institutions worldwide, including senior Ph.D. students, postdocs, professors and research scientists. 

CritPt is an actively maintained benchmark. Current deployed version is CritPt v2.

## Updates
**Nov 21, 2025**: CritPt made its official debut on [Artificial Analysis](https://artificialanalysis.ai/evaluations/critpt-v2)! 

**Nov 21, 2025**: The evaluation pipeline and challenge dataset are now publicly available! To evaluate your own models on CritPt, please follow the instructions provided on this page. 
                  An example challenge can be found [here](https://critpt.com/example.html). See 70 test challenges in [this folder](https://github.com/CritPt-Benchmark/CritPt/tree/main/data) or on [Hugging Face](https://huggingface.co/datasets/CritPt-Benchmark/CritPt).

## Leaderboard (v2)

| Model                    | Challenge Accuracy¹ |
| ------------------------ | ------------------- |
| gpt-6-astra              | 83.1                |
| gpt-6-1-sol              | 82.0                |
| claude-opus-5-5          | 77.7                |
| gpt-5-6-sol              | 74.9                |
| muse-wonder-full         | 73.1                |
| gemini-4-argon           | 72.9                |
| claude-fable-5-1         | 72.0                |
| muse-wonder              | 69.7                |
| grok-4-7                 | 50.6                |
| deepseek-v4-1-flash-1017 | 50.0                |
| gpt-5                    | 28.9                |

¹ We use average accuracy over 5 runs × 70 test challenges as our primary performance metric.  


*To evaluate your own LMs on CritPt challenges, follow the steps below*

## 0. Setup
- Clone this repository:
```
git clone https://github.com/CritPt-Benchmark/CritPt.git && cd ./CritPt
```
- Install required packages:
```
pip install -r requirements.txt
```
- Setup model API keys in .env as environment variables:
```
ANTHROPIC_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
GOOGLE_API_KEY=
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
TOGETHER_API_KEY=
```

## 1. Generation of model responses
To generate responses using an API model, run the following command:
```
python -m critpt generate generate model=google/gemini-2.5-flash task_config=.config/multiturn_without_answer_without_tool.json
```
This command will launch the generation process with the specified model and task configuration.

### Advanced Generation with Model-specific Configs
Below are the model configurations used in our evaluation. You can modify the parameters to customize the setup.

Claude & Gemini

```
python -m critpt generate epochs=5 model=google/gemini-2.5-pro,anthropic/claude-opus-4-20250514 reasoning_tokens=27000
```
DeepSeek

```
python -m critpt generate epochs=5 model=openai-api/deepseek/deepseek-reasoner max_connections=256
```
GPT-5, o3 and o4-mini

```
python -m critpt generate epochs=5 model=openai/gpt-5,openai/o3,openai/o4-mini reasoning_effort=high max_connections=150
```
GPT-5 (minimal)

```
python -m critpt generate epochs=5 model=openai/gpt-5 reasoning_effort=minimal model_label=nothinking
```
Non-reasoning models

```
python -m critpt generate epochs=5 model=together/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8,openai/gpt-4o,google/gemini-2.5-flash
```
Tool Use (GPT-5 Only)

```
python -m critpt generate epochs=5 model=openai/gpt-5 reasoning_effort=high model_label=python+search max_connections=150 task_config=.config/multiturn_without_answer_with_tool_web.json

python -m critpt generate epochs=5 model=openai/gpt-5 reasoning_effort=high model_label=python max_connections=150 task_config=.config/multiturn_without_answer_with_tool.json
```

## 2. Grading
Model responses should be submitted to our grading server to receive a score. The server is operated in partnership with Artificial Analysis (more API details [here](https://artificialanalysis.ai/documentation#critpt-api)).

To ensure data integrity and prevent leakage, our grading server only accepts complete batches containing responses to all 70 problems. Any submission with missing or unmatched problem IDs will be automatically rejected.
Each account is limited to **10 submissions within a 24-hour window**.

To initiate the grading process, run the following command:

```
python ../evaluate_all_results.py
```

# Contact
For inquiries or collbarations, please contact minhui.zhu@anl.gov, mtian8@illinois.edu.

# Citation
```
@article{zhu2025probing,
  title={Probing the Critical Point (CritPt) of AI Reasoning: a Frontier Physics Research Benchmark},
  author={Zhu, Minhui and Tian, Minyang and Yang, Xiaocheng and Zhou, Tianci and Zhu, Penghao and Chertkov, Eli and Liu, Shengyan and Du, Yufeng and Yuan, Lifan and Ji, Ziming and others},
  journal={arXiv preprint arXiv:2509.26574},
  year={2025}
}
```
