# QA Evaluation （Beta）

# requirements
Python 3.8+
TensorFlow
BLEURT
rouge_score
ftfy

# Usage
testbasement.json：包含问题及其参考答案。
generated_result.json：包含生成的答案。
testbasement.json Example：
```
[
    {
        "id": "q1",
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    },
    {
        "id": "q2",
        "question": "Who is the president of the United States?",
        "answer": "The president of the United States is Joe Biden."
    },
    {
        "id": "q3",
        "question": "What is the largest planet in our solar system?",
        "answer": "The largest planet in our solar system is Jupiter."
    }
]
```
generated_result.json Example：
```
[
    {
        "id": "q1",
        "answer": "Paris is the capital of France."
    },
    {
        "id": "q2",
        "answer": "Joe Biden is the president of the United States."
    },
    {
        "id": "q3",
        "answer": "Jupiter is the largest planet in our solar system."
    }
]

```
# Running the Evaluation
```
python eval.py
```
# Example Output
Here’s an example of the output you will see for each question:
```
Question ID: q1
STRICT_ACC: 1
LOOSE_ACC: 1
ROUGE1: 1.0
ROUGE2: 0.8333333333333334
ROUGEL: 0.8148148148148148
BLEURT: 0.8939730525016785

Question ID: q2
STRICT_ACC: 1
LOOSE_ACC: 1
ROUGE1: 1.0
ROUGE2: 0.75
ROUGEL: 0.6666666666666666
BLEURT: 0.9096779227256775

Question ID: q3
STRICT_ACC: 1
LOOSE_ACC: 1
ROUGE1: 1.0
ROUGE2: 0.75
ROUGEL: 0.7777777777777778
BLEURT: 0.9145138263702393
```
# License
Provided under the MIT License.

