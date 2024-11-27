# QA Evaluation （Beta）

# requirements
```
Python 3.8+

TensorFlow

BLEURT

rouge_score

ftfy
```
# Methods

此工具的评分方法参考了论文中的评估公式，具体包括 **严格匹配**（Strict）和 **宽松匹配**（Loose）的计算方法。以下是相关的公式：

### **Loose 匹配**（Loose Match）：
对于参考答案 `R` 和生成答案 `G`，**宽松匹配**被定义为：

$$
Loose(R, G) = \frac{|R \cap G|}{|R|}
$$

- `|R|`：参考答案中的词汇数量。
- `|R \cap G|`：参考答案和生成答案的交集词汇数。

### **Strict 匹配**（Strict Match）：
**严格匹配**则检查参考答案和生成答案是否完全一致：

$$
Strict(R, G) = \mathbb{1}[Loose(R, G) = 1]
$$

- 如果宽松匹配为 1，则严格匹配得分为 1，否则为 0。

---





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

```
AVG ACC: 0.5000
PCT PFT: 0.5000
=== ROUGE ===
--- rouge1 ---
precision: 1.0
recall: 1.0
fscore: 1.0
--- rouge2 ---
precision: 0.875
recall: 0.875
fscore: 0.875
--- rougeL ---
precision: 0.861111111111111
recall: 0.861111111111111
fscore: 0.861111111111111

BLEURT: 0.8911631256341934

Question ID: q1
ACC: 0.5
PERFECT: 0.5
ROUGE1: {'precision': 1.0, 'recall': 1.0, 'fscore': 1.0}
ROUGE2: {'precision': 0.875, 'recall': 0.875, 'fscore': 0.875}
ROUGEL: {'precision': 0.861111111111111, 'recall': 0.861111111111111, 'fscore': 0.861111111111111}
BLEURT: 0.8911631256341934
GPT: A  
```

