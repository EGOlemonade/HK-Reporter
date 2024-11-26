import json
import re
import ftfy
from collections import namedtuple
from rouge_score import rouge_scorer
from bleurt import score
import asyncio
from pathlib import Path

AccuracyResult = namedtuple("AccuracyResult", "found score missing")


class Scorer:
    def __init__(self, results, key, questions=None):
        self.key = key
        self.questions = questions or self.load_questions()
        self.questions_by_id = {q["id"]: q for q in self.questions}
        self.results = results
        self.results_by_id = {r["id"]: r for r in self.results}
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.bleurt = score.BleurtScorer("BLEURT-20")  # 加载 BLEURT 检查点
        self.eval_len = len(self.questions)

    @staticmethod
    def normalize(text):

        text = ftfy.fix_text(text)
        return re.sub(r'[^\w\s]', '', text.lower().strip())

    def get_qa_pairs(self):

        for q in self.questions:
            a = self.results_by_id.get(q["id"])
            yield q, a

    def answer_in_text(self, reference, candidate):

        norm_ans = self.normalize(reference)
        norm_cand = self.normalize(candidate)
        missing = []

        if not re.search(rf"\b{re.escape(norm_ans)}\b", norm_cand):
            missing.append(norm_ans)
            return AccuracyResult(found=False, score=0, missing=missing)

        return AccuracyResult(found=True, score=1, missing=missing)

    def score_question(self, question, result):

        # Strict,Loose
        acc_result = self.answer_in_text(question["answer"], result["answer"])

        # ROUGE
        rouge_result = self.rouge.score(self.normalize(question["answer"]), self.normalize(result["answer"]))

        # BLEURT
        bleurt_score = self.bleurt.score(references=[question["answer"]], candidates=[result["answer"]])[0]

        return {
            "strict_acc": acc_result.score,
            "loose_acc": acc_result.score, 
            "rouge1": rouge_result["rouge1"].fmeasure,
            "rouge2": rouge_result["rouge2"].fmeasure,
            "rougeL": rouge_result["rougeL"].fmeasure,
            "bleurt": bleurt_score
        }

    async def score(self):
        all_scores = {}
        for question in self.questions:
            result = self.results_by_id.get(question["id"])
            if result:
                question_score = self.score_question(question, result)
                all_scores[question["id"]] = question_score
                # 输出
                print(f"\nQuestion ID: {question['id']}")
                for metric, value in question_score.items():
                    print(f"{metric.upper()}: {value}")

        with open(f"score-{self.key}.json", "w", encoding='utf-8') as f:
            json.dump(all_scores, f, indent=2)


if __name__ == "__main__":
    with open("testbasement.json", "r", encoding="utf-8") as qf:
        questions = json.load(qf)  # 参考数据集

    with open("generatedresult.json", "r", encoding="utf-8") as rf:
        results = json.load(rf)  # 生成结果数据集

    scorer = Scorer(results=results, key="custom_eval", questions=questions)
    asyncio.run(scorer.score())
