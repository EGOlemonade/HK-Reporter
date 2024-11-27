import json
import re
import ftfy
from collections import namedtuple
from rouge_score import rouge_scorer
from bleurt import score
import asyncio
import hashlib
from pathlib import Path
from kani import Kani
from kani.engines.openai import OpenAIEngine

AccuracyResult = namedtuple("AccuracyResult", "found score missing")


class Scorer:
    def __init__(self, results, key, questions=None, openai_api_key=None):
        self.key = key
        self.questions = questions or self.load_questions()
        self.questions_by_id = {q["id"]: q for q in self.questions}
        self.results = results
        self.results_by_id = {r["id"]: r for r in self.results}
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.bleurt = score.BleurtScorer("BLEURT-20")  # 加载 BLEURT 检查点
        self.eval_len = len(self.questions)

        # GPT Engine for factuality judgment
        self.openai_api_key = openai_api_key
        self.engine = OpenAIEngine(api_key=openai_api_key)
        self.factuality_system = "You are comparing a submitted answer to an expert answer on a given question."

    @staticmethod
    def normalize(text):
        text = ftfy.fix_text(text)
        return re.sub(r'[^\w\s]', '', text.lower().strip())

    def get_qa_pairs(self):
        for q in self.questions:
            a = self.results_by_id.get(q["id"])
            yield q, a

    def answer_in_text(self, reference, candidate: str) -> AccuracyResult:
        norm_ans = self.normalize(reference)
        norm_cand = self.normalize(candidate)
        missing = []
        
        if not re.search(rf"\b{re.escape(norm_ans)}\b", norm_cand):
            missing.append(norm_ans)
            return AccuracyResult(found=False, score=0, missing=missing)

        return AccuracyResult(found=True, score=1, missing=missing)

    def score_accuracy(self):
        """计算准确率：包括严格匹配和宽松匹配。"""
        accs = []  # 存储每个问题的匹配分数
        n_perfect = 0  # 完全匹配的数量

        for q, a in self.get_qa_pairs():
            if a is None:  # 如果生成答案为空，记为 0 分
                accs.append(0)
                continue

            # 检查答案匹配情况
            result = self.answer_in_text(q["answer"], a["answer"])
            accs.append(result.score)  # 记录匹配分数

            # 如果完全匹配，计数增加
            if result.found:
                n_perfect += 1
                
        assert len(accs) == self.eval_len, "评估结果数量与问题集长度不一致！"
        # 计算平均准确率和完美匹配率
        avg_acc = sum(accs) / self.eval_len
        pct_perfect = n_perfect / self.eval_len
        # 输出
        print(f"AVG ACC: {avg_acc:.4f}")
        print(f"PCT PFT: {pct_perfect:.4f}")
        return {"acc": avg_acc, "perfect": pct_perfect}

    def score_rouge(self):
        """计算 ROUGE 分数"""
        rouge_types = ["rouge1", "rouge2", "rougeL"]
        scores = {t: [] for t in rouge_types}
        for q, a in self.get_qa_pairs():
            if a is None:
                for score in scores.values():
                    score.append(rouge_scorer.Score(0, 0, 0))
                continue
            results = self.rouge.score(self.normalize(q["answer"]), self.normalize(a["answer"]))
            for k, v in results.items():
                scores[k].append(v)

        assert all(len(v) == self.eval_len for v in scores.values())
        print("=== ROUGE ===")
        out = {}
        for k, v in scores.items():
            print(f"--- {k} ---")
            avg_precision = sum(s.precision for s in v) / self.eval_len
            avg_recall = sum(s.recall for s in v) / self.eval_len
            avg_fscore = sum(s.fmeasure for s in v) / self.eval_len
            print(f"precision: {avg_precision}")
            print(f"recall: {avg_recall}")
            print(f"fscore: {avg_fscore}")
            out[k] = {"precision": avg_precision, "recall": avg_recall, "fscore": avg_fscore}
        print()
        return out

    def score_bleurt(self):
        """计算 BLEURT 分数"""
        references = []
        candidates = []
        for q, a in self.get_qa_pairs():
            if a is None:
                candidates.append("")
            else:
                candidates.append(self.normalize(a["answer"]))
            references.append(self.normalize(q["answer"]))

        scores = self.bleurt.score(references=references, candidates=candidates)
        assert len(scores) == self.eval_len
        avg_score = sum(scores) / self.eval_len
        print(f"BLEURT: {avg_score}")
        return avg_score

    async def score_gpt_factuality(self, question, reference, answer):
        """使用 GPT 判断答案的事实性"""
        prompt = self.factuality_prompt(question["question"], reference, answer)
        ai = Kani(self.engine, system_prompt=self.factuality_system)
        response = await ai.chat_round_str(prompt)
        return response.strip()

    def factuality_prompt(self,question, reference, answer):
        return (
            f"[BEGIN DATA]\n************\n[Question]: {question}\n************\n[Expert]:"
            f" {reference}\n************\n[Submission]: {answer}\n************\n[END DATA]\n\nCompare the factual content"
            " of the submitted answer with the expert answer. Ignore any differences in style, grammar, or"
            " punctuation.\nThe submitted answer may either be a subset or superset of the expert answer, or it may"
            " conflict with it. Determine which case applies. First, write out in a step by step manner your reasoning"
            " about the factual content to be sure that your conclusion is correct. Avoid simply stating the correct"
            ' answers at the outset. Then print only the single character "A", "B", "C", "D", "E", or "F" (without quotes'
            " or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the letter"
            " again by itself on a new line.\n(A) The submitted answer is a subset of the expert answer and is fully"
            " consistent with it.\n(B) The submitted answer is a superset of the expert answer and is fully consistent"
            " with it.\n(C) The submitted answer contains all the same details as the expert answer.\n(D) There is a"
            " disagreement between the submitted answer and the expert answer.\n(E) The answers differ, but these"
            " differences don't matter from the perspective of factuality.\n(F) The submitted answer does not answer the"
            " question or is otherwise invalid."
            "Please respond with only one of the options above, i.e., A, B, C, D, E, or F."
            "Do not add any extra text, explanations, or reasoning"
        )

    async def score(self):
        """计算所有评分"""
        all_scores = {}
        for question in self.questions:
            result = self.results_by_id.get(question["id"])
            if result:
                # 计算指标
                question_score = self.score_accuracy()
                rouge_score = self.score_rouge()  # 计算ROUGE分数
                bleurt_score = self.score_bleurt()  # 计算BLEURT分数
                gpt_score = await self.score_gpt_factuality(question, question["answer"],
                                                            result["answer"]) 
                all_scores[question["id"]] = {**question_score, **rouge_score, **{"bleurt": bleurt_score},
                                              **{"gpt": gpt_score}}

                # 输出
                print(f"\nQuestion ID: {question['id']}")
                for metric, value in all_scores[question["id"]].items():
                    print(f"{metric.upper()}: {value}")

        with open(f"score-{self.key}.json", "w", encoding='utf-8') as f:
            json.dump(all_scores, f, indent=2)


# 载入问题和答案数据
with open("testbasement.json", "r", encoding="utf-8") as qf:
    questions = json.load(qf)  # 参考答案数据集

with open("generatedresult.json", "r", encoding="utf-8") as rf:
    results = json.load(rf)  # 生成结果数据集

api_key=''
scorer = Scorer(results=results, key="custom_eval", questions=questions, openai_api_key=api_key)
asyncio.run(scorer.score())
