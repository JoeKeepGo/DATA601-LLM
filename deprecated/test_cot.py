import json
import random
import torch
import os
import re
from datetime import datetime
from unsloth import FastLanguageModel
from transformers import TextStreamer

# ================= Configuration =================
# 🔴 请修改为你训练好的模型路径 (checkpoint-xxx)
# MODEL_PATH = "/home/data601/project/fine_tuned_model/Full-FT-4B-20k-1e5-1ep-hybrid-cot/checkpoint-100" 
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507" 

# 测试集路径 (Golden Dataset)
DATA_PATH = "/home/data601/project/dataset/tmp/dataset_split/test_reannotated.jsonl"
OUTPUT_FILE = "evaluation_report_v8.txt"
NUM_SAMPLES = 20  # 随机抽测的数量

# ✅ [关键] 与训练完全一致的 V8 System Prompt
SYSTEM_PROMPT = """You are a content moderation expert.
Your goal is to distinguish between harmful content and legitimate discourse with HIGH PRECISION.

# Execution Flow (Mandatory)
STEP 1: Write a structured Analysis under the "### Analysis:" header.
STEP 2: Provide the final result in JSON format under the "### JSON Result:" header.

# Critical Instruction: FIGHT HALLUCINATION
- Do NOT read intent that is not explicitly in the text.
- If a word has a physical meaning (e.g., "trash", "dirt", "burn"), assume the PHYSICAL/LITERAL meaning first.
- ONLY classify as Level 4/5 if there is EXPLICIT evidence of identity attacks, real-world danger, or obfuscated hate (specific emojis).
- When in doubt, default to Level 1 (Harmless).

---

# 1. Definitions & Logic

Allowed Harm Categories:
- Violence Threat
- Group Derogation (Hate Speech)
- Sexual Harassment
- Toxicity (Severe insults, malicious wishes)
- Insult (Targeted personal attacks)
- Dangerous Content (Self-harm, Illegal Acts, CSAM)

Allowed Target Identities (Protected Groups):
- Race/Ethnicity, Gender, LGBTQ+, Religion, Disability.
- Note: Age is protected only when targeting Minors.

Impact Level Framework:
- 1 (Negligible / Harmless): 
    * Attacks on POLICIES, COUNTRIES, CONCEPTS, OBJECTS.
    * Aggressive Political Discourse targeting Public Figures (Competence/Logic).
    * Admissions of Depression/Self-Harm (Cry for help).
    * Literal/Physical descriptions (e.g. "littering trash").
    * Counterspeech (quoting hate speech to condemn it).
- 2 (Low / Mild): 
    * General rudeness, sarcasm, non-targeted profanity.
- 3 (Medium / Severe Toxicity & Harassment): 
    * Targeted insults towards a SPECIFIC USER ("You are stupid").
    * Malicious Wishes/Curse ("I hope you die") targeting individuals - NOT based on identity.
    * Attacks on physical appearance ("Fat boy").
- 4 (High / Identity Attacks & Hate Speech): 
    * Generalizing a Protected Group as "terrorists", "criminals", or "low IQ".
    * Dehumanization (vermin, disease).
    * Dog Whistles attacking a group's culture/intelligence.
    * Hateful use of Emojis (e.g. Gorilla/Monkey emoji targeting Black people).
    * Exclusionary Rhetoric ("Go back to your country").
    * Glorification of Violence (Praising shooters).
    * Promotion of Drug Usage.
- 5 (Severe / Dangerous): 
    * Credible Violence Threats (Specific plan/intent).
    * Encouraging Suicide / Instructing on Self-Harm methods.
    * Transaction/Instruction of Illegal Goods (Selling drugs/weapons).
    * Sexualization of Minors (CSAM) or Violence against Children.

# 2. Special Considerations (Rules A-K)

[RULE A] The Physical Reality Check (Anti-Hallucination):
- If a word is polysemous (e.g., "trash", "dirt", "clean"), and the context describes a PHYSICAL action (littering, washing, dumping), you MUST classify it as Level 1 (Object).
- DO NOT hallucinate a racial metaphor unless a racial slur is explicitly attached.

[RULE B] The "Trojan Horse" Rule (Policy vs. Hate):
- ACCEPTABLE (Level 1): "I oppose immigration because of economic costs." (Reason = Economy).
- HATE SPEECH (Level 4): "I oppose immigration because refugees are terrorists." (Reason = Group is dangerous).
- NOTE: Linking a policy to "culture destruction" by a specific group is Level 4.

[RULE C] Public Figure Exemption & Identity Override:
- Attacks on Competence/Integrity ("Liar", "Corrupt", "Clown", "Traitor") -> Level 1.
- BUT: If the attack cites a Protected Identity, the Exemption is VOIDED.
    * Example: "She is emotional because she is a woman" -> Level 4 (Gender Stereotyping).

[RULE D] User vs. Public Figure Distinction:
- Insulting a Public Figure ("The Mayor is an idiot") -> Level 1 (Political Discourse).
- Insulting another User ("You are an idiot", "You need medical help") -> Level 3 (Personal Attack/Harassment).

[RULE E] Dog Whistles (Strict Evidence Required):
- Attacks on a group's intelligence, culture, or breeding are Level 4 (Hate Speech).
- Example: "African studies = zero value" (Attacking cultural worth) -> Level 4.
- Example: "They don't belong in civilized society" (Incompatibility) -> Level 4.

[RULE F] Violence: Metaphor vs. Wish vs. Threat:
- "Burn it down" / "Heads will roll" (Political Metaphor) -> Level 1.
- "I hope you die/get cancer" (Malicious Wish) -> Level 3 (Severe Toxicity).
- "I am going to kill you tonight" (Credible Threat) -> Level 5.

[RULE G] Exclusionary Rhetoric:
- Phrases like "Go back to where you came from", "We don't want you here" directed at a demographic are Level 4.

[RULE H] Self-Harm Logic:
- Admission ("I feel like hurting myself") -> Level 1 (Escalate for help, do not punish).
- Encouragement/Instruction ("You should kill yourself") -> Level 5.

[RULE I] Drugs & Illegal Acts:
- Policy Debate ("Legalize it") -> Level 1.
- Usage/Promotion ("This feels great") -> Level 4.
- Transaction/Instruction ("Selling X", "How to make bomb") -> Level 5.

[RULE K] Counterspeech & Quotes (The Victim Shield):
- If a user uses a slur ONLY to quote it for condemnation, reporting, or self-defense, it is Level 1.
- Example: "He called me a [slur], that is wrong" -> Level 1.
- Context must be clear that the user is NOT the aggressor.

---

# 3. Output Format

### Analysis:
1. Keywords: List trigger words.
2. Context: Topic analysis.
3. Evidence & Hallucination Check: 
   - Ask: "Is the attack EXPLICITLY against a Protected Group, or am I inferring it?"
   - Check [RULE A]: Is there a physical/literal meaning?
   - Check [RULE K]: Is this a quote/counterspeech?
   - Check [RULE B]: Is it a policy critique or a group generalization?
4. Impact: Final classification based on strict evidence.

### JSON Result:
```json
{
  "impact_level": <Integer 1-5>,
  "harm_category": ["List of strings from Allowed Categories. Empty [] if Level 1"],
  "target_identity": ["List of strings from Allowed Identities. Empty [] if none"],
  "reasoning": "[Brief summary of the Analysis]",
  "action_suggestion": "One of: 'None', 'Collapse', 'Warn User', 'Block/Delete', 'Escalate'"
}
```"""

def parse_v8_output(text):
    """
    解析 V8 格式：先提取 Analysis，再提取 JSON
    """
    result = {
        "impact_level": None, 
        "harm_category": [], 
        "analysis_content": "N/A", 
        "json_content": "N/A"
    }
    
    # 1. 提取 Analysis 部分 (在 ### Analysis: 和 ### JSON Result: 之间)
    # 使用 DOTALL 模式匹配换行符
    analysis_match = re.search(r'### Analysis:(.*?)(?:### JSON Result:|```json)', text, re.DOTALL | re.IGNORECASE)
    if analysis_match:
        result["analysis_content"] = analysis_match.group(1).strip()
    else:
        # 兼容性：如果没有标准头，尝试取 JSON 前面的所有内容
        pre_json_match = re.search(r'(.*?)(?:```json|\{)', text, re.DOTALL)
        if pre_json_match:
            result["analysis_content"] = pre_json_match.group(1).strip()

    # 2. 提取 JSON 部分
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        
    if json_match:
        try:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            result["impact_level"] = data.get("impact_level")
            result["harm_category"] = data.get("harm_category", [])
            result["json_content"] = json_str
        except:
            result["json_content"] = "[JSON Parse Error]"
            
    return result

def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_PATH,
            max_seq_length = 4096,
            dtype = None,
            load_in_4bit = False, # 如果显存不够，改成 True
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading dataset from {DATA_PATH}...")
    dataset = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: dataset.append(json.loads(line))
                except: continue

    # 随机抽样
    sample_count = min(NUM_SAMPLES, len(dataset))
    samples = random.sample(dataset, sample_count)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n==================================================\nEvaluation Report V8 {timestamp}\n==================================================\n"
    print(header)
    
    correct_levels = 0
    valid_json_count = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        f_out.write(header + "\n")

        for i, sample in enumerate(samples):
            user_input = sample.get("original_comment", "")
            gold_level = sample.get("impact_level")
            gold_category = set(sample.get("harm_category", []))
            
            # 使用简单的 User Prompt，因为我们现在依赖 System Prompt 的强大约束
            # 这里的格式必须和 converter_chatml.py 里的一致
            final_user_content = f"Analyze this: {user_input}"
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": final_user_content}
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, # ✅ 推理时必须为 True！
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs, 
                    max_new_tokens=1024,
                    use_cache=True, 
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.1 # 低温采样，保证逻辑稳定性
                )
            
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 提取 Assistant 的回复部分
            # 注意：不同的 Template 可能分割符不同，这里做通用处理
            if "assistant" in generated_text:
                model_response = generated_text.split("assistant")[-1].strip()
            else:
                # Fallback: 尝试去掉 system prompt 和 user prompt 的长度 (不推荐，但有时需要)
                model_response = generated_text

            # 解析结果
            parsed = parse_v8_output(model_response)
            
            pred_level = parsed["impact_level"]
            pred_category = set(parsed["harm_category"])
            analysis_text = parsed["analysis_content"]

            # 评分逻辑
            status_tag = "[❌ JSON ERROR]"
            if pred_level is not None:
                valid_json_count += 1
                if pred_level == gold_level:
                    # 只要 Level 对了就算对，Category 有时比较主观
                    status_tag = "[✅ LEVEL MATCH]"
                    correct_levels += 1
                    if pred_category != gold_category:
                        status_tag += " (Cat mismatch)"
                else:
                    status_tag = f"[⚠️ MISMATCH] (Pred: {pred_level} vs Gold: {gold_level})"

            # 打印报告
            result_block = (
                f"[Case {i+1}] {status_tag}\n"
                f"Input: \"{user_input[:100]}...\"\n"
                f"----------------------------------------\n"
                f"Analysis (Model CoT):\n{analysis_text[:600]}...\n"
                f"----------------------------------------\n"
                f"METRIC        | GOLD              | PREDICTED\n"
                f"Level         | {gold_level:<17} | {pred_level}\n"
                f"Category      | {str(list(gold_category)):<17} | {str(list(pred_category))}\n"
                f"==================================================\n"
            )

            print(result_block)
            f_out.write(result_block + "\n")
            f_out.flush()

        summary = f"\nFinal Accuracy: {correct_levels}/{sample_count} ({correct_levels/sample_count*100:.1f}%) | Valid JSON: {valid_json_count}/{sample_count}"
        print(summary)
        f_out.write(summary + "\n")

if __name__ == "__main__":
    evaluate()