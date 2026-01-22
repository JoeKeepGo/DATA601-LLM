import json
import os
import random
import re # 把正则库放在最上面

# ================= 配置路径 =================
BASE_DIR = "/home/data601/project"

# 1. 输入：新数据 (V8 Prompt)
SYNTHETIC_FILE = os.path.join(BASE_DIR, "dataset/tmp/synthetic_train_final_v8prompt.jsonl")

# 2. 输入：黄金数据集
GOLDEN_FILE = os.path.join(BASE_DIR, "dataset/tmp/dataset_split/train_reannotated.jsonl")

# 3. 输出
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset/train/train_quality_hybrid_cot.jsonl")

# 超采样倍数
# 0 = 不额外复制 (只保留原始的1份黄金数据)
# 1 = 额外复制1份 (总共2份黄金数据)
OVERSAMPLE_RATIO = 0 

# ================= System Prompt =================
# (请保持你的 V8 Prompt 内容)
SYSTEM_PROMPT = """
# Role
You are a content moderation expert. Your goal is to distinguish between harmful content and legitimate discourse with HIGH PRECISION.

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
"""

# 用户指令模板
USER_PROMPT_TEMPLATES = [
    "Analyze this: {}",
    "Review this comment: {}",
    "Check for harmful content: {}",
    "Classify this text: {}",
    "Safety audit: {}",
    # 纯文本/零样本 (高权重)
    "{}", 
    "{}", 
    "{}", 
    "{}", 
    "{}", 
]

def process_file(filepath, label_tag="Data"):
    converted_buffer = []
    if not os.path.exists(filepath):
        print(f"[{label_tag}] Warning: File not found at {filepath}")
        return []

    print(f"[{label_tag}] Processing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                user_input = record.get('original_comment', '').strip()
                if not user_input: continue

                # ================= 核心适配逻辑 =================
                analysis_text = ""
                json_data = {}

                # 情况 A: 新格式 (reprocess_cot.py 生成的)
                if "cot_analysis" in record and "label_json" in record:
                    analysis_text = record["cot_analysis"]
                    json_data = record["label_json"]
                    if "original_comment" in json_data: del json_data["original_comment"]

                # 情况 B: 旧格式 (Golden Data 可能还是旧的)
                else:
                    json_data = record.copy()
                    if "original_comment" in json_data: del json_data["original_comment"]
                    analysis_text = json_data.pop("reasoning", "Analysis provided in JSON.")
                    json_data["reasoning"] = "See detailed analysis above."

                # 验证完整性
                if not analysis_text or not json_data:
                    continue
                    
                # ❌ 删除这行会导致报错的旧代码:
                # reasoning_text = output_payload.pop('reasoning', "Analysis provided in JSON.")
                
                # ✅ 修复 1 & 2: 对 analysis_text 进行清洗，并在后面使用它
                # 清洗可能存在的尾部标题，防止重复
                analysis_text = re.sub(r'###\s*JSON Result:?\s*$', '', analysis_text.strip(), flags=re.IGNORECASE).strip()
                
                # ================= 构建 Hybrid Response =================
                # 严格遵循 V8 格式：### Analysis: -> [Content] -> ### JSON Result: -> [JSON]
                hybrid_response = (
                    f"### Analysis:\n{analysis_text}\n\n" # 使用清洗后的变量
                    f"### JSON Result:\n```json\n"
                    f"{json.dumps(json_data, ensure_ascii=False)}\n"
                    f"```"
                )
                
                # 随机选择模板
                prompt_template = random.choice(USER_PROMPT_TEMPLATES)
                final_user_content = prompt_template.format(user_input)

                conversation = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT.strip()},
                        {"role": "user", "content": final_user_content},
                        {"role": "assistant", "content": hybrid_response}
                    ]
                }
                converted_buffer.append(conversation)
            except Exception as e:
                continue
    
    print(f"[{label_tag}] Loaded {len(converted_buffer)} samples.")
    return converted_buffer

def main():
    # 1. 读取新生成的合成数据
    synthetic_data = process_file(SYNTHETIC_FILE, label_tag="Synthetic-V8")
    
    # 2. 读取黄金数据 (作为补充)
    golden_data = process_file(GOLDEN_FILE, label_tag="Golden-Legacy")
    
    if not synthetic_data and not golden_data:
        print("Error: No data loaded.")
        return

    # 3. 合并与打乱
    # ✅ 修复 3: 这里的逻辑现在是 "合成数据 + 1份黄金数据 + 0份额外黄金数据"
    # 如果你想彻底去掉黄金数据，请把中间那个 golden_data 删掉
    # final_dataset = synthetic_data + golden_data + (golden_data * OVERSAMPLE_RATIO)
    final_dataset = synthetic_data + (golden_data * OVERSAMPLE_RATIO)
    random.shuffle(final_dataset)
    
    print(f"\nTotal samples: {len(final_dataset)}")
    
    # 4. 写入输出
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Dataset saved to: {OUTPUT_FILE}")
    
    # 5. 打印一条样本进行人工核对
    if len(final_dataset) > 0:
        print("\n" + "="*50)
        print("SAMPLE CHECK (Verify this matches Prompt V8 Format):")
        sample = final_dataset[0]
        print(f"User: {sample['messages'][1]['content']}")
        print(f"Assistant:\n{sample['messages'][2]['content']}")
        print("="*50)

if __name__ == "__main__":
    main()