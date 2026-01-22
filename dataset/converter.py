import json
import os
import random

# 配置路径
BASE_DIR = "/home/data601/project"

# 合成数据输入 (Quantity)
SYNTHETIC_FILE = os.path.join(BASE_DIR, "dataset/tmp/synthetic_train_final_v7prompt.jsonl")
# 黄金数据输入 (Quality)
GOLDEN_FILE = os.path.join(BASE_DIR, "dataset/tmp/dataset_split/train_reannotated.jsonl")
# 输出路径
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset/train/train_quality.jsonl")

# 超采样倍数
OVERSAMPLE_RATIO = 0 

# 用户提示词
USER_PROMPT_TEMPLATES = [
    "Analyze this: {}",
    "Classify the following comment based on the safety guidelines: {}",
    "Review this text for harmful content: {}",
    "Please evaluate the impact level of this input: {}",
    "Check this comment for policy violations: {}",
    "Detect if there is any hate speech or toxicity in this text: {}",
    "Content Moderation Task: Analyze the following text: {}",
    "{}", # 模拟无指令纯文本
]

# 系统提示词
# 训练数据集中不需要 "original_comment": "The input text",
SYSTEM_PROMPT = """
You are a content moderation expert. Your goal is to distinguish between harmful content and legitimate discourse with HIGH PRECISION. 

CRITICAL INSTRUCTION: FIGHT HALLUCINATION
- Do NOT read intent that is not explicitly in the text.
- If a word has a physical meaning (e.g., "trash", "dirt", "burn"), assume the PHYSICAL/LITERAL meaning first.
- ONLY classify as Level 4/5 if there is EXPLICIT evidence of identity attacks, real-world danger, or obfuscated hate (specific emojis).
- When in doubt, default to Level 1 (Harmless).

1. JSON Output Schema:
{
  "impact_level": <Integer 1-5>,
  "harm_category": ["List of strings from Allowed Categories. Empty [] if Level 1"],
  "target_identity": ["List of strings from Allowed Identities. Empty [] if none"],
  "reasoning": "Keywords: '...'. Context: ... . Evidence & Hallucination Check: ... . Impact: ... .",
  "action_suggestion": "One of: 'None', 'Collapse', 'Warn User', 'Block/Delete', 'Escalate'"
}

2. Definitions & Logic:

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

3. Special Considerations (Rules A-K):

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

4. Reasoning Field Format:
You must structure the 'reasoning' field with exactly these four parts:
1. Keywords: List trigger words.
2. Context: Topic analysis.
3. Evidence & Hallucination Check: 
   - Ask: "Is the attack EXPLICITLY against a Protected Group, or am I inferring it?"
   - Check [RULE A]: Is there a physical/literal meaning?
   - Check [RULE K]: Is this a quote/counterspeech?
   - Check [RULE B]: Is it a policy critique or a group generalization?
4. Impact: Final classification based on strict evidence.
"""

def process_file(filepath, label_tag="Data"):
    """
    读取文件并将每行转换为 SFT 格式
    """
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
                # 确保输出包含完整的 JSON 
                # assistant_output = json.dumps(record, ensure_ascii=False)
                
                if not user_input: continue

                 # 创建副本
                output_payload = record.copy()
                # 删除 original_comment 字段
                if 'original_comment' in output_payload:
                    del output_payload['original_comment']
                # 将处理后的字典转为 JSON 字符串作为 assistant 的回复
                assistant_output = json.dumps(output_payload, ensure_ascii=False)
                
                # 随机选择指令模板
                prompt_template = random.choice(USER_PROMPT_TEMPLATES)
                final_user_content = prompt_template.format(user_input)

                conversation = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT.strip()},
                        {"role": "user", "content": final_user_content},
                        {"role": "assistant", "content": assistant_output}
                    ]
                }
                converted_buffer.append(conversation)
            except Exception as e:
                # 忽略错误行，保持静默或打印日志
                continue
    
    print(f"[{label_tag}] Loaded {len(converted_buffer)} samples.")
    return converted_buffer

def main():
    # 处理合成数据
    synthetic_data = process_file(SYNTHETIC_FILE, label_tag="Synthetic")
    
    # 处理黄金数据
    golden_data = process_file(GOLDEN_FILE, label_tag="Golden")
    
    if not synthetic_data and not golden_data:
        print("Error: No data loaded from either source.")
        return

    # 3超采样
    # 如果黄金数据存在，将其复制 N 倍
    if golden_data:
        oversampled_golden = golden_data * OVERSAMPLE_RATIO
        print(f"[Oversample] Golden data multiplied by {OVERSAMPLE_RATIO}x: {len(golden_data)} -> {len(oversampled_golden)}")
    else:
        oversampled_golden = []

    # 合并与打乱
    final_dataset = synthetic_data + oversampled_golden
    random.shuffle(final_dataset) 
    
    print(f"\nTotal samples ready for training: {len(final_dataset)}")
    print(f"\n(Synthetic: {len(synthetic_data)} + Golden(x{OVERSAMPLE_RATIO}): {len(oversampled_golden)})")

    # 写入文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Writing to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("Dataset creation complete!")

    # 验证
    print("\nRandomly Sampled Examples Verification:")
    if len(final_dataset) >= 3:
        samples = random.sample(final_dataset, 3)
        for i, sample in enumerate(samples):
            print(f"\n[Sample {i+1}] User Content:")
            print(sample['messages'][1]['content'])
            # 这里可以检查 output 是不是也有内容
            # print("Assistant Preview:", sample['messages'][2]['content'][:50] + "...")

if __name__ == "__main__":
    main()