import json
import random
import torch
import os
import re
from datetime import datetime
from unsloth import FastLanguageModel

# ================= Configuration =================
MODEL_PATH = "/home/data601/project/fine_tuned_model/Full-FT-4B-20k-1e5-1ep-hybrid/checkpoint-100"
DATA_PATH = "/home/data601/project/dataset/tmp/dataset_split/train_reannotated.jsonl"
OUTPUT_FILE = "evaluation_comparison_report.txt"
NUM_SAMPLES = 20

# Full System Prompt (Keep this intact)
SYSTEM_PROMPT = """You are a content moderation expert. Your goal is to distinguish between harmful content and legitimate discourse with HIGH PRECISION. 

CRITICAL INSTRUCTION: FIGHT HALLUCINATION
- Do NOT read intent that is not explicitly in the text.
- If a word has a physical meaning (e.g., "trash", "dirt", "burn"), assume the PHYSICAL/LITERAL meaning first.
- ONLY classify as Level 4/5 if there is EXPLICIT evidence of identity attacks, real-world danger, or obfuscated hate (specific emojis).
- When in doubt, default to Level 1 (Harmless).

1. JSON Output Schema:
{
  "original_comment": "The input text",
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

def extract_json_from_text(text):
    """Attempt to extract valid JSON object from a string."""
    try:
        # 1. Try finding a JSON code block
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # 2. Try finding raw JSON object logic (first '{' to last '}')
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # 3. Last resort: assume the whole text is JSON
        return json.loads(text)
    except Exception:
        return None

def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    print(f"Loading dataset from {DATA_PATH}...")
    dataset = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    dataset.append(json.loads(line))
                except:
                    continue

    sample_count = min(NUM_SAMPLES, len(dataset))
    samples = random.sample(dataset, sample_count)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n--------------------------------------------------\nAutomated Comparison Report {timestamp}\n--------------------------------------------------\n"
    
    print(header)
    
    correct_levels = 0
    valid_json_count = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        f_out.write(header + "\n")

        for i, sample in enumerate(samples):
            user_input = sample.get("original_comment", "")
            
            # Gold Standards
            gold_level = sample.get("impact_level")
            gold_category = set(sample.get("harm_category", []))
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True,
                return_tensors = "pt",
            ).to("cuda")

            attention_mask = torch.ones_like(inputs)
            
            print(f"Processing Case {i+1} / {sample_count}...")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids = inputs,
                    attention_mask = attention_mask,
                    max_new_tokens = 512,
                    use_cache = True,
                    pad_token_id = tokenizer.eos_token_id,
                    temperature = 0.1,
                )
            
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract Text Response
            try:
                model_raw_response = generated_text.split(user_input)[-1].strip()
                if model_raw_response.startswith("assistant"):
                    model_raw_response = model_raw_response.replace("assistant", "", 1).strip()
            except:
                model_raw_response = generated_text

            # Parse JSON
            model_json = extract_json_from_text(model_raw_response)
            
            # Comparison Logic
            status_tag = ""
            pred_level = "N/A"
            pred_category = "N/A"

            if model_json:
                valid_json_count += 1
                pred_level = model_json.get("impact_level")
                pred_category = set(model_json.get("harm_category", []))
                
                # Check Match
                level_match = (pred_level == gold_level)
                cat_match = (pred_category == gold_category)

                if level_match and cat_match:
                    status_tag = "[✅ PERFECT MATCH]"
                    correct_levels += 1
                elif not level_match:
                    status_tag = "[⚠️ LEVEL MISMATCH]"
                else:
                    status_tag = "[⚠️ CATEGORY MISMATCH]"
            else:
                status_tag = "[❌ JSON ERROR]"

            # Output Formatting
            result_block = (
                f"[Case {i+1}] {status_tag}\n"
                f"Input: \"{user_input[:100]}...\"\n"
                f"----------------------------------------\n"
                f"METRIC        | GOLD (Ref)        | PREDICTED (Model)\n"
                f"Level         | {gold_level:<17} | {pred_level}\n"
                f"Category      | {str(list(gold_category)):<17} | {str(list(pred_category))}\n"
                f"----------------------------------------\n"
                f"Reasoning (Model):\n{model_json.get('reasoning', 'Parse Error') if model_json else model_raw_response}\n"
                f"==================================================\n"
            )

            print(result_block)
            f_out.write(result_block + "\n")
            f_out.flush() 

        # Summary
        summary = (
            f"\n--------------------------------------------------\n"
            f"FINAL SUMMARY:\n"
            f"Total Samples: {sample_count}\n"
            f"Valid JSON:    {valid_json_count}\n"
            f"Level Accuracy: {correct_levels}/{sample_count} ({correct_levels/sample_count*100:.1f}%)\n"
            f"--------------------------------------------------\n"
        )
        print(summary)
        f_out.write(summary + "\n")

    print(f"\nEvaluation Complete. Results saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    evaluate()