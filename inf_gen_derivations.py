import transformers
import pandas as pd
import os
import torch
import argparse
from unsloth import FastLanguageModel


def init():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--dataset_name",
                             default="2sat_25mixVars_25fixCls_4-hop_10K_500_OR",
                             type=str)
    args_parser.add_argument("--dataset_dir", default="dataset/Derivations/25vars_25cls", type=str)
    args_parser.add_argument("--model_id", default="unsloth/llama-3-8b-Instruct-bnb-4bit", type=str)  
    args_parser.add_argument("--output_dir", default="output/gen_Derivations", type=str)
    args_parser.add_argument("--prompt_type", default=5, type=int)  # 1:entailment, 2/3/4:inconsistencies
    args_parser.add_argument("--max_new_tokens", default=2, type=int)
    args = args_parser.parse_args()
    return args


def formatting_prompts_test(sentence1, sentence2):
    text = ""
    match args.prompt_type:
        case 1:  # Prompt for predicting entailment
            messages = [
                {"role": "system", "content": "The input below provides pair of a set of premises and a "
                                              "hypothesis. Is the hypothesis entailed by the premises? "
                                              "Answer with yes or no only."},
                {"role": "user", "content": "Premises: " + sentence1 + " Hypothesis: " +
                                            sentence2 + "."},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        case 2:  # Prompt for identifying inconsistency
            messages = [
                {"role": "system", "content": "The input below contains a set of premises. Are there two or "
                                              "more premises that contradict each other within these "
                                              "premises? Answer with yes or no only."},
                {"role": "user", "content": sentence1},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        case 3:
            messages = [
                {"role": "system", "content": "The input below contains a set of premises. Is the premises "
                                              "inconsistent? Answer with yes or no only."},
                {"role": "user", "content": sentence1},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        case 4:
            messages = [
                {"role": "system", "content": "The input below contains a set of premises. If there are any "
                                              "two or more premises that contradict each other, it means "
                                              "that the set of premises is inconsistent. "
                                              "Are the premises inconsistent? Answer with yes or no only."},
                {"role": "user", "content": sentence1},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        case 5:
            messages = [
                {"role": "system", "content": "The input below provides pair of a set of premises and a hypothesis. "
                                              "Generate derivations from the given premises."},
                {"role": "user", "content": sentence1},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return text


if __name__ == '__main__':
    args = init()

    max_seq_length = 8192
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )

    FastLanguageModel.for_inference(model)

    # ---- Testing the finetuned model ---
    df_test = pd.read_csv(os.path.join(args.dataset_dir, args.dataset_name + ".csv"), sep=None, engine='python')

    f_preds = os.path.join(args.output_dir,  "PRED_" + str(args.max_new_tokens) + "_" + args.dataset_name + ".csv")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with open(f_preds, 'w') as file_output:
        file_output.write("id;pred;label;num_atoms;k_hop\n")

        for index, row in df_test.iterrows():
            text = formatting_prompts_test(row['sentence1'], row['sentence2'])

            outputs = pipeline(
                text,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            result = outputs[0]["generated_text"][len(text):]
            str_output = str(row['idx']) + ";" + str(result) + ";" + str(row['label']) \
                         + ";" + str(row['num_atoms']) + ";" + str(row['k_hop']) + "\n"
            file_output.write(str_output)

            print(str(row['idx']) + " " + result)

        file_output.close()
