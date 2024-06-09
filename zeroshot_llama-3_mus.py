import argparse

from unsloth import FastLanguageModel
import torch
import os.path
import transformers
import pandas as pd


def init():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--dataset_name", default="2sat_25mixVars_25fixCls_0.3_1-hop_100K_500_OR",
                             type=str)
    args_parser.add_argument("--dataset_dir", default="dataset/Entailments/25vars_25cls", type=str)
    args_parser.add_argument("--model_name", default="unsloth/llama-3-8b-Instruct-bnb-4bit", type=str)
    args_parser.add_argument("--output_dir", default="output/zero_shot", type=str)
    args_parser.add_argument("--prompt_type", default=1, type=int)
    args = args_parser.parse_args()
    return args


id_to_label = {
    0: "No",
    1: "Yes"
}


if __name__ == '__main__':
    args = init()

    model_id = args.model_name
    output_dir = os.path.join(args.output_dir, args.model_name)

    df_test = pd.read_csv(os.path.join(args.dataset_dir, args.dataset_name + "_test.csv"), sep=None, engine='python')
    f_preds = os.path.join(output_dir, "PREDS_type" + str(args.prompt_type) + "_" + args.dataset_name + ".csv")

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    def formatting_prompts_test(sentence1, sentence2, prompt_type):
        str_text = ""
        match prompt_type:
            case 1:  # Prompt for predicting entailment
                prompt = [
                    {"role": "system", "content": "The input below provides pair of a set of premises and a "
                                                  "hypothesis. Is the hypothesis entailed by the premises? "
                                                  "Answer with yes or no only."},
                    {"role": "user", "content": "Premises: " + sentence1 + " Hypothesis: " +
                                                sentence2 + "."},
                ]
                str_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            case 2:  # Prompt for identifying inconsistency
                prompt = [
                    {"role": "system", "content": "The input below contains a set of premises. Are there two or "
                                                  "more premises that contradict each other within these "
                                                  "premises? Answer with yes or no only."},
                    {"role": "user", "content": sentence1},
                ]
                str_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            case 3:
                prompt = [
                    {"role": "system", "content": "The input below contains a set of premises. Is the premises "
                                                  "inconsistent? Answer with yes or no only."},
                    {"role": "user", "content": sentence1},
                ]
                str_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            case 4:
                prompt = [
                    {"role": "system", "content": "The input below contains a set of premises. If there are any "
                                                  "two or more premises that contradict each other, it means "
                                                  "that the set of premises is inconsistent. "
                                                  "Are the premises inconsistent? Answer with yes or no only."},
                    {"role": "user", "content": sentence1},
                ]
                str_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            case 5:
                prompt = [
                    {"role": "system", "content": "The following input consists of a set of premises (P), some of "
                                                  "which are inconsistent. Find the minimal unsatisfiable subset "
                                                  "of P. Briefly answer the question by mentioning the MUS premises "
                                                  "only. All MUS premises must be exist in P. State the MUS premises "
                                                  "consecutively and separated by a period as in the following "
                                                  "example: 'Premise a. Premise b. ....'. "
                                                  "Do not use a new line in the answer."},
                    {"role": "user", "content": sentence1},
                ]
                str_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        return str_text


    with open(f_preds, 'a') as file_output:
        file_output.write("id;label;pred;num_atoms;num_clauses;k_hop\n")
        for index, row in df_test.iterrows():

            messages = formatting_prompts_test(row['sentence1'], row['label'], args.prompt_type)

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

            outputs = pipeline(
                messages,
                max_new_tokens=100,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            result = outputs[0]["generated_text"][len(messages):]

            str_output = str(row['id']) + ";" + row['label'] + ";" + result \
                         + ";" + str(row['num_atoms']) + ";" + str(row['num_clauses']) + ";" + str(row['k_hop']) + "\n"
            file_output.write(str_output)

            print(str(row['id']) + " " + result)

    file_output.close()
