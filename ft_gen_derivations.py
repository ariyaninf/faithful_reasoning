import transformers
import pandas as pd
import os
import torch
import argparse
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def init():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--dataset_name",
                             default="2sat_25mixVars_25fixCls_4-hop_10K_500_OR",
                             type=str)
    args_parser.add_argument("--dataset_dir", default="dataset/Derivations/25vars_25cls", type=str)
    args_parser.add_argument("--model_id", default="unsloth/llama-3-8b-Instruct-bnb-4bit", type=str)  
    args_parser.add_argument("--output_dir", default="output/gen_Derivations", type=str)
    args_parser.add_argument("--prompt_type", default=5, type=int)  # 1:entailment, 2/3/4:inconsistencies
    args_parser.add_argument("--batch_size", default=2, type=int)
    args_parser.add_argument("--epochs", default=2, type=int)
    args_parser.add_argument("--save_steps", default=1250, type=int)
    args = args_parser.parse_args()
    return args


def load_datasets():
    df_train = pd.read_csv(os.path.join(args.dataset_dir, args.dataset_name + "_train.csv"), sep=None, engine='python')
    df_val = pd.read_csv(os.path.join(args.dataset_dir, args.dataset_name + "_val.csv"), sep=None, engine='python')

    data_train = Dataset.from_pandas(df_train)
    data_val = Dataset.from_pandas(df_val)
    return data_train, data_val


def formatting_prompts_func(examples):
    match args.prompt_type:
        case 1:  # Prompt for predicting entailment
            messages = [
                {"role": "system", "content": "The input below provides pair of a set of premises and a "
                                              "hypothesis. Is the hypothesis entailed by the premises? "
                                              "Answer with yes or no only."},
                {"role": "user", "content": "Premises: " + examples['sentence1'] + " Hypothesis: " +
                                            examples['sentence2'] + "."},
                {"role": "system", "content": id_to_label[examples['label']]},
            ]
            examples["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

        case 2:  # Prompt for identifying inconsistency
            messages = [
                {"role": "system", "content": "The input below contains a set of premises. Are there two or "
                                              "more premises that contradict each other within these "
                                              "premises? Answer with yes or no only."},
                {"role": "user", "content": examples['sentence1']},
                {"role": "system", "content": id_to_label[examples['label']]},
            ]
            examples["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

        case 3:
            messages = [
                {"role": "system", "content": "The input below contains a set of premises. Is the premises "
                                              "inconsistent? Answer with yes or no only."},
                {"role": "user", "content": examples['sentence1']},
                {"role": "system", "content": id_to_label[examples['label']]},
            ]
            examples["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

        case 4:
            messages = [
                {"role": "system", "content": "The input below contains a set of premises. If there are any "
                                              "two or more premises that contradict each other, it means "
                                              "that the set of premises is inconsistent. "
                                              "Are the premises inconsistent? Answer with yes or no only."},
                {"role": "user", "content": examples['sentence1']},
                {"role": "system", "content": id_to_label[examples['label']]},
            ]
            examples["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

        case 5:
            messages = [
                {"role": "system", "content": "The input below provides pair of a set of premises and a hypothesis. "
                                              "Generate derivations from the given premises."},
                {"role": "user", "content": examples['sentence1']},
                {"role": "system", "content": examples['explanation']},
            ]
            examples["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return examples


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

    ds_train, ds_val = load_datasets()
    ds_train = ds_train.map(formatting_prompts_func)
    ds_val = ds_val.map(formatting_prompts_func)

    print(ds_train["text"][2])

    output_dir = os.path.join(args.output_dir, args.model_id, args.dataset_name + "_prompt_" + str(args.prompt_type))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy="steps",
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1000,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=args.save_steps
        )
    )

    trainer_stats = trainer.train()

    pretrained_dir = os.path.join(output_dir, "lora_model")
    model.save_pretrained(pretrained_dir)
    tokenizer.save_pretrained(pretrained_dir)

    # ---- Testing the finetuned model ---
    df_test = pd.read_csv(os.path.join(args.dataset_dir, args.dataset_name + "_test.csv"), sep=None, engine='python')

    f_preds = os.path.join(output_dir, "predictions.csv")

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
        file_output.write("id;num_clauses;pred;k_hop\n")

        for index, row in df_test.iterrows():
            text = formatting_prompts_test(row['sentence1'], row['sentence2'])

            outputs = pipeline(
                text,
                max_new_tokens=2000,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            result = outputs[0]["generated_text"][len(text):]
            str_output = str(row['idx']) + ";" + str(row['num_clauses']) \
                         + ";" + str(result) + ";" + str(row['k_hop']) + "\n"
            file_output.write(str_output)

            print(str(row['idx']) + " " + result)

        file_output.close()
