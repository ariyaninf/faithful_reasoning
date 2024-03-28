import argparse
import os.path
from loader import *
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorWithPadding, \
    TrainingArguments, pipeline
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
import torch
import logging


def init():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--dataset", default="SENT_inferred_2sat_5fixVars_5fixCls_2K_100_OR", type=str)
    args_parser.add_argument("--dataset_dir", default="dataset/Synthetic_CounterEx/5_var_5_cls", type=str)
    args_parser.add_argument("--output_dir", default="output/trained_genProof", type=str)
    args_parser.add_argument("--model_name", default="save-model-llama2", type=str)
    args_parser.add_argument("--model_dir", default="output/trained_genProof/Llama-2-7b-chat-hf/"
                                                    "SENT_inferred_2sat_5fixVars_5fixCls_entailed_1K_100_OR_type5_4bit"
                             , type=str)
    args_parser.add_argument("--format_type", default=5, type=int)
    args_parser.add_argument("--load_in_bit", default=4, type=int)
    args_parser.add_argument("--batch_size", default=4, type=int)
    args_parser.add_argument("--lora_alpha", default=16, type=int)
    args_parser.add_argument("--lora_dropout", default=0.1, type=float)
    args_parser.add_argument("--lora_r", default=16, type=int)
    args_parser.add_argument("--epochs", default=3, type=int)
    args_parser.add_argument("--save_steps", default=250, type=int)
    args_parser.add_argument("--max_seq_length", default=256, type=int)
    args = args_parser.parse_args()
    return args

def load_datasets():
    df_train = pd.read_csv(os.path.join(args.dataset_dir, args.dataset + "_train.csv"), sep=None, engine='python')
    df_val = pd.read_csv(os.path.join(args.dataset_dir, args.dataset + "_val.csv"), sep=None, engine='python')
    df_test = pd.read_csv(os.path.join(args.dataset_dir, args.dataset + "_test.csv"), sep=None, engine='python')

    data_train = Dataset.from_pandas(df_train)
    data_val = Dataset.from_pandas(df_val)
    data_test = Dataset.from_pandas(df_test)

    return data_train, data_val, data_test


def formatting_prompts_func(example):
    match args.format_type:
        case 1:
            example["text"] = f"### Question: Premises: {example['sentence1']} Hypothesis: {example['sentence2']}\n" \
                              f"### Answer: {example['explanation']}"
        case 2:
            example["text"] = f"### Question: Generate reasoning steps to prove the entailment. " \
                              f"Premises: {example['sentence1']} " \
                              f"Hypothesis: {example['sentence2']}\n" \
                              f"### Answer: {example['explanation']}"
        case 3:
            example["text"] = f"<s>[INST]<<SYS>>" \
                              f"Premises: {example['sentence1']}\n" \
                              f"Hypothesis: {example['sentence2']}<<SYS>>[/INST]\n" \
                              f"{example['explanation']}</s>"
        case 4:
            example["text"] = f"<s>[INST]<<SYS>>Generate reasoning steps to prove the entailment. <</SYS>>" \
                              f"Premises: {example['sentence1']}\n" \
                              f"Hypothesis: {example['sentence2']}[/INST]\n" \
                              f"{example['explanation']}</s>"
        case 5:
            example["text"] = f"""
                        <<SYS>>
                        Generate deduction steps from the given premises in the content for 'User' to prove that 
                        the given hypothesis is entailed by the premises.\n
                        <<SYS>>
                        [INST]
                        User: Premises: {example['sentence1']} Hypothesis: {example['sentence2']}
                        [/INST]\n
                        
                        Assistant: 
                        {example['explanation']}
                        """
    return example


def formatting_test_func(example):
    match args.format_type:
        case 1:
            example["text"] = f"### Question: Premises: {example['sentence1']} Hypothesis: {example['sentence2']}\n" \
                              f"### Answer: "
        case 2:
            example["text"] = f"### Question: Generate reasoning steps to prove the entailment. " \
                              f"Premises: {example['sentence1']} " \
                              f"Hypothesis: {example['sentence2']}\n" \
                              f"### Answer: "
        case 3:
            example["text"] = f"<s>[INST]<<SYS>>" \
                              f"Premises: {example['sentence1']}\n" \
                              f"Hypothesis: {example['sentence2']}<<SYS>>[/INST]</s>"
        case 4:
            example["text"] = f"<s>[INST]<<SYS>>Generate reasoning steps to prove the entailment. <</SYS>>" \
                              f"Premises: {example['sentence1']}\n" \
                              f"Hypothesis: {example['sentence2']}[/INST]</s>"
        case 5:
            example["text"] = f"""
                                <<SYS>>
                                Generate deduction steps from the given premises in the content for 'User' to prove that 
                                the given hypothesis is entailed by the premises.\n
                                <<SYS>>
                                [INST]
                                User: Premises: {example['sentence1']} Hypothesis: {example['sentence2']}
                                [/INST]\n

                                Assistant: 
                                """
    return example


def load_in_bit(n_bit):
    match n_bit:
        case 8:
            bnb_8bit_compute_dtype = "float16"
            compute_dtype = getattr(torch, bnb_8bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype
            )
            return bnb_config, compute_dtype
        case 4:
            bnb_4bit_compute_dtype = "float16"
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False
            )
            return bnb_config, compute_dtype


if __name__ == '__main__':
    args = init()

    output_dir = os.path.join(args.output_dir, "trained_from_entailment", args.dataset + "_type" +
                              str(args.format_type) + "_" + str(args.load_in_bit) + "bit")

    # output_dir = os.path.join(args.output_dir, args.model_name, args.dataset + "_type" + str(args.format_type) +
    #                           "_" + str(args.load_in_bit) + "bit")

    print('output_dir: ', output_dir)
    model_name = os.path.join(args.model_dir, args.model_name)

    logging_path = init_logging_path(output_dir)
    print(logging_path)
    logging.basicConfig(filename=logging_path, encoding='utf-8', level=logging.INFO)
    logging.info(str(args))

    ds_train, ds_val, ds_test = load_datasets()
    ds_train = ds_train.map(formatting_prompts_func)
    ds_val = ds_val.map(formatting_prompts_func)
    ds_test = ds_test.map(formatting_test_func)

    columns_to_remove = ['num_atoms', 'num_clauses', 'derivations']
    ds_test = ds_test.remove_columns(columns_to_remove)

    print(ds_train["text"][2])

    bnb_config, compute_dtype = load_in_bit(args.load_in_bit)

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )

    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------- 2. Fine-tune pretrained model -------- #
    # Define Trainer parameters
    logging.info("Start training...")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        # eval_steps=625,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps
        # seed=100
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        data_collator=data_collator,
        packing=True
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir + "/save-model-llama2")

    df_logs = pd.DataFrame(trainer.state.log_history)
    print(df_logs)

    df_logs.to_csv(output_dir + "/train_log.csv", sep=';')

    # --------------- 3. Evaluate model using data test --------------------#
    model.eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=compute_dtype,
        top_k=50,
        max_new_tokens=300,
        return_full_text=False,
        temperature=.9,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    '''
    sequences = pipeline(
        ds_test["text"],
        num_return_sequence=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        # early_stopping=True
    )
    '''
    prompts = ds_test["text"]
    ids = ds_test["idx"]
    f_preds = os.path.join(output_dir, "preds_temp_09.txt")

    with open(f_preds, 'a') as file_preds:
        file_preds.write("idx;preds\n")
        for idx, prompt in zip(ids, prompts):
            generated_text = generator(prompt)
            result = generated_text[0]['generated_text']
            result = result.replace("\n", " ")
            file_preds.write(str(idx) + ";" + result + "\n")
            print(result)
    file_preds.close()

    '''
    f_preds = os.path.join(output_dir, "preds.txt")

    with open(f_preds, 'a') as file_preds:
        file_preds.write("generated_proof\n")
        for seq in sequences:
            result = seq[0]['generated_text']
            result = result.replace("\n", " ")
            file_preds.write(result + "\n")
        file_preds.close()
    '''