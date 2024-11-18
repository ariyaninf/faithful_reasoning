import pandas as pd
import os
import torch
import argparse
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def init():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--dataset_name", default="2sat_15_mixVars_50_mixCls_100K_500_OR", type=str)
    args_parser.add_argument("--dataset_dir", default="dataset/Entailments_v5/", type=str)
    args_parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct", type=str)
    args_parser.add_argument("--output_dir", default="output/Entailments_v5/", type=str)
    args_parser.add_argument("--max_seq_length", default=1024, type=int)
    args_parser.add_argument("--prompt_type", default=1, type=int)
    args_parser.add_argument("--batch_size", default=4, type=int)
    args_parser.add_argument("--epochs", default=2, type=int)
    args = args_parser.parse_args()
    return args


id_to_label = {
    0: "No",
    1: "Yes"
}


def formatting_prompts_func(examples):
    match args.prompt_type:
        # Llama, Qwen
        case 1:
            messages = [
                {"role": "system", "content": "The input below provides pair of a set of premises and a "
                                              "hypothesis. Is the hypothesis entailed by the premises? "
                                              "Answer with yes or no only."},
                {"role": "user", "content": "Premises: " + examples['sentence1'] + " Hypothesis: " +
                                            examples['sentence2'] + "."},
                {"role": "system", "content": id_to_label[examples['label']]},
            ]
            examples["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

        # Phi-3.5
        case 2:
            messages = [
                {"role": "system", "content": "The input below provides pair of a set of premises and a "
                                              "hypothesis. Is the hypothesis entailed by the premises? "
                                              "Answer with yes or no only."},
                {"role": "user", "content": "Premises: " + examples['sentence1'] + " Hypothesis: " +
                                            examples['sentence2'] + "."},
                {"role": "assistant", "content": id_to_label[examples['label']]},
            ]
            examples["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

        # Mistral-7b, gemma-tb-it, Falcon
        case 6:
            messages = [
                {"role": "user", "content": "The input below provides pair of a set of premises and a hypothesis. "
                                            "Is the hypothesis entailed by the premises? Answer with yes or no "
                                            "only.\nPremises: " + examples['sentence1'] + "\nHypothesis: "
                                            + examples['sentence2'] + "."},
                {"role": "assistant", "content": id_to_label[examples['label']]},
            ]
            examples["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return examples


def load_datasets():
    df_train = pd.read_csv(os.path.join(args.dataset_dir, args.dataset_name + "_train.csv"), sep=None, engine="python")
    df_val = pd.read_csv(os.path.join(args.dataset_dir, args.dataset_name + "_val.csv"), sep=None, engine='python')

    data_train = Dataset.from_pandas(df_train)
    data_val = Dataset.from_pandas(df_val)
    return data_train, data_val


if __name__ == '__main__':
    args = init()

    max_seq_length = args.max_seq_length
    dtype = None
    load_in_4bit = True
    output_dir = os.path.join(args.output_dir, args.model_id, args.dataset_name + "_prompt_" + str(args.prompt_type))

    model_id = args.model_id

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"Output directory {output_dir} already exists")

    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # For 4 bit quantization
    compute_dtype = getattr(torch, "float16")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


    def get_device_map() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'


    device_map = get_device_map()  # 'cpu'

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=quantization_config,
                                                 device_map={"": 0},
                                                 )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="linear",
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    torch.cuda.empty_cache()
    ds_train, ds_val = load_datasets()
    ds_train = ds_train.map(formatting_prompts_func)
    ds_val = ds_val.map(formatting_prompts_func)

    print(ds_train["text"][2])

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=False,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
    )

    trainer.train()

    pretrained_dir = os.path.join(output_dir, "lora_model")
    trainer.model.save_pretrained(pretrained_dir)
    trainer.tokenizer.save_pretrained(pretrained_dir)
