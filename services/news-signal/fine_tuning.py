from typing import Tuple, Optional
import os
import comet_ml
from loguru import logger
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from trl import SFTTrainer
from sklearn.model_selection import train_test_split

def load_base_llm_and_tokenizer(
        base_llm_name: str,
        max_seq_length: Optional[int] = 2048,
        dtype: Optional[str] = None,
        load_in_4bit: Optional[bool] = True,
        ) -> Tuple[FastLanguageModel, AutoTokenizer]:
    """
    Loads and returns the base LLM and tokenizer.

    Args:
        base_llms_name: The name of the base LLM to load.
        max_seq_length: The maximum sequence length to use for the model.
        dtype: The data type to use for the model.
        load_in_4bit: Whether to load the model in 4-bit mode.

    Returns:
        The base LLM and tokenizer.
    """

    logger.info("Adding base model and tokenizer")

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_llm_name, # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    return model, tokenizer

def add_lora_adapters(
    model: FastLanguageModel) -> FastLanguageModel:
    """
    Add LoRA adapters to the base model

    """

    model = FastLanguageModel.get_peft_model(

        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model
    
def load_and_split_dataset(
    dataset_path: str,
    eos_token: str,
    ) -> Tuple[Dataset, Dataset]:
    """
    Loads and preprocesses the dataset.
    """
    # load the dataset form JSONL file into a HuggingFace Dataset Object
    logger.info(f"Loading and preprocessing dataset {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path)
    dataset = dataset['train']

    def format_prompts(examples):
    # chat template we use to format the data we feed into the model
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provies further context. Write a responde that appropiatley completes the request.



        ### Input:
        {}

        ### Response:
        {}
        """
        
        # instructions = examples["Instruction"] # Imstruction are omitted from the dataset
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for input, output in zip(inputs, outputs):
            # Must add eos_token, otherwise your generation will go on forever!
            text = alpaca_prompt.format(input, output) + eos_token
            texts.append(text)
           
        return { "text" : texts, }

    dataset = dataset.map(format_prompts, batched = True,)
    # split the dataset into train and test, with a fix seed to ensure reproducibility
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset['train'], dataset['test']

def fine_tune(
        model: FastLanguageModel,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        test_dataset: Dataset,
        max_seq_length: int,

        ):

    """
    fine-tunes the model using supervised fine tuning.
    """

    # 1. train with Hyperparameter Optimization

    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False, # Can make training 5x faster for shoert sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            learning_rate=2e-4,
            max_steps=60,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed = 3407,
            output_dir="outputs",
            report_to="comet_ml", # Use this for WandB etc
        ),
    )
    # start training
    trainer.train()

    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False, # Can make training 5x faster for shoert sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            learning_rate=2e-4,
            max_steps=60,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed = 3407,
            output_dir="outputs",
            report_to="comet_ml", # Use this for WandB etc
        ),

def run(
        base_llm_name: str,
        dataset_path: str,
        comet_ml_project_name: str,
        max_seq_length: Optional[int] = 2048,
        ):
    """
    
    Fine-tunes a base LLM using supervised fine tuning.
    The training results are logged to CometML.
    The final artifact is saved as an Ollama model, so we can use it to generate signals

    Args:
        base_llms_name: The name of the base LLM to fine-tune.
        dataset_path: The path to the dataset to use for fine-tuning.
    """

    # Dashboard logging
    os.environ["COMET_LOG_ASSETS"] = "True"

    comet_ml.login(project_name="comet-example-unsloth")
    logger.info(f"Logged in to CometML project {comet_ml_project_name}")

    # 1. Load the base LLM and tokenizer
    model, tokenizer = load_base_llm_and_tokenizer(base_llm_name, max_seq_length=max_seq_length)

    # 2. add LoRA adapters to the base model
    model = add_lora_adapters(model)

    # 3. Load the dataset with (instruction, input, output) tuples into a HuggingFace Dataset object
    # with alpaca prompt format
    train_dataset, test_dataset = load_and_split_dataset(dataset_path, eos_token=tokenizer.eos_token)

    # 4. Fine-tune the base LLM
    fine_tune(model, tokenizer, train_dataset, test_dataset, max_seq_length=max_seq_length)


if __name__ == '__main__':
    from fire import Fire
    Fire(run)
