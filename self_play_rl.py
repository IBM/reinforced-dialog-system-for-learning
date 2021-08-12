import argparse
import datasets
from datasets import load_dataset, load_metric
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING
)
from utils.self_play_infra_utils import *
from utils.self_play_model_utils import *
import random
from accelerate import Accelerator
import pickle
from utils.reward_utils import *
from consts import *


logger = logging.get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
TMP_PATH = 'tmp/'


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--train_size", type=int, default=None, help="Number of instances used for training"
    )
    parser.add_argument(
        "--eval_size", type=int, default=None, help="Number of instances used for validation"
    )
    # parser.add_argument(
    #     "--max_length",
    #     type=int,
    #     default=128,
    #     help=(
    #         "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
    #         " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    #     ),
    # )
    parser.add_argument(
        "--num_turns",
        type=int,
        help="Number of turns in a conversation",
        default=5
    )
    parser.add_argument(
        "--num_candicates",
        type=int,
        default=16,
        help="Number of candidate responses",
    )
    parser.add_argument(
        "--reverse",
        type=bool,
        help="Is the model input in reverse order",
        default=True
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    # parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Number of steps to save a model"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Number of steps to eval a model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model")
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Where to store the final log"
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--wiz_path",
        default=None,
        help="Path to pretrained wizard model"
    )
    parser.add_argument(
        "--app_path",
        default=None,
        help="Path to pretrained apprentice model"
    )

    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args


def main():
    args = parse_args()
    with open('../Talk_/za/args/args_self_play_rl.pkl', 'wb') as f:
        pickle.dump(args, f)
    # exit()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    log_level = 'DEBUG'
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()

    # get model config
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # selector
    with open('./za/args/args_self_play.pkl', 'rb') as f:
        args_sel = pickle.load(f)
        args_sel.model_file_path = args.model_name_or_path

    sel = BertSel(args_sel)

    # wizard
    with open('../Talk_/za/args/args_doha_train.pkl', 'rb') as f:
        args_wiz = pickle.load(f)
        args_wiz.experiment_type = 'chat_document'
        args_wiz.model_file_path = args.wiz_path
    wiz = MultiBartQA(args_wiz)

    # apprentice
    with open('../Talk_/za/args/args_bart_train.pkl', 'rb') as f:
        args_app = pickle.load(f)
        args_app.experiment_type = 'chat_document'
        args_app.model_file_path = args.app_path
    app = BartQA(args_app)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in sel.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in sel.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # device = accelerator.device
    # sel.to(device)
    wiz, app, sel, optimizer = accelerator.prepare(wiz, app, sel, optimizer)

    # Get scorers
    scorer_coverage = CoverageScorer()
    alpha_coverage = 1.0
    scorers = [scorer_coverage]
    alphas = [alpha_coverage]

    trainer = RLTrainer(args, wiz, app, sel, scorers, alphas, optimizer, accelerator)

    assert args.train_file is not None or args.validation_file is not None
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, field='data')
    if args.train_size is not None:
        train_dataset = raw_datasets['train'].select(range(args.train_size))
    else:
        train_dataset = raw_datasets['train']
    if args.eval_size is not None:
        eval_dataset = raw_datasets['validation'].select(range(args.eval_size))
    else:
        eval_dataset = raw_datasets['validation']
    # logger.info(f"  Total optimization steps =  {min(len(train_dataset), args.num_train_epochs)}")
    # logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    trainer.train_self_play_rl(train_dataset, eval_dataset)


if __name__ == "__main__":
    main()
