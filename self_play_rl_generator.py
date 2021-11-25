import datasets
from datasets import load_dataset, load_metric
from transformers import (
    MODEL_MAPPING
)
from accelerate import Accelerator
import pickle
from utils.self_play_infra_utils import *
from utils.self_play_train_utils import *
from consts import *

logger = logging.get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
TMP_PATH = 'tmp/'


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--train_file_rl", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file_rl", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--train_file_mle", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file_mle", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--train_size", type=int, default=None, help="Number of instances used for training"
    )
    parser.add_argument(
        "--eval_size", type=int, default=None, help="Number of instances used for validation"
    )
    parser.add_argument(
        "--num_turns",
        type=int,
        help="Number of turns in a conversation",
        default=3
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
        "--wiz_model_name",
        default='facebook/bart-base'
    )
    parser.add_argument(
        "--app_model_name",
        default='facebook/bart-base'
    )
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
    parser.add_argument(
        "--coh_path",
        default=None,
        help="Path to pretrained coherence model"
    )
    parser.add_argument(
        "--alpha_cov",
        type=float,
        default=0.9,
        help="weight for coverage score"
    )
    parser.add_argument(
        "--alpha_coh",
        type=float,
        default=0.1,
        help="weight for coherence score"
    )
    parser.add_argument(
        "--selector_type",
        type=str,
        default='pre-uttr',
        help="to apply post selector or pre selector"
    )
    parser.add_argument(
        "--batch_size_mle",
        type=int,
        default=8
    )
    parser.add_argument(
        "--batch_size_rl",
        type=int,
        default=5
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=10
    )
    parser.add_argument(
        "--num_mle_per_rl",
        type=int,
        default=3
    )
    parser.add_argument(
        "--finetune_mle",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--finetune_rl",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--cached_train_file_mle",
        type=str,
        default=None
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=600
    )
    parser.add_argument(
        "--write_to_log",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--num_cached_responses",
        type=int,
        default=5
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=5
    )
    parser.add_argument(
        "--max_cov_score",
        type=float,
        default=0.5
    )
    parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False, default=1.0,
                        type=float)
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_dir + 'args.pkl', 'wb') as f:
        pickle.dump(args ,f)
    return args


def main():
    """
    Part0: Initialization
    """
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    device = accelerator.device
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
    """
    Part1: Prepare model
    """
    with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
        args_wiz = pickle.load(f)
        args_wiz.learning_rate = args.learning_rate
        args_wiz.experiment_type = 'chat_document'
        args_wiz.model_file_path = args.wiz_path
        args_wiz.model_name = args.wiz_model_name
    wiz = MultiBartQA(args_wiz, device)
    # apprentice
    with open(BASE_PATH + 'za/args/args_bart_train.pkl', 'rb') as f:
        args_app = pickle.load(f)
        args_app.experiment_type = 'chat_document'
        args_app.model_file_path = args.app_path
        args_app.model_name = args.app_model_name
    app = BartQA(args_app, device)
    wiz, app = accelerator.prepare(wiz, app)
    # coverage scorers
    scorer_cov = CoverageScorer()
    # coherence scorer
    with open(BASE_PATH + 'za/args/args_coh.pkl', 'rb') as f:
        args_coh = pickle.load(f)
        args_coh.model_name_or_path = args.coh_path
    scorer_coh = CoherenceScorer(args_coh, accelerator.device)
    scorers = [scorer_cov, scorer_coh]
    alphas = [args.alpha_cov, args.alpha_coh]
    trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)
    """
    Part2: Prepare data
    """
    assert args.train_file_rl is not None or args.validation_file_rl is not None
    data_files_rl = {}
    if args.train_file_rl is not None:
        data_files_rl["train"] = args.train_file_rl
    if args.validation_file_rl is not None:
        data_files_rl["validation"] = args.validation_file_rl
    extension = args.train_file_rl.split(".")[-1]
    raw_datasets_rl = load_dataset(extension, data_files=data_files_rl, field='data')
    if args.train_size is not None:
        train_dataset_rl = raw_datasets_rl['train'].select(range(args.train_size))
    else:
        train_dataset_rl = raw_datasets_rl['train']
    if args.eval_size is not None:
        eval_dataset_rl = raw_datasets_rl['validation'].select(range(args.eval_size))
    else:
        eval_dataset_rl = raw_datasets_rl['validation']
    assert args.cached_train_file_mle is not None
    if os.path.exists(args.cached_train_file_mle):
        train_dataset_mle = torch.load(args.cached_train_file_mle)
    else:
        train_dataset_mle = None
    """
    Part3: Start fine-tuning
    """
    trainer.finetune(train_dataset_rl, eval_dataset_rl, train_dataset_mle)


if __name__ == "__main__":
    main()








