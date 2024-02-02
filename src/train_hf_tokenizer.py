import sys
import yaml

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from utils.data_utils import Struct # TODO: this might break

def main(config: Struct):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        show_progress=True,
        special_tokens=["<pad>", "<bos>", "<unk>"])
    
    tokenizer.pre_tokenizer = Whitespace()

    if config.raw_dataset_path is None:
        tokenizer.train(config.raw_train_path, trainer)
    else:
        tokenizer.train(config.raw_dataset_path, trainer)

    # trim_offsets=False tells post-processor to keep spaces as part of tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A",
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>"))],
    )

    # Add decoder for converting tokens back to text
    #tokenizer.decoder = decoders.ByteLevel()

    # Enable padding
    # tokenizer.enable_padding(
    #     direction="right",
    #     pad_id=0,
    #     pad_token="<pad>",
    #     length=config.seq_len + 1)

    # Enable truncation
    # tokenizer.enable_truncation(
    #     max_length=config.seq_len + 1,
    #     direction="right")

    # Wrap tokenizer with transformers library
    # tokenizer = PreTrainedTokenizerFast(
    #     model_max_length=config.seq_len,
    #     padding_side="right",
    #     truncation_side="right",
    #     bos_token="<bos>",
    #     unk_token="<unk>",
    #     pad_token="<pad>",
    #     tokenizer_object=tokenizer)

    # # Save tokenizer to file
    # tokenizer_save_path = Path(config.tokenizer_path)
    # tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    # tokenizer.save_pretrained(tokenizer_save_path)
    tokenizer.save(config.tokenizer_path)


if __name__ == '__main__':
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    main(config)
