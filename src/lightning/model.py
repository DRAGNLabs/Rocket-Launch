from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from transformers import (
    LlamaForCausalLM as LanguageModel, 
    LlamaConfig as HFConfig
)

# Use a lower precision for better performance
torch.set_float32_matmul_precision('medium')

class Model(LightningModule):
    def __init__(self,
                 tokenizer, 
                 config: dict = None):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        if config.from_pretrained is not True:
            # * Configure necessary HF model parameters here
            model_config = HFConfig(
                vocab_size = config.vocab_size,
                max_position_embeddings = config.max_sequence_embeddings,
                hidden_size=config.dim,
                num_hidden_layers=config.n_layers,
                num_attention_heads=config.n_heads,
                rms_norm_eps=config.norm_eps,
                pad_token_id=config.pad_id
            )
            self.model = LanguageModel(model_config)
        elif config.from_pretrained is True and config.model_name is not None:
            self.model = LanguageModel.from_pretrained(config.model_name)
        else:
            raise ValueError("Must provide model_name if from_pretrained is True")
        
        self.validation_step_outputs = [] # Used for saving predictions throughout training

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        x, x_mask, y_true = batch

        output = self.model(input_ids=x, attention_mask=x_mask, labels=y_true)

        loss = output.loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_mask, y_true = batch

        output = self.model(input_ids=x, attention_mask=x_mask, labels=y_true)
        val_loss = output.loss
        y_hat = output.logits

        if self.config.save_predictions_during_training:
            # Decode predictions and add to valuation predictions list
            probs = torch.softmax(y_hat, dim=2)
            preds = torch.argmax(probs, 2).detach().cpu().tolist()

            #y_true_decoded = self.tokenizer.decode(y_true[0].tolist())
            decoded = self.tokenizer.decode(preds[0])

            self.validation_step_outputs.append(decoded)

        perplexity = torch.exp(val_loss)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, x_mask, y_true = batch

        output = self.model(input_ids=x, attention_mask=x_mask, labels=y_true)

        loss = output.loss

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.config.save_predictions_during_training == True:
            dir_path = Path(self.config.default_root_dir)
            file_path = dir_path / 'validation_predictions.txt'

            # Check if the directory exists. If not, create it
            dir_path.mkdir(parents=True, exist_ok=True)

            # Check if the file exists. If not, create it and append the outputs
            with file_path.open('a', encoding="utf-8") as f:
                for item in self.validation_step_outputs:
                    f.write(str(self.current_epoch) + ': ')
                    f.write(str(item) + '\n')

            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)  # model.paramaters = weights tensor
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]
