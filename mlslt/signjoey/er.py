import random
from torchtext import data
from typing import List
from signjoey.batch import Batch
from signjoey.data import make_data_iter


class ExperienceReplay:
    def __init__(self, trainer, model, train_data, languages: List[str], buffer_size: int = 100) -> None:
        self.trainer = trainer
        self.model = model
        self.train_data = train_data
        temp_data, self.memory = {}, {}
        for lang in languages:
            temp_data[lang], self.memory[lang] = [], []
            for row in train_data:
                if row.sign_lang[0] == lang:
                    temp_data[lang].append(row)
            
            # Create a memory buffer for each language
            self.memory[lang] = random.sample(temp_data[lang], buffer_size)
            trainer.logger.info(f"Memory buffer for {lang} created with {len(self.memory[lang])} samples")
            
    
    def sample(self, batch_size):
        lang_wise_samples = {}
        for lang in self.memory:
            if len(self.memory[lang]) < batch_size:
                raise ValueError("Memory is not enough to sample")
            
            curr_sample = random.sample(self.memory[lang], batch_size)
            lang_wise_samples[lang] = data.Dataset(curr_sample, fields=self.train_data.fields)
        
        return lang_wise_samples

    def replay(self, batch_size):
        if len(self.memory) == 0:
            raise ValueError("Memory is empty")
        
        replay_batches = self.sample(batch_size)
        total_loss = 0
        for lang in replay_batches:
            batch_iter = make_data_iter(replay_batches[lang], self.trainer.batch_size, self.trainer.batch_type, train=True, shuffle=True)
            for torch_batch in batch_iter:
                batch = Batch(
                    lang=[lang],
                    sign_lang_vocab=self.model.sign_lang_vocab,
                    is_train=True,
                    torch_batch=torch_batch,
                    txt_pad_index=self.trainer.txt_pad_index,
                    sgn_dim=self.trainer.feature_size,
                    use_cuda=self.trainer.use_cuda,
                    frame_subsampling_ratio=self.trainer.frame_subsampling_ratio,
                    random_frame_subsampling=self.trainer.random_frame_subsampling,
                    random_frame_masking_ratio=self.trainer.random_frame_masking_ratio,
                )

                total_loss += self._train_batch(batch)
        
        return total_loss
    
    
    def _train_batch(self, batch):
        
        recognition_loss, translation_loss = self.model.get_loss_for_batch(
            batch=batch,
            recognition_loss_function=None,
            translation_loss_function=self.trainer.translation_loss_function,
            recognition_loss_weight=None,
            translation_loss_weight=self.trainer.translation_loss_weight
        )

        # normalize translation loss
        if self.trainer.do_translation:
            if self.trainer.translation_normalization_mode == "batch":
                txt_normalization_factor = batch.num_seqs
            elif self.trainer.translation_normalization_mode == "tokens":
                txt_normalization_factor = batch.num_txt_tokens
            else:
                raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

            # division needed since loss.backward sums the gradients until updated
            normalized_translation_loss = translation_loss / (
                txt_normalization_factor * self.trainer.batch_multiplier
            )
        else:
            normalized_translation_loss = 0

        if self.trainer.do_recognition:
            normalized_recognition_loss = recognition_loss / self.trainer.batch_multiplier
        else:
            normalized_recognition_loss = 0

        total_loss = normalized_recognition_loss + normalized_translation_loss
        return total_loss
