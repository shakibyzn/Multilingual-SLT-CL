from copy import deepcopy

import torch
from torch.autograd import Variable
import torch.utils.data
from signjoey.batch import Batch
from signjoey.data import make_data_iter


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self,
                 trainer,
                 model,
                 batch_list):
        
        self.model = model
        self.batch_list = batch_list
        self.trainer = trainer

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data) # Previous task parameters

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
    
    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        train_iter = make_data_iter(
                self.batch_list,
                batch_size=self.trainer.batch_size,
                batch_type=self.trainer.batch_type,
                train=False,
                shuffle=self.trainer.shuffle,
            )
        
        total_batch_list_len = 0
        for torch_batch in iter(train_iter):
            batch = Batch(
                lang=self.trainer.languages,
                sign_lang_vocab=self.model.sign_lang_vocab,
                is_train=False,
                torch_batch=torch_batch,
                txt_pad_index=self.trainer.txt_pad_index,
                sgn_dim=self.trainer.feature_size,
                use_cuda=self.trainer.use_cuda,
                frame_subsampling_ratio=None,
                random_frame_subsampling=None,
                random_frame_masking_ratio=None,
            )
            total_batch_list_len += batch.num_seqs
            
        self.model.eval()
        for torch_batch in iter(train_iter):
            
            batch = Batch(
                lang=self.trainer.languages,
                sign_lang_vocab=self.model.sign_lang_vocab,
                is_train=False,
                torch_batch=torch_batch,
                txt_pad_index=self.trainer.txt_pad_index,
                sgn_dim=self.trainer.feature_size,
                use_cuda=self.trainer.use_cuda,
                frame_subsampling_ratio=None,
                random_frame_subsampling=None,
                random_frame_masking_ratio=None,
            )
            
            self.model.zero_grad()
            
            loss = self._train_batch(batch)

            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / total_batch_list_len
                else:
                    print("n:", n)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        
        return loss