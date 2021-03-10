import time
import numpy
from os.path import join
from os.path import exists

import torch

import nmt.all_constants as ac
import nmt.utils as ut
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations
from nmt.validator import Validator

if torch.cuda.is_available():
    torch.cuda.manual_seed(ac.SEED)
else:
    torch.manual_seed(ac.SEED)


class Trainer(object):
    """Trainer"""
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.config = getattr(configurations, args.proto)()
        self.num_preload = args.num_preload
        self.logger = ut.get_logger(self.config['log_file'])

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.normalize_loss = self.config['normalize_loss']
        self.patience = self.config['patience']
        self.lr = self.config['lr']
        self.lr_decay = self.config['lr_decay']
        self.max_epochs = self.config['max_epochs']
        self.warmup_steps = self.config['warmup_steps']

        self.train_smooth_perps = []
        self.train_true_perps = []

        self.data_manager = DataManager(self.config)
        self.validator = Validator(self.config, self.data_manager)

        self.val_per_epoch = self.config['val_per_epoch']
        self.validate_freq = int(self.config['validate_freq'])
        self.logger.info('Evaluate every {} {}'.format(self.validate_freq, 'epochs' if self.val_per_epoch else 'batches'))

        # For logging
        self.log_freq = 100  # log train stat every this-many batches
        self.log_train_loss = 0. # total train loss every log_freq batches
        self.log_nll_loss = 0.
        self.log_train_weights = 0.
        self.num_batches_done = 0 # number of batches done for the whole training
        self.epoch_batches_done = 0 # number of batches done for this epoch
        self.epoch_loss = 0. # total train loss for whole epoch
        self.epoch_nll_loss = 0. # total train loss for whole epoch
        self.epoch_weights = 0. # total train weights (# target words) for whole epoch
        self.epoch_time = 0. # total exec time for whole epoch, sounds like that tabloid

        # get model
        self.model = Model(self.config).to(self.device)
        if args.model_file is not None:
            self.logger.info('Restore model from {}'.format(args.model_file))
            self.model.load_state_dict(torch.load(args.model_file))

        param_count = sum([numpy.prod(p.size()) for p in self.model.parameters()])
        self.logger.info('Model has {:,} parameters'.format(param_count))

        # get optimizer
        beta1 = self.config['beta1']
        beta2 = self.config['beta2']
        epsilon = self.config['epsilon']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(beta1, beta2), eps=epsilon)

    def report_epoch(self, e):
        self.logger.info('Finish epoch {}'.format(e))
        self.logger.info('    It takes {}'.format(ut.format_seconds(self.epoch_time)))
        self.logger.info('    Avergage # words/second    {}'.format(self.epoch_weights / self.epoch_time))
        self.logger.info('    Average seconds/batch    {}'.format(self.epoch_time / self.epoch_batches_done))

        train_smooth_perp = self.epoch_loss / self.epoch_weights
        train_true_perp = self.epoch_nll_loss / self.epoch_weights

        self.epoch_batches_done = 0
        self.epoch_time = 0.
        self.epoch_nll_loss = 0.
        self.epoch_loss = 0.
        self.epoch_weights = 0.

        train_smooth_perp = numpy.exp(train_smooth_perp) if train_smooth_perp < 300 else float('inf')
        self.train_smooth_perps.append(train_smooth_perp)
        train_true_perp = numpy.exp(train_true_perp) if train_true_perp < 300 else float('inf')
        self.train_true_perps.append(train_true_perp)

        self.logger.info('    smoothed train perplexity: {}'.format(train_smooth_perp))
        self.logger.info('    true train perplexity: {}'.format(train_true_perp))

    def run_log(self, b, e, batch_data):
        start = time.time()
        src_toks, trg_toks, targets = batch_data
        src_toks_cuda = src_toks.to(self.device)
        trg_toks_cuda = trg_toks.to(self.device)
        targets_cuda = targets.to(self.device)

        # zero grad
        self.optimizer.zero_grad()

        # get loss
        ret = self.model(src_toks_cuda, trg_toks_cuda, targets_cuda)
        loss = ret['loss']
        nll_loss = ret['nll_loss']
        if self.normalize_loss == ac.LOSS_TOK:
            opt_loss = loss / (targets_cuda != ac.PAD_ID).type(loss.type()).sum()
        elif self.normalize_loss == ac.LOSS_BATCH:
            opt_loss = loss / targets_cuda.size()[0].type(loss.type())
        else:
            opt_loss = loss

        opt_loss.backward()
        # clip gradient
        global_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

        # update
        self.adjust_lr()
        self.optimizer.step()

        # update training stats
        num_words = (targets != ac.PAD_ID).detach().numpy().sum()

        loss = loss.cpu().detach().numpy()
        nll_loss = nll_loss.cpu().detach().numpy()
        self.num_batches_done += 1
        self.log_train_loss += loss
        self.log_nll_loss += nll_loss
        self.log_train_weights += num_words

        self.epoch_batches_done += 1
        self.epoch_loss += loss
        self.epoch_nll_loss += nll_loss
        self.epoch_weights += num_words
        self.epoch_time += time.time() - start

        if self.num_batches_done % self.log_freq == 0:
            acc_speed_word = self.epoch_weights / self.epoch_time
            acc_speed_time = self.epoch_time / self.epoch_batches_done

            avg_smooth_perp = self.log_train_loss / self.log_train_weights
            avg_smooth_perp = numpy.exp(avg_smooth_perp) if avg_smooth_perp < 300 else float('inf')
            avg_true_perp = self.log_nll_loss / self.log_train_weights
            avg_true_perp = numpy.exp(avg_true_perp) if avg_true_perp < 300 else float('inf')

            self.log_train_loss = 0.
            self.log_nll_loss = 0.
            self.log_train_weights = 0.

            self.logger.info('Batch {}, epoch {}/{}:'.format(b, e + 1, self.max_epochs))
            self.logger.info('   avg smooth perp:   {0:.2f}'.format(avg_smooth_perp))
            self.logger.info('   avg true perp:   {0:.2f}'.format(avg_true_perp))
            self.logger.info('   acc trg words/s: {}'.format(int(acc_speed_word)))
            self.logger.info('   acc sec/batch:   {0:.2f}'.format(acc_speed_time))
            self.logger.info('   global norm:     {0:.2f}'.format(global_norm))

    def run_discriminative(self, src_toks, trg_toks, targets, num_samples):
        #torch.autograd.set_detect_anomaly(True)

        start = time.time()

        src_toks_cuda = src_toks.to(self.device)
        trg_toks_cuda = trg_toks.to(self.device)
        targets_cuda = targets.to(self.device)

        self.logger.info("Discriminative phase")

        # Positive example
        gold_nll = self.model(src_toks_cuda, trg_toks_cuda, targets_cuda)['loss']
        gold_lr = self.model.length_reward*(targets_cuda != ac.PAD_ID).sum()
        self.logger.info("    gold: {} (score={})".format(
            self.validator._ids_to_trans(targets_cuda[0].tolist()),
            -gold_nll+gold_lr))

        # Generate negative samples

        samples = self.model.beam_decode(src_toks_cuda, mode='random_particle', beam_size=num_samples)
        # If the samples aren't weighted, supply uniform weights
        for sample in samples:
            if 'weights' not in sample:
                sample['weights'] = torch.ones(num_samples, device=sample['scores'].device).float() / num_samples
        assert len(samples) == 1
        samples = samples[0]

        sample_lengths = (samples['symbols'] != ac.EOS_ID).sum(dim=-1) + 1
        for i in range(num_samples):
            self.logger.info('    sample: {} (weight={}, score={})'.format(
                self.validator._ids_to_trans(samples['symbols'][i].tolist()), 
                samples['weights'][i], 
                samples['scores'][i]))

        loss = gold_nll - gold_lr + (samples['scores']*samples['weights']).sum()

        if self.normalize_loss == ac.LOSS_TOK:
            opt_loss = loss / (targets_cuda != ac.PAD_ID).type(loss.type()).sum()
        elif self.normalize_loss == ac.LOSS_BATCH:
            opt_loss = loss / targets_cuda.size()[0].type(loss.type())
        else:
            opt_loss = loss

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
        self.adjust_lr()
        self.optimizer.step()

        self.logger.info('   time:               {}'.format(time.time() - start))
        self.logger.info('   length reward grad: {}'.format(self.model.length_reward.grad))
        self.logger.info('   length reward:      {}'.format(self.model.length_reward.item()))
        #self.logger.info('   bad weights:     {}'.format(self.model.bad_affine.weight))
        self.logger.info('   good bias grad:     {}'.format(self.model.bad_affine.bias.grad))
        self.logger.info('   good bias:          {}'.format(self.model.bad_affine.bias.item()))

    def adjust_lr(self):
        if self.config['warmup_style'] == ac.ORG_WARMUP:
            step = self.num_batches_done + 1.0
            if step < self.config['warmup_steps']:
                lr = self.config['embed_dim'] ** (-0.5) * step * self.config['warmup_steps'] ** (-1.5)
            else:
                lr = max(self.config['embed_dim'] ** (-0.5) * step ** (-0.5), self.config['min_lr'])
            for p in self.optimizer.param_groups:
                p['lr'] = lr
        elif self.config['warmup_style'] == ac.FIXED_WARMUP:
            warmup_steps = self.config['warmup_steps']
            step = self.num_batches_done + 1.0
            start_lr = self.config['start_lr']
            peak_lr = self.config['lr']
            min_lr = self.config['min_lr']
            if step < warmup_steps:
                lr = start_lr + (peak_lr - start_lr) * step / warmup_steps
            else:
                lr = max(min_lr, peak_lr * warmup_steps ** (0.5) * step ** (-0.5))
            for p in self.optimizer.param_groups:
                p['lr'] = lr
        elif self.config['warmup_style'] == ac.UPFLAT_WARMUP:
            warmup_steps = self.config['warmup_steps']
            step = self.num_batches_done + 1.0
            start_lr = self.config['start_lr']
            peak_lr = self.config['lr']
            min_lr = self.config['min_lr']
            if step < warmup_steps:
                lr = start_lr + (peak_lr - start_lr) * step / warmup_steps
                for p in self.optimizer.param_groups:
                    p['lr'] = lr
        else:
            pass

    def train(self):
        self.model.train()
        #for e in range(self.max_epochs):
        for e in range(0):
            b = 0
            for batch_data in self.data_manager.get_batch(mode=ac.TRAINING, num_preload=self.num_preload):
                b += 1
                self.run_log(b, e, batch_data)
                if not self.val_per_epoch:
                    self.maybe_validate()

            self.report_epoch(e + 1)
            if self.val_per_epoch and (e + 1) % self.validate_freq == 0:
                self.maybe_validate(just_validate=True)

        """for p in self.model.parameters():
            if p is self.model.length_reward:
                self.logger.info("not freezing length reward")
            elif p is self.model.bad_affine.weight:
                self.logger.info("not freezing p(BAD) weights")
            elif p is self.model.bad_affine.bias:
                self.logger.info("not freezing p(BAD) bias")
            else:
                p.requires_grad = False"""

        self.logger.info('It is finally done, mate!')
        self.logger.info('Train smoothed perps:')
        self.logger.info(', '.join(map(str, self.train_smooth_perps)))
        self.logger.info('Train true perps:')
        self.logger.info(', '.join(map(str, self.train_true_perps)))
        numpy.save(join(self.config['save_to'], 'train_smooth_perps.npy'), self.train_smooth_perps)
        numpy.save(join(self.config['save_to'], 'train_true_perps.npy'), self.train_true_perps)

        self.logger.info('Save final checkpoint')
        self.save_checkpoint()

        d = 1./self.model.bad_affine.weight.size(1)**0.5
        torch.nn.init.uniform_(self.model.bad_affine.weight, -d, d)
        torch.nn.init.uniform_(self.model.bad_affine.bias, -d, d)

        for e in range(10):
            #for b, batch_data in enumerate(self.data_manager.get_batch(mode=ac.TRAINING, num_preload=self.num_preload)):
            for b, batch_data in enumerate(self.data_manager.get_batch(mode=ac.VALIDATING, num_preload=self.num_preload)):
                src_toks, trg_toks, targets = batch_data
                bsz = src_toks.size()[0]
                for i in range(bsz):
                    self.run_discriminative(src_toks[i:i+1], 
                                            trg_toks[i:i+1], 
                                            targets[i:i+1], max(1,bsz//5))
                if (b+1) % 10 == 0:
                    self.maybe_validate(just_validate=True)
        self.maybe_validate(just_validate=True)

        # Evaluate on test
        test_file = self.data_manager.data_files[ac.TESTING][self.data_manager.src_lang]
        if exists(test_file):
            self.logger.info('Evaluate on test')
            self.restart_to_best_checkpoint()
            self.validator.translate(self.model, test_file)
            self.logger.info('Also translate dev set')
            self.validator.translate(self.model, self.data_manager.data_files[ac.VALIDATING][self.data_manager.src_lang])

    def save_checkpoint(self):
        cpkt_path = join(self.config['save_to'], '{}.pth'.format(self.config['model_name']))
        torch.save(self.model.state_dict(), cpkt_path)

    def restart_to_best_checkpoint(self):
        if self.config['val_by_bleu']:
            best_bleu = numpy.max(self.validator.best_bleus)
            best_cpkt_path = self.validator.get_cpkt_path(best_bleu)
        else:
            best_perp = numpy.min(self.validator.best_perps)
            best_cpkt_path = self.validator.get_cpkt_path(best_perp)

        self.logger.info('Restore best cpkt from {}'.format(best_cpkt_path))
        self.model.load_state_dict(torch.load(best_cpkt_path))

    def maybe_validate(self, just_validate=False):
        if self.num_batches_done % self.validate_freq == 0 or just_validate:
            self.save_checkpoint()
            self.validator.validate_and_save(self.model)

            # if doing annealing
            step = self.num_batches_done + 1.0
            warmup_steps = self.config['warmup_steps']
            if self.config['warmup_style'] == ac.NO_WARMUP or (self.config['warmup_style'] == ac.UPFLAT_WARMUP and step >= warmup_steps) and self.lr_decay > 0:
                if self.config['val_by_bleu']:
                    cond = len(self.validator.bleu_curve) > self.patience and self.validator.bleu_curve[-1] < min(self.validator.bleu_curve[-1 - self.patience:-1])
                else:
                    cond = len(self.validator.perp_curve) > self.patience and self.validator.perp_curve[-1] > max(self.validator.perp_curve[-1 - self.patience:-1])

                if cond:
                    if self.config['val_by_bleu']:
                        metric = 'bleu'
                        scores = self.validator.bleu_curve[-1 - self.patience:]
                        scores = map(str, list(scores))
                        scores = ', '.join(scores)
                    else:
                        metric = 'perp'
                        scores = self.validator.perp_curve[-1 - self.patience:]
                        scores = map(str, list(scores))
                        scores = ', '.join(scores)

                    self.logger.info('Past {} are {}'.format(metric, scores))
                    # when don't use warmup, decay lr if dev not improve
                    if self.lr * self.lr_decay >= self.config['min_lr']:
                        self.logger.info('Anneal the learning rate from {} to {}'.format(self.lr, self.lr * self.lr_decay))
                        self.lr = self.lr * self.lr_decay
                        for p in self.optimizer.param_groups:
                            p['lr'] = self.lr
