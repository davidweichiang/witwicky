from __future__ import print_function
from __future__ import division

import os
import nmt.all_constants as ac


def az2en():
    config = {}

    config['model_name']        = 'az2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'az'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/az2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 1
    config['num_enc_heads']     = 4
    config['num_dec_layers']    = 1
    config['num_dec_heads']     = 4
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 1024
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 300
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def be2en():
    config = {}

    config['model_name']        = 'be2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'be'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/be2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 1
    config['num_enc_heads']     = 4
    config['num_dec_layers']    = 1
    config['num_dec_heads']     = 4
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 1024
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 300
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.TRG_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def gl2en():
    config = {}

    config['model_name']        = 'gl2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'gl'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/gl2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 1
    config['num_enc_heads']     = 4
    config['num_dec_layers']    = 1
    config['num_dec_heads']     = 4
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 1024
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 300
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def sk2en():
    config = {}

    config['model_name']        = 'sk2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'sk'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/sk2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2vi():
    config = {}

    config['model_name']        = 'en2vi'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'vi'
    config['data_dir']          = './nmt/data/en2vi'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def ar2en():
    config = {}

    config['model_name']        = 'ar2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'ar'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/ar2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.TRG_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def de2en():
    config = {}

    config['model_name']        = 'de2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'de'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/de2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def he2en():
    config = {}

    config['model_name']        = 'he2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'he'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/he2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.TRG_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def it2en():
    config = {}

    config['model_name']        = 'it2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'it'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/it2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2ja():
    config = {}

    config['model_name']        = 'en2ja'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'ja'
    config['data_dir']          = './nmt/data/en2ja'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 48200
    config['trg_vocab_size']    = 49100
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.TRG_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def wmt14_en2de():
    config = {}

    config['model_name']        = 'wmt14_en2de'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'de'
    config['data_dir']          = './nmt/data/en2de'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.1
    config['word_dropout']      = 0.0
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 24
    config['validate_freq']     = 20000
    config['val_per_epoch']     = False # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 32768
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = True
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2az():
    config = {}

    config['model_name']        = 'en2az'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'az'
    config['data_dir']          = './nmt/data/en2az'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 1
    config['num_enc_heads']     = 4
    config['num_dec_layers']    = 1
    config['num_dec_heads']     = 4
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 1024
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 300
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2be():
    config = {}

    config['model_name']        = 'en2be'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'be'
    config['data_dir']          = './nmt/data/en2be'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 1
    config['num_enc_heads']     = 4
    config['num_dec_layers']    = 1
    config['num_dec_heads']     = 4
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 1024
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 300
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.TRG_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2gl():
    config = {}

    config['model_name']        = 'en2gl'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'gl'
    config['data_dir']          = './nmt/data/en2gl'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 1
    config['num_enc_heads']     = 4
    config['num_dec_layers']    = 1
    config['num_dec_heads']     = 4
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 1024
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 300
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2sk():
    config = {}

    config['model_name']        = 'en2sk'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'sk'
    config['data_dir']          = './nmt/data/en2sk'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2ar():
    config = {}

    config['model_name']        = 'en2ar'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'ar'
    config['data_dir']          = './nmt/data/en2ar'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.TRG_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2de():
    config = {}

    config['model_name']        = 'en2de'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'de'
    config['data_dir']          = './nmt/data/en2de'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2he():
    config = {}

    config['model_name']        = 'en2he'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'he'
    config['data_dir']          = './nmt/data/en2he'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.TRG_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def en2it():
    config = {}

    config['model_name']        = 'en2it'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'it'
    config['data_dir']          = './nmt/data/en2it'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def ha2en():
    config = {}

    config['model_name']        = 'ha2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'ha'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/ha2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def tu2en():
    config = {}

    config['model_name']        = 'tu2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'tu'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/tu2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def uz2en():
    config = {}

    config['model_name']        = 'uz2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'uz'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/uz2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config


def hu2en():
    config = {}

    config['model_name']        = 'hu2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'hu'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/hu2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['fix_norm']          = False
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['norm_in']           = True # if False, dropout->add->norm (orgpaper), else norm->dropout->add
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['max_train_length']  = 1000 # actually can go to 1023 (length + special token eos/bos)
    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['embed_init_range']  = 0.01
    config['embed_init_type']   = ac.EMBED_NORMAL
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True
    config['label_smoothing']   = 0.1
    config['restore_segments']  = True
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = False
    config['reload']            = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config