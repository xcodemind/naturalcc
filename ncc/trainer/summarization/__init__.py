# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc.trainer.summarization.sl_trainer import SLTrainer
from ncc.trainer.summarization.sc_trainer import SCTrainer
from ncc.trainer.summarization.pg_trainer import PGTrainer
from ncc.trainer.summarization.critic_trainer import CriticTrainer
from ncc.trainer.summarization.ac_trainer import ACTrainer
from ncc.trainer.summarization.disc_trainer import DiscTrainer
from ncc.trainer.summarization.gan_trainer import GANTrainer
from ncc.trainer.summarization.arel_trainer import ARELTrainer
from ncc.trainer.summarization.codenn_sl_trainer import CodeNNSLTrainer
from ncc.trainer.summarization.ast_attendgru_sl_trainer import AstAttendGruSLTrainer

from ncc.trainer.summarization.finetune_trainer import FTTrainer
from ncc.trainer.summarization.maml_trainer import MAMLTrainer
from ncc.trainer.summarization.dtrl_trainer import DTRLTrainer
from ncc.trainer.summarization.kd_sl_trainer import KDSLTrainer


__all__ = [
    'SLTrainer',
    'SCTrainer',
    'PGTrainer',
    'CriticTrainer',
    'ACTrainer',
    'DiscTrainer',
    'GANTrainer',
    'ARELTrainer',
    'CodeNNSLTrainer',
    'AstAttendGruSLTrainer',
    'FTTrainer',
    'MAMLTrainer',
    'DTRLTrainer',
    'KDSLTrainer',
]