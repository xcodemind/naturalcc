# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc.trainer.summarization.unilang.sl_trainer import SLTrainer
from ncc.trainer.summarization.unilang.sc_trainer import SCTrainer
from ncc.trainer.summarization.unilang.pg_trainer import PGTrainer
from ncc.trainer.summarization.unilang.critic_trainer import CriticTrainer
from ncc.trainer.summarization.unilang.ac_trainer import ACTrainer
from ncc.trainer.summarization.unilang.disc_trainer import DiscTrainer
from ncc.trainer.summarization.unilang.gan_trainer import GANTrainer
from ncc.trainer.summarization.unilang.arel_trainer import ARELTrainer
from ncc.trainer.summarization.unilang.codenn_sl_trainer import CodeNNSLTrainer
from ncc.trainer.summarization.unilang.ast_attendgru_sl_trainer import AstAttendGruSLTrainer

from ncc.trainer.summarization.xlang.finetune_trainer import FTTrainer
from ncc.trainer.summarization.xlang.maml_trainer import MAMLTrainer
from ncc.trainer.summarization.xlang.dtrl_trainer import DTRLTrainer
from ncc.trainer.summarization.xlang.kd_sl_trainer import KDSLTrainer


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