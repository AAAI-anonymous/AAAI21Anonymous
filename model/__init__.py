
# MASS
from . import transformer_mass


# K-PLUG
from . import transformer_kplug, matchgo_task, fairseq_task_patch, multitask_s2s_kplug, transformer
#from .tasks import multitask_s2s_kplug, sequence_tagging, sentence_prediction_bert
from .criterions import auto_criterion, sequence_tagging, label_smoothed_cross_entropy_rl, rougescore

# attention
from . import multihead_attention, transformer_layer

