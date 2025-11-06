
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gsa2.configuration_gsa2 import GSA2Config
from fla.models.gsa2.modeling_gsa2 import GSA2ForCausalLM, GSA2Model

AutoConfig.register(GSA2Config.model_type, GSA2Config, exist_ok=True)
AutoModel.register(GSA2Config, GSA2Model, exist_ok=True)
AutoModelForCausalLM.register(GSA2Config, GSA2ForCausalLM, exist_ok=True)


__all__ = ['GSA2Config', 'GSA2ForCausalLM', 'GSA2Model']
