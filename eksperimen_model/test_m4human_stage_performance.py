"""Offline bounded physical and frozen-language profiling contracts; no downloads."""
import copy
import contextlib
import io
import json
import sys
from unittest.mock import patch
import yaml
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from eksperimen_model import m4human_training as physical
from eksperimen_model.profile_m4human_stages import profile_stage,profile_generation,_physical_dataset
from eksperimen_model.test_m4human_integration import fixture
from eksperimen_model.test_m4human_language_contract import TinyTokenizer,TinyCausalLM
from eksperimen_model.models.m4human_projector import FrozenM4HumanLanguage
from eksperimen_model.datasets.m4human_qa_dataset import canonical_answer
from eksperimen_model.utils.m4human_kinematics import TASK_DEFINITIONS
from eksperimen_model.utils.m4human_runtime import seed_everything,state_dict_hash


class GeneratingTinyLM(TinyCausalLM):
    @torch.no_grad()
    def generate(self,inputs_embeds,attention_mask,max_new_tokens,**kwargs):
        positions=(attention_mask.cumsum(-1)-1).clamp_min(0).masked_fill(attention_mask==0,0)
        logits=self.forward(inputs_embeds,attention_mask,positions).logits[:,-1]
        return SimpleNamespace(logits=[logits],sequences=torch.ones(len(inputs_embeds),1,dtype=torch.long))


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        config=fixture(Path(directory))
        config.update(micro_batch=1,accumulation=1,effective_batch=1,workers=0,cpu_threads=1,ram_reserve_mib=0,vram_reserve_mib=0,resource_poll_ms=50)
        dataset,stats=_physical_dataset(config,'M',None)
        def motion_builder(precision):
            seed_everything(config['seed']);return physical._models('M',stats,config)
        result=profile_stage(dataset,motion_builder,'M',None,config,stats,'cpu',1,1)
        assert result['samples']==1 and result['successful_updates']==1 and result['loaded_samples_bound']==2
        assert result['numerical_parity']['passed'] and not result['numerical_parity']['cuda_measured']
        assert result['median_logical_update_seconds']==result['p95_logical_update_seconds']>0
        assert not result['sustained_resource_profile'] and not result['scientific_gates_passed']
        # Public CLI persists explicit protocol metadata and never a checkpoint.
        import eksperimen_model.profile_m4human_stages as profiler
        config_file=Path(directory)/'profile.yaml'; report_file=Path(directory)/'profile.json'; selected_file=Path(directory)/'selected.yaml'
        config_file.write_text(yaml.safe_dump(config))
        arguments=['profile','--stage','M','--config',str(config_file),'--output',str(report_file),'--selected-config',str(selected_file),
                   '--device','cpu','--micro-batches','1','--workers','0','--precisions','fp32','--warmup-steps','1','--timed-steps','1','--synthetic-smoke']
        with patch.object(sys,'argv',arguments),contextlib.redirect_stdout(io.StringIO()):profiler.main()
        report=json.loads(report_file.read_text());selected=yaml.safe_load(selected_file.read_text())
        assert report['recommended_settings']['effective_batch']==1 and report['screening_only']
        assert report['scientific_gates_passed'] is False and selected['performance_profile_policy']['batch_policy_changed'] is False
        assert not list(Path(directory).glob('*.pt'))
        # Physical trainer locks default AdamW decay; stray config field must not change profiling.
        base=copy.deepcopy(config);base['weight_decay']=.9;recorded=[]
        native_adamw=torch.optim.AdamW
        def record_optimizer(groups,**kwargs):
            grouped=list(groups);recorded.extend(group['weight_decay'] for group in grouped)
            return native_adamw(grouped,**kwargs)
        with patch.object(profiler.torch.optim,'AdamW',side_effect=record_optimizer):
            profile_stage(dataset,motion_builder,'M',None,base,stats,'cpu',1,1)
        assert set(recorded)=={0.,.01}
        samples=[]
        for i in range(2):
            sample=copy.deepcopy(dataset[i]); sample['sensor'].update(H=torch.randn(32,23,128),latent_valid=torch.ones(32,23,dtype=torch.bool),common_frame_valid=torch.ones(32,dtype=torch.bool),U=torch.randn(16,256),token_mask=torch.ones(16,dtype=torch.bool))
            samples.append(sample)
        for stage,condition in [('C','C_base'),('C','C_kin'),('probe','M'),('probe','U_base')]:
            scales=physical._fit_statistics(samples,stage,0)
            def builder(precision):
                seed_everything(config['seed']);return physical._models(stage,scales,config)
            profiled=profile_stage(samples,builder,stage,condition,config,scales,'cpu',1,1)
            assert profiled['numerical_parity']['passed'] and profiled['overflow_skips']==0 and profiled['samples']==1
    settings={'micro_batch':1,'accumulation':1,'effective_batch':1,'workers':0,'cpu_threads':1,'seed':42,
              'precision':'fp32','learning_rate':.001,'clip_grad_norm':1.,'weight_decay':.01,
              'ram_reserve_mib':0,'vram_reserve_mib':0,'resource_poll_ms':50}
    language_config={'training':settings}
    sample={'U':torch.ones(16,256),'token_mask':torch.ones(16,dtype=torch.bool),'question':'Physical question?',
            'task':'root_speed_trend','answers':TASK_DEFINITIONS['root_speed_trend']['answers'],
            'canonical_answer':canonical_answer('root_speed_trend','speeding_up')}
    def language_builder(precision):
        seed_everything(42);return FrozenM4HumanLanguage(GeneratingTinyLM(),TinyTokenizer(),max_prefix_tokens=1024,max_total_tokens=1200)
    language=profile_stage([sample,sample],language_builder,'L','C_base',language_config,None,'cpu',1,1)
    assert language['frozen_weights_verified'] and language['numerical_parity']['passed'] and language['workers_active']==0
    generation=profile_generation([sample,sample],language_builder,settings,'cpu',1,1,1,2)
    assert generation['samples']==1 and generation['generated_tokens']==1 and generation['frozen_weights_verified']
    # CLI uses the real generation config namespace and refuses unusable L exports.
    from eksperimen_model.utils.m4human_runtime import save_checkpoint
    class FixtureQA(list):
        metadata={'data_kind':'m4human','scientific_eligible':True,'K':16}
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary); initial=folder/'initial.pt'; config_file=folder/'language.yaml'
        config={'llm':{'revision':'a'*40},'lineage':{},'primary_budget':16,'training':dict(settings,max_prefix_tokens=1024,max_total_tokens=1200),
                'generation':{'max_new_tokens':2}}
        save_checkpoint(initial,language_builder('fp32').projector,dict(llm_revision='a'*40,seed=42,role='paired_initial'))
        config_file.write_text(yaml.safe_dump(config))
        for generation_valid in (True,False):
            output=folder/('valid.json' if generation_valid else 'rejected.json');selected=folder/('valid.yaml' if generation_valid else 'rejected.yaml')
            arguments=['profile','--stage','L','--condition','C_base','--config',str(config_file),'--output',str(output),'--selected-config',str(selected),
                       '--device','cpu','--qa','fixture_qa','--u-cache','fixture_cache','--initial',str(initial),'--micro-batches','1',
                       '--workers','0','--precisions','fp32','--warmup-steps','1','--timed-steps','1','--generation-batches','1']
            with patch('eksperimen_model.datasets.m4human_qa_dataset.M4HumanQADataset',return_value=FixtureQA([sample,sample])),patch('eksperimen_model.models.m4human_projector.load_frozen_qwen',side_effect=lambda *args:language_builder('fp32')),patch.object(sys,'argv',arguments),contextlib.redirect_stdout(io.StringIO()):
                if generation_valid:profiler.main()
                else:
                    with patch.object(profiler,'profile_generation',side_effect=ValueError('fixture generation OOM rejection')):profiler.main()
            report=json.loads(output.read_text())
            if generation_valid:
                resolved=yaml.safe_load(selected.read_text())
                assert resolved['generation']['batch_size']==1 and resolved['generation']['max_new_tokens']==2
                assert 'generation_batch_size' not in resolved['training'] and report['selection_status']=='selected'
            else:
                assert not selected.exists() and report['selection_status']=='no_valid_generation_candidate'
    bad=copy.deepcopy(language_config);bad['training']['workers']=1
    try:profile_stage([sample,sample],language_builder,'L','C_base',bad,None,'cpu',1,1)
    except ValueError:pass
    else:raise AssertionError('inactive L workers accepted')
    print('PASS bounded M/C/probe and frozen tiny-language train/generation profiles; CUDA and real Qwen not measured')

if __name__=='__main__':main()
