"""One FP16 greedy request path. E014-specific validation is separate from GEMM."""
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
from torch.nn.attention import sdpa_kernel, SDPBackend

from fetch import sha, MODEL, REVISION

GENERATED = 64


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def mark(name, enabled):
    return torch.profiler.record_function(name) if enabled else nullcontext()


class Workload:
    def __init__(self, result):
        self.result = result
        self.descriptor = json.loads((result/'model.json').read_text())
        self.path = Path(self.descriptor['path'])
        assert self.descriptor['model_id'] == MODEL and self.descriptor['revision'] == REVISION
        for name,info in self.descriptor['files'].items():
            assert sha(self.path/name) == info['sha256'],name
        assert os.environ['OMPI_COMM_WORLD_SIZE'] == '1'
        assert os.environ['OMPI_COMM_WORLD_RANK'] == '0'
        assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 5060 Ti'
        assert torch.cuda.get_device_capability() == (12,0)
        torch.manual_seed(1)
        torch.set_num_threads(16)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        # One supported fused backend for every cache path. Never silently use math.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path,local_files_only=True,trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.path,local_files_only=True,
            trust_remote_code=False,dtype=torch.float16,attn_implementation='sdpa').eval().to('cuda')
        torch.cuda.synchronize()
        self.load_s = time.perf_counter()-start
        assert all(p.dtype == torch.float16 and p.device.type == 'cuda' for p in self.model.parameters())
        assert not hasattr(self.model.forward,'_torchdynamo_orig_callable')
        self.prompts = json.loads(Path(__file__).with_name('prompts.json').read_text())
        self.weight_hashes = {name:tensor_sha(p) for name,p in self.model.named_parameters()}
        self.weights_sha256 = hashlib.sha256(json.dumps(self.weight_hashes,sort_keys=True).encode()).hexdigest()
        self.source_hashes = {}
        import transformers.cache_utils, transformers.masking_utils
        import transformers.models.qwen2.modeling_qwen2
        import transformers.integrations.sdpa_attention
        for module in (transformers.cache_utils,transformers.masking_utils,
                       transformers.models.qwen2.modeling_qwen2,transformers.integrations.sdpa_attention):
            path = Path(inspect.getfile(module))
            self.source_hashes[module.__name__] = dict(path=str(path),sha256=sha(path))
        self.identity = dict(weights_sha256=self.weights_sha256,weights=self.weight_hashes,
            model_revision=REVISION,model_id=MODEL,load_s=self.load_s,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()),
            source=self.source_hashes,attention='sdpa/EFFICIENT_ATTENTION only',
            torch=torch.__version__,transformers=__import__('transformers').__version__,
            gpu=__import__('subprocess').check_output(['nvidia-smi','--query-gpu=uuid,driver_version',
                   '--format=csv,noheader'],text=True).strip(),
            affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads())
        self.eos = self.model.generation_config.eos_token_id
        self.eos = [self.eos] if isinstance(self.eos,int) else self.eos

    def prompt(self, name, length):
        # Repeat a declared passage and truncate token IDs, preserving exact work.
        text = self.prompts[name]*max(1,length//32+1)
        ids = self.tokenizer(text,add_special_tokens=False)['input_ids'][:length]
        assert len(ids) == length
        return text,ids

    def weights_unchanged(self):
        assert {name:tensor_sha(p) for name,p in self.model.named_parameters()} == self.weight_hashes


class Runner:
    def __init__(self, workload, variant, length):
        self.workload,self.variant,self.length = workload,variant,length
        self.model = workload.model
        self.capacity = length+GENERATED
        start = time.perf_counter()
        self.cache = None
        if variant != 'A':
            self.cache = StaticCache(config=self.model.config,max_cache_len=self.capacity)
            self.cache.early_initialization(batch_size=1,num_heads=self.model.config.num_key_value_heads,
                head_dim=self.model.config.hidden_size//self.model.config.num_attention_heads,
                dtype=torch.float16,device=torch.device('cuda'))
        self.keys = torch.arange(self.capacity,device='cuda')
        self.position = torch.zeros(1,dtype=torch.long,device='cuda')
        self.input_token = torch.zeros((1,1),dtype=torch.long,device='cuda')
        torch.cuda.synchronize()
        self.setup_s = time.perf_counter()-start
        self.decode = self.eager_decode
        if variant == 'C':
            torch._dynamo.config.suppress_errors = False
            self.decode = torch.compile(self.eager_decode,mode='reduce-overhead',fullgraph=True,dynamic=False)

    def bias(self, positions, kv_len):
        padded = (kv_len+7)//8*8
        keys = torch.arange(padded,device='cuda')
        mask = torch.full((1,1,positions.numel(),padded),torch.finfo(torch.float16).min,
                          dtype=torch.float16,device='cuda')
        return mask.masked_fill(keys[None,None,None,:] <= positions[None,None,:,None],0)[...,:kv_len]

    def eager_decode(self, token, position, bias):
        logits = self.model(input_ids=token,position_ids=position.view(1,1),cache_position=position,
            attention_mask=bias,past_key_values=self.cache,use_cache=True,logits_to_keep=1).logits[:,0,:]
        return logits,logits.argmax(dim=-1).view(1,1)

    @torch.inference_mode()
    def request(self, prompt_name, generated=GENERATED, capture_logits=False, markers=False):
        assert 2 <= generated <= GENERATED
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with mark('E014/input_and_cache',markers):
            text,ids = self.workload.prompt(prompt_name,self.length)
            inputs = torch.tensor([ids],dtype=torch.long,device='cuda')
            if self.variant == 'A':
                self.cache = DynamicCache(config=self.model.config)
            else:
                self.cache.reset()
            tokens = torch.empty((generated,),dtype=torch.long,device='cuda')
            positions = torch.arange(self.length,device='cuda')
            prefill_bias = self.bias(positions,self.length if self.variant == 'A' else self.capacity)
        prefill_start = time.perf_counter()
        with mark('E014/prefill',markers):
            output = self.model(input_ids=inputs,position_ids=positions.view(1,-1),
                attention_mask=prefill_bias,past_key_values=self.cache,cache_position=positions,
                use_cache=True,logits_to_keep=1)
            logits = output.logits[:,0,:]
            next_token = logits.argmax(dim=-1).view(1,1)
            tokens[0].copy_(next_token[0,0])
            first = int(next_token.item())
        first_end = time.perf_counter()
        saved = [logits.detach().cpu()] if capture_logits else []
        self.input_token.copy_(next_token)
        decode_start = time.perf_counter()
        with mark('E014/decode',markers):
            for i in range(1,generated):
                self.position.fill_(self.length+i-1)
                kv_len = self.length+i if self.variant == 'A' else self.capacity
                bias = self.bias(self.position,kv_len)
                if self.variant == 'C':
                    torch.compiler.cudagraph_mark_step_begin()
                step_logits,next_token = self.decode(self.input_token,self.position,bias)
                tokens[i].copy_(next_token[0,0])
                self.input_token.copy_(next_token)
                if capture_logits:
                    saved.append(step_logits.detach().cpu())
            torch.cuda.synchronize()
        decode_end = time.perf_counter()
        with mark('E014/output',markers):
            output_ids = tokens.cpu().tolist()
            output_text = self.workload.tokenizer.decode(output_ids,skip_special_tokens=False)
        end = time.perf_counter()
        assert output_ids[0] == first and len(output_ids) == generated
        cache_length = int(self.cache.get_seq_length())
        assert cache_length == self.length+generated-1
        return dict(prompt=prompt_name,context=self.length,generated=generated,decode_forwards=generated-1,
            input_ids=ids,input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            output_ids=output_ids,output_text=output_text,eos_positions=[i for i,x in enumerate(output_ids) if x in self.workload.eos],
            eos_policy='fixed length, EOS recorded and not used to stop',
            ttft_s=first_end-start,prefill_to_first_s=first_end-prefill_start,
            preparation_s=prefill_start-start,decode_s=decode_end-decode_start,e2e_s=end-start,
            post_first_boundary_s=decode_start-first_end,output_s=end-decode_end,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            device_total=torch.cuda.get_device_properties(0).total_memory,
            cache_class=type(self.cache).__name__,static_capacity=self.capacity if self.variant != 'A' else None,
            cache_length=cache_length,
            logits=torch.stack(saved) if capture_logits else None)
