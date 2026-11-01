"""审查问题的负例与数值回归；不训练，不下载权重。"""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'llama_steps')]
import minimal_llama as minimal
import step06_model as core
import step09_real_llama as real
from step08_parity import to_hf_state


class ReviewRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, limit=8, batch=1):
        torch.manual_seed(17)
        config = core.LlamaConfig(vocab_size=31, block_size=limit)
        return core.MiniLlama(config).eval(), torch.ones(batch, 7, dtype=torch.long)

    def test_different_generation_lengths_rejected(self):
        with self.assertRaises(AssertionError):
            real.assert_same_generation(torch.tensor([[1,2,3,4]]), torch.tensor([[1,2]]),
                                        prompt_length=1, expected_new_tokens=3)

    def test_equal_but_incomplete_generations_rejected(self):
        with self.assertRaises(AssertionError):
            real.assert_same_generation(torch.tensor([[1,2]]), torch.tensor([[1,2]]),
                                        prompt_length=1, expected_new_tokens=3)

    def test_equal_complete_generation_accepted(self):
        x=torch.tensor([[1,2,3]])
        real.assert_same_generation(x,x.clone(),prompt_length=1,expected_new_tokens=2)

    def test_counting_uses_actual_generated_tokens(self):
        model, ids=self.model()
        with patch.object(real.time, 'perf_counter', side_effect=[10.0,16.0]):
            out, mean=real.generate(model,ids,12,greedy=True)
        self.assertEqual(out.size(1)-ids.size(1),2)
        self.assertEqual(mean,3.0)

    def test_zero_generation_is_noop(self):
        model, ids=self.model()
        with patch.object(model,'forward',side_effect=AssertionError('must not run')):
            out, mean=real.generate(model,ids,0,greedy=True)
        self.assertTrue(torch.equal(ids,out))
        self.assertEqual(mean,0.0)

    def test_negative_generation_rejected(self):
        model, ids=self.model()
        with self.assertRaises(ValueError): real.generate(model,ids,-1,greedy=True)

    def test_prompt_too_long_rejected(self):
        model,_=self.model()
        with self.assertRaises(ValueError): real.generate(model,torch.ones(1,9,dtype=torch.long),2,True)

    def test_batch_two_cache(self):
        model,ids=self.model(batch=2)
        out,_=real.generate(model,ids,1,True)
        self.assertEqual(tuple(out.shape),(2,8))

    def test_invalid_config_rejected_by_both_versions(self):
        for module in [minimal,core]:
            for settings in [dict(n_embd=12,n_head=5,n_kv_head=1),
                             dict(n_head=4,n_kv_head=3),dict(n_head=0),
                             dict(n_embd=12,n_head=4,n_kv_head=2)]:
                with self.subTest(module=module.__name__,settings=settings):
                    with self.assertRaises(ValueError): module.LlamaConfig(vocab_size=31,**settings)

    def test_valid_mha_gqa_mqa(self):
        for module in [minimal,core]:
            for kv in [1,2,4]:
                model=module.MiniLlama(module.LlamaConfig(vocab_size=31,n_kv_head=kv)).eval()
                with torch.no_grad(): self.assertEqual(tuple(model(torch.ones(1,3,dtype=torch.long)).shape),(1,3,31))

    def test_rmsnorm_dtype_and_values_match(self):
        for dtype in [torch.float32,torch.bfloat16]:
            a,b=minimal.RMSNorm(8),core.RMSNorm(8)
            x=torch.randn(2,8,dtype=dtype)
            self.assertEqual(a(x).dtype,dtype)
            self.assertTrue(torch.equal(a(x),b(x)))

    def test_unknown_source_tensor_rejected(self):
        config=core.LlamaConfig(vocab_size=31)
        sd=to_hf_state(core.MiniLlama(config)); sd['model.unmapped.weight']=torch.ones(1)
        with self.assertRaises(ValueError): real.hf_to_ours(sd,config)

    def test_tied_source_head_must_equal_embedding(self):
        config=core.LlamaConfig(vocab_size=31,tie_word_embeddings=True)
        model=core.MiniLlama(config)
        sd=to_hf_state(model); sd['lm_head.weight']=sd['model.embed_tokens.weight'].clone()+1
        with self.assertRaises(ValueError): real.hf_to_ours(sd,config)
        sd['lm_head.weight']=sd['model.embed_tokens.weight']
        real.hf_to_ours(sd,config)

    def test_chunked_prefill_and_decode_matches_full(self):
        model,_=self.model(limit=32)
        idx=torch.randint(0,31,(2,15)); cache=core.KVCache(model.config,2,torch.device('cpu'))
        with torch.no_grad():
            full=model(idx)
            for start,end in [(0,5),(5,8),(8,15)]:
                actual=model(idx[:,start:end],cache,start)
                torch.testing.assert_close(actual,full[:,start:end],atol=1e-5,rtol=1e-5)


if __name__=='__main__': unittest.main()
