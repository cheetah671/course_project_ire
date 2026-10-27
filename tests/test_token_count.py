from ragroute.harness.run_stages import chat_token_count


class _ListTok:  # transformers < 5: list of ids
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return list(range(17))


class _DictTok:  # transformers >= 5: BatchEncoding-like dict
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return {"input_ids": list(range(17)), "attention_mask": [1] * 17}


class _BatchedTok:  # dict with a batch dimension
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return {"input_ids": [list(range(17))], "attention_mask": [[1] * 17]}


def test_chat_token_count_handles_all_return_types():
    msgs = [{"role": "user", "content": "hi"}]
    assert chat_token_count(_ListTok(), msgs) == 17
    assert chat_token_count(_DictTok(), msgs) == 17     # len(dict) would wrongly be 2
    assert chat_token_count(_BatchedTok(), msgs) == 17
