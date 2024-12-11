import numpy as np
import torch
from ctg.new_ctg import ModelWrapper


class TokenMaskingCTG(ModelWrapper):

    def generate_moderated(
        self, prompt, embedder, clf, target="", tau=0.5, max_tokens=100, verbose=False
    ):

        edited = False
        input = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]

        input_ids = self._get_ids(input)
        sentence = target

        if verbose:
            print(sentence, end="")

        j = 0
        while True:

            logits_top, logits_top_idx, last_hidden_state = self.get_top_logits_from_ids(input_ids)

            mask = np.ones(len(logits_top_idx))

            scores = []
            for i, idx in enumerate(logits_top_idx):
                if self.tokenizer.name_or_path.find("mistral") > -1:
                    next_token_str = self.tokenizer.convert_ids_to_tokens(
                        idx.item()
                    ).replace("▁", " ")
                else:
                    next_token_str = self.tokenizer.decode(idx.item())
                # print(sentence+next_token_str)
                embedding = embedder.encode(sentence + next_token_str)
                score = clf.predict_proba([embedding])[0][1]
                scores.append(score)
                if score > tau:
                    mask[i] = 0
                    edited = True
            # print(scores)

            mask = torch.from_numpy(np.array(mask, dtype="bool"))
            logits_top = torch.masked_select(logits_top, mask)
            logits_top_idx = torch.masked_select(logits_top_idx, mask)

            probs_top = torch.nn.functional.softmax(
                logits_top / self.temperature, dim=-1
            )

            if len(logits_top_idx) >= 1:
                next_token = logits_top_idx[
                    self._rng.choice(len(logits_top_idx), p=probs_top.detach().numpy())
                ]
            else:
                _, last_hidden_state = self._forward_pass_from_ids(input_ids)
                if verbose:
                    print("\n")
                return sentence, last_hidden_state, edited

            j += 1
            if (next_token.item() == self.tokenizer.eos_token_id) or (j > max_tokens):
                _, last_hidden_state = self._forward_pass_from_ids(input_ids)
                if verbose:
                    print("\n")
                return sentence, last_hidden_state, edited


            if self.tokenizer.name_or_path.find("mistral") > -1:
                next_token_str = self.tokenizer.convert_ids_to_tokens(
                    next_token.item()
                ).replace("▁", " ")
            else:
                next_token_str = self.tokenizer.decode(next_token.item())

            sentence += next_token_str
            if verbose:
                print(next_token_str, end="")

            input_ids = torch.cat((input_ids, next_token.reshape(1, 1)), dim=1)
