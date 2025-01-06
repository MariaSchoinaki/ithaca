# Copyright 2021 the Ithaca Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ithaca model."""

from . import bigbird
from . import common_layers

import flax.linen as nn
import jax
import jax.numpy as jnp


class Model(nn.Module):
    """Transformer Model for sequence tagging."""
    vocab_char_size: int = 164
    vocab_word_size: int = 100004
    output_subregions: int = 85
    output_date: int = 160
    output_date_dist: bool = True
    output_return_emb: bool = False
    use_output_mlp: bool = True
    num_heads: int = 8
    num_layers: int = 6
    word_char_emb_dim: int = 192
    emb_dim: int = 512
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 1024
    causal_mask: bool = False
    feature_combine_type: str = 'concat'
    posemb_combine_type: str = 'add'
    region_date_pooling: str = 'first'
    learn_pos_emb: bool = True
    use_bfloat16: bool = False
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    activation_fn: str = 'gelu'
    model_type: str = 'bigbird'

    def setup(self):
        self.text_char_emb = nn.Embed(
            num_embeddings=self.vocab_char_size,
            features=self.word_char_emb_dim,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name='char_embeddings')
        self.text_word_emb = nn.Embed(
            num_embeddings=self.vocab_word_size,
            features=self.word_char_emb_dim,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name='word_embeddings')

    @nn.compact
    def __call__(self,
                 text_char=None,
                 text_word=None,
                 text_char_onehot=None,
                 text_word_onehot=None,
                 text_char_emb=None,
                 text_word_emb=None,
                 padding=None,
                 is_training=True,
                 return_embeddings=False):
        """Applies Ithaca model on the inputs."""

        # Handle padding and positional embeddings
        if text_char is not None and padding is None:
            padding = jnp.where(text_char > 0, 1, 0)
        elif text_char_onehot is not None and padding is None:
            padding = jnp.where(text_char_onehot.argmax(-1) > 0, 1, 0)
        padding_mask = padding[..., jnp.newaxis]
        text_len = jnp.sum(padding, 1)

        if self.posemb_combine_type == 'add':
            posemb_dim = None
        elif self.posemb_combine_type == 'concat':
            posemb_dim = self.word_char_emb_dim
        else:
            raise ValueError('Invalid posemb_combine_type value.')

        # Character embeddings
        if text_char is not None:
            x = self.text_char_emb(text_char)
        elif text_char_onehot is not None:
            x = self.text_char_emb.attend(text_char_onehot)
        elif text_char_emb is not None:
            x = text_char_emb
        else:
            raise ValueError('Invalid inputs for character embeddings.')

        # Word embeddings
        if text_word is not None:
            text_word_emb_x = self.text_word_emb(text_word)
        elif text_word_onehot is not None:
            text_word_emb_x = self.text_word_emb.attend(text_word_onehot)
        elif text_word_emb is not None:
            text_word_emb_x = text_word_emb
        else:
            raise ValueError('Invalid inputs for word embeddings.')

        # Combine embeddings
        if self.feature_combine_type == 'add':
            x = x + text_word_emb_x
        elif self.feature_combine_type == 'concat':
            x = jax.lax.concatenate([x, text_word_emb_x], 2)
        else:
            raise ValueError('Invalid feature_combine_type value.')

        # Add positional embeddings
        pe_init = common_layers.sinusoidal_init(
            max_len=self.max_len) if self.learn_pos_emb else None
        x = common_layers.AddPositionEmbs(
            posemb_dim=posemb_dim,
            posemb_init=pe_init,
            max_len=self.max_len,
            combine_type=self.posemb_combine_type,
            name='posembed_input',
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)

        # Set floating point precision
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32

        # Transformer layers
        for lyr in range(self.num_layers):
            x = bigbird.BigBirdBlock(
                qkv_dim=self.qkv_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dtype=dtype,
                causal_mask=self.causal_mask,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                deterministic=not is_training,
                activation_fn=self.activation_fn,
                connectivity_seed=lyr,
                name=f'encoderblock_{lyr}',
            )(x, padding_mask=padding_mask)

        # Normalize and store embeddings
        x = common_layers.LayerNorm(dtype=dtype, name='encoder_norm')(x)
        torso_output = x  # Save embeddings from the last layer

        # Compute outputs
        pred_date, logits_subregion, logits_mask, logits_nsp = self.compute_outputs(
            x, torso_output, is_training)

        if return_embeddings:
            return (pred_date, logits_subregion, logits_mask, logits_nsp), torso_output
        return pred_date, logits_subregion, logits_mask, logits_nsp

    def compute_outputs(self, x, torso_output, is_training):
        """Computes model outputs."""
        # Region logits
        logits_subregion = self.get_output(x, self.output_subregions, is_training)
        # Date predictions
        output_date_dim = self.output_date if self.output_date_dist else 1
        pred_date = self.get_output(x, output_date_dim, is_training)
        # Mask logits
        logits_mask = self.get_output(torso_output, self.word_char_emb_dim, is_training)
        # NSP logits
        logits_nsp = self.get_output(x, 2, is_training)
        return pred_date, logits_subregion, logits_mask, logits_nsp

    def get_output(self, x, out_dim, is_training):
        """Helper to compute output logits."""
        if self.use_output_mlp:
            return common_layers.MlpBlock(
                out_dim=out_dim,
                mlp_dim=self.emb_dim,
                dtype=x.dtype,
                out_dropout=False,
                dropout_rate=self.dropout_rate,
                deterministic=not is_training,
                activation_fn=self.activation_fn)(x)
        return nn.Dense(out_dim)(x)


def forward_fn_with_embeddings(model_instance, text_char, text_word, rngs, is_training):
    """
    Wrapper for the model's forward pass to return both logits and embeddings.
    """
    return model_instance(
        text_char=text_char,
        text_word=text_word,
        is_training=is_training,
        return_embeddings=True
    )
