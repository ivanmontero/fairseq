import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention

class AutoencoderDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.autoencoder_hidden_size = getattr(args, 'autoencoder_hidden_size', self.embed_dim) # autoencoder_hidden_size
        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not self.cross_self_attention,
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.proj_v = nn.Linear(self.autoencoder_hidden_size, self.embed_dim)
            self.proj_gh = nn.Linear(self.embed_dim, self.embed_dim)
            self.proj_gz = nn.Linear(self.autoencoder_hidden_size, self.embed_dim)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False


    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        bottleneck_out=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_attn=False,
        need_head_weights=False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        # if self.cross_self_attention and not (incremental_state is not None and "prev_key" in self.self_attn._get_input_buffer(incremental_state)):
        #     if self_attn_mask is not None:
        #         self_attn_mask = torch.cat((x.new(x.size(0), encoder_out.size(0)).zero_(), self_attn_mask), dim=1)
        #     if self_attn_padding_mask is not None:
        #         if encoder_padding_mask is None:
        #             encoder_padding_mask = self_attn_padding_mask.new(encoder_out.size(1), encoder_out.size(0)).zero_()
        #         self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
        #     y = torch.cat((encoder_out, x), dim=0)
        # else:
        #     y = x
        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn_layer_norm is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)


            bottleneck_expanded = bottleneck_out.expand_as(x)
            x = torch.sigmoid(self.proj_gh(x) + self.proj_gz(bottleneck_expanded)) * self.proj_v(bottleneck_expanded)
            #         return (output,)
            # if prev_attn_state is not None:
            #     if incremental_state is None:
            #         incremental_state = {}
            #     prev_key, prev_value = prev_attn_state[:2]
            #     saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            #     if len(prev_attn_state) >= 3:
            #         saved_state["prev_key_padding_mask"] = prev_attn_state[2]
            #     self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            # x, attn = self.encoder_attn(
            #     query=x,
            #     key=encoder_out,
            #     value=encoder_out,
            #     key_padding_mask=encoder_padding_mask,
            #     incremental_state=incremental_state,
            #     static_kv=True,
            #     need_weights=need_attn or (not self.training and self.need_attn),
            #     need_head_weights=need_head_weights,
            # )

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            if self_attn_padding_mask is not None:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"], saved_state["prev_key_padding_mask"]
            else:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

# class HFBertAECrossAttention(nn.Module):
#     def __init__(self, config, autoencoder_hidden_size):
#         super().__init__()
#         self.autoencoder_hidden_size = autoencoder_hidden_size
#         self.proj_v = nn.Linear(self.autoencoder_hidden_size, config.hidden_size)
#         self.proj_gh = nn.Linear(config.hidden_size, config.hidden_size)
#         self.proj_gz = nn.Linear(self.autoencoder_hidden_size, config.hidden_size)

#     def prune_heads(self, heads):
#         pass

#     # Override here
#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#     ):
#         z_expand = encoder_hidden_states.expand_as(hidden_states)
#         output = torch.sigmoid(self.proj_gh(hidden_states) + self.proj_gz(z_expand)) * self.proj_v(z_expand)
#         return (output,)

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
