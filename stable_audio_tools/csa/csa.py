import einops
from einops import rearrange

import torch
import torch.nn as nn
from ..inference.utils import prepare_audio
from .utils import conv_nd
from ..models.transformer import *
from ..models.dit import *
# from ..models.diffusion import DiTWrapper
from .utils import zero_module, TimestepEmbedSequential
from ..models.blocks import FourierFeatures
# from ..models.autoencoders import create_encoder_from_config

class prepare_conditioned_audio(nn.Module):
    # def __init__(self, encoder_model, pre_transform_io_channels, batch_size, dim_in, model_sample_rate ):
    def __init__(self, encoder_config, pre_transform_io_channels, dim_in, model_sample_rate ):
        super().__init__()
        self.io_channels = pre_transform_io_channels
        self.prepare_audio = prepare_audio
        
        if encoder_config:
            from ..models.autoencoders import create_encoder_from_config
        self.encoder = create_encoder_from_config(encoder_config)
        # self.encoder = encoder_model
        # self.repeat = repeat(batch_size, 1, 1)
        ## TODO: Bring the information about the output channles of the encoder
        # self.preprocess_conv_init = nn.Conv1d(self.io_channels*2, self.io_channels, 1, bias=False)
        self.preprocess_conv = nn.Conv1d(self.io_channels*2, self.io_channels*2, 1, bias=False)
        self.model_sample_rate = model_sample_rate
        self.sample_size = 2097152
        # self.io_channels = model.pretransform.io_channels
        # self.prepare_audio = prepare_audio
        # self.encoder = model.pretransform.encode
        # self.repeat = repeat(batch_size, 1, 1)
        # self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self,x):
        
        audio_sample_size = self.sample_size
        # in_sr, init_audio = init_audio
        in_sr = self.model_sample_rate
        device = x.device

        # x = self.prepare_audio(x, in_sr, self.model_sample_rate, target_length=audio_sample_size, target_channels=self.io_channels)

        x = self.prepare_audio(x, in_sr, self.model_sample_rate, target_length=audio_sample_size, target_channels=self.io_channels, device=device)
        # x = self.prepare_audio(x, init_audio, in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=self.io_channels, device=device)

        x = self.encoder(x)
        ### Im not sure i need here the repeat ###
        # x = self.repeat(x)
        # x = self.preprocess_conv_init(x)
        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")
        return x


class ControlledContinuousTransformer(ContinuousTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for param in self.layers.parameters():
            param.requires_grad = False
        # self.layers.requires_grad = False

    # @torch.no_grad()
    def forward(
            self,
            x,
            control_sig,
            mask = None,
            prepend_embeds = None,
            prepend_mask = None,
            global_cond = None,
            return_info = False,
            only_mid_control=False,
            **kwargs
        ):
        with torch.no_grad():

            batch, seq, device = *x.shape[:2], x.device

            info = {
                "hidden_states": [],
            }

            x = self.project_in(x)

            if prepend_embeds is not None:
                prepend_length, prepend_dim = prepend_embeds.shape[1:]

                assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

                x = torch.cat((prepend_embeds, x), dim = -2)

                if prepend_mask is not None or mask is not None:
                    mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                    prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                    mask = torch.cat((prepend_mask, mask), dim = -1)

                # Attention layers 

                if self.rotary_pos_emb is not None:
                    rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
                else:
                    rotary_pos_emb = None

                if self.use_sinusoidal_emb or self.use_abs_pos_emb:
                    x = x + self.pos_emb(x)

            # flip the order of the controlnet architecture:
            # control_reverse = control_sig.reverse()


            # Iterate over the transformer layers


            for idx, layer in enumerate(self.layers):
                if not(idx == 0):
                    x = x + control_sig.pop(0)
                    # x = torch.cat([x, control_sig.pop(0)], dim=1)
                #x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
                x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
        
                if return_info:
                    info["hidden_states"].append(x)

            ### Adding the output of the last layer: ###    
            x = x + control_sig.pop(0)
            if return_info:
                    info["hidden_states"].append(x)


            x = self.project_out(x)

            if return_info:
                return x, info
            
            return x

class ControlNet(nn.Module):
    # class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        sample_rate,
        encoder,
        io_channels = 32,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        **kwargs
        ):

        super().__init__()
        self.sample_rate = sample_rate
        self.dim = dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])
        self.io_channels = io_channels
        self.encoder_config = encoder
        # self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        # self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        # if self.encoder_config:
        #     from ..models.autoencoders import create_encoder_from_config

        # self.encoder = create_encoder_from_config(self.encoder_config)
        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(self.dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(self.dim, abs_pos_emb_max_length)
        
        #### first zeroconv: for the conditioned input ####
        #### TODO: from where to bring the 1024? ####
        self.zero_convs_init = self.make_zero_conv(1024)
        self.zero_convs = nn.ModuleList([])

        #### prepare the condtioned audio ####
        # self.prepare_conditioned_audio = prepare_conditioned_audio(encoder_model, io_channels, batch_size, self.dim_in, self.sample_rate )

        self.prepare_conditioned_audio = prepare_conditioned_audio(self.encoder_config, io_channels, self.dim_in, self.sample_rate )

        # #### adding preprcess to the conditioned audio signal #### :
        # # in_sr, init_audio = init_audio

        # # io_channels = model.io_channels

        # io_channels = model.pretransform.io_channels

        # # Prepare the initial audio for use by the model
        # init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)
        # init_audio = model.pretransform.encode(init_audio)
        # init_audio = init_audio.repeat(batch_size, 1, 1)

        self.project_in = nn.Linear(self.dim_in, self.dim, bias=False) if self.dim_in is not None else nn.Identity()
        self.project_in_control = nn.Linear(self.dim_in*2, self.dim, bias=False) if self.dim_in is not None else nn.Identity()


        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    self.dim,
                    dim_heads = dim_heads,
                    cross_attend = cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )
            ## TODO" understand from where to bring the 1024
            self.zero_convs.append(self.make_zero_conv(1025))


        for param in self.project_in.parameters():
            param.requires_grad = False
        




    
    def make_zero_conv(self, channels):
        ## TODO: pat attentuon to add another variable as dim in the future
        return TimestepEmbedSequential(zero_module(conv_nd(1, channels, channels, 1, padding=0)))

    def forward(
        self,
        x,
        hint,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        return_info = False,
        **kwargs
    ):

    ########  The preprocess of x and concat with "global cond" ########
        batch, seq, device = *x.shape[:2], x.device

        info = {
            "hidden_states": [],
        }

        x_prepend = self.project_in(x)

        # if prepend_embeds is not None:
        #     prepend_length, prepend_dim = prepend_embeds.shape[1:]

        #     assert prepend_dim == x_prepend.shape[-1], 'prepend dimension must match sequence dimension'

        #     x_prepend = torch.cat((prepend_embeds, x_prepend), dim = -2)

        #     if prepend_mask is not None or mask is not None:
        #         mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
        #         prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

        #         mask = torch.cat((prepend_mask, mask), dim = -1)
        #############################################
    
        ######## The preprocess of the conditional audio: #######
        guided_hint =  self.prepare_conditioned_audio(hint)

        batch, seq, device = *guided_hint.shape[:2], guided_hint.device
        # info = {
        #     "hidden_states": [],
        # }

        guided_hint =  self.project_in_control(guided_hint)
        guided_hint = self.zero_convs_init(guided_hint)

        #########################################################

        ###### Summing both signals: x, cond_audio and concat with the global_info #####

        assert guided_hint.shape == x_prepend.shape, f"Both signals have different dimentiins. {guided_hint.shape} , {x_prepend.shape}"

        summed_signal = guided_hint + x_prepend

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == summed_signal.shape[-1], 'prepend dimension must match sequence dimension'

            summed_signal = torch.cat((prepend_embeds, summed_signal), dim = -2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                mask = torch.cat((prepend_mask, mask), dim = -1)
        
        ########################################################

        # if prepend_embeds is not None:
        #     prepend_length, prepend_dim = prepend_embeds.shape[1:]

        #     assert prepend_dim == guided_hint.shape[-1], 'prepend dimension must match sequence dimension'

        #     guided_hint = torch.cat((prepend_embeds, guided_hint), dim = -2)

        #     if prepend_mask is not None or mask is not None:
        #         mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
        #         prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

        #         mask = torch.cat((prepend_mask, mask), dim = -1)

        ##### Defining the Rotary. For now - It is By the original conrol signal #####
        #### For succeedding in using just the original control, Ishould make 
        #### a new signal of 1025 with the cntrol and global concat. im not sure
        ### if it is matter. but still doing so.

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == guided_hint.shape[-1], 'prepend dimension must match sequence dimension'

            guided_hint_cat_global = torch.cat((prepend_embeds, guided_hint), dim = -2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                mask = torch.cat((prepend_mask, mask), dim = -1)
        
        ### Now the rotary: 
        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(guided_hint_cat_global.shape[1])
        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            guided_hint_cat_global = guided_hint_cat_global + self.pos_emb(guided_hint_cat_global)
        #################################
        
        outs = []
        # h = x.type(self.dtype)

        # Iterate over the transformer layers

        # for layer, zero_conv in zip(self.layers, self.zero_convs):
        #     if guided_hint is not None:
        #         x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
        #         x += guided_hint
        #         guided_hint = None
        #     else:
        #         x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
        
        #     outs.append(zero_conv(x))

        # for layer, zero_conv in zip(self.layers, self.zero_convs):
        #     if guided_hint is not None:
        #         x_prepend += guided_hint
        #         cur_sig = checkpoint(layer, x_prepend, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
                
        #         guided_hint = None
        #     else:
        #         cur_sig = checkpoint(layer, cur_sig, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
        
        #     outs.append(zero_conv(cur_sig))





        for layer, zero_conv in zip(self.layers, self.zero_convs):
   
            summed_signal = checkpoint(layer, summed_signal, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
        
            outs.append(zero_conv(summed_signal))

        return outs

        # for layer in self.layers:

            
        #     #x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
        #     x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)

        #     if return_info:
        #         info["hidden_states"].append(x)

        # x = self.project_out(x)

        # if return_info:
        #     return x, info
        
        # return x


class ControlDiffusionTransformer(DiffusionTransformer):

    ### A module which include both modules the controler and the controled.
    ### It is DiTWrapper object, which defines another NeuralNet inside it.
    # def __init__(self, controlnet, control_element, only_specific_control, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     ### initializing the control mid 
    #     self.control_model = controlnet
    #     # self.control_model = instantiate_from_config(control_stage_config)
    #     self.control_element = control_element
    #     self.only_spesific_control = only_specific_control
    #     self.control_scales = [1.0] * 13

### For now, because i need to change inner lines in init,
### I just defined it from the begginning.
### TODO: make sure the inherit is ok
    def __init__(self,
            controlnet_config,
            only_specific_control, 
            io_channels=32, 
            patch_size=1,
            embed_dim=768,
            cond_token_dim=0,
            project_cond_tokens=True,
            global_cond_dim=0,
            project_global_cond=True,
            input_concat_dim=0,
            prepend_cond_dim=0,
            depth=12,
            num_heads=8,
            transformer_type: tp.Literal["x-transformers", "continuous_transformer", "controled_continuous_transformer"] = "controled_continuous_transformer",
            global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
            **kwargs):

            # super(nn.Module).__init__()
            nn.Module.__init__(self)
            # super().__init__(self)

            ### Roi: #####
            ### initializing the control mid 
            self.control_model_config = controlnet_config
            # self.control_model = instantiate_from_config(control_stage_config)
            self.only_spesific_control = only_specific_control
            self.control_scales = [1.0] * 13
            #######

            self.cond_token_dim = cond_token_dim

            # Timestep embeddings
            timestep_features_dim = 256

            self.timestep_features = FourierFeatures(1, timestep_features_dim)

            self.to_timestep_embed = nn.Sequential(
                nn.Linear(timestep_features_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),
            )

            if cond_token_dim > 0:
                # Conditioning tokens

                cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
                self.to_cond_embed = nn.Sequential(
                    nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
                )
            else:
                cond_embed_dim = 0

            if global_cond_dim > 0:
                # Global conditioning
                global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
                self.to_global_embed = nn.Sequential(
                    nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(global_embed_dim, global_embed_dim, bias=False)
                )

            if prepend_cond_dim > 0:
                # Prepend conditioning
                self.to_prepend_embed = nn.Sequential(
                    nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim, bias=False)
                )

            self.input_concat_dim = input_concat_dim

            dim_in = io_channels + self.input_concat_dim

            self.patch_size = patch_size

            # Transformer

            self.transformer_type = transformer_type

            self.global_cond_type = global_cond_type

            if self.transformer_type == "x-transformers":
                self.transformer = ContinuousTransformerWrapper(
                    dim_in=dim_in * patch_size,
                    dim_out=io_channels * patch_size,
                    max_seq_len=0, #Not relevant without absolute positional embeds
                    attn_layers = Encoder(
                        dim=embed_dim,
                        depth=depth,
                        heads=num_heads,
                        attn_flash = True,
                        cross_attend = cond_token_dim > 0,
                        dim_context=None if cond_embed_dim == 0 else cond_embed_dim,
                        zero_init_branch_output=True,
                        use_abs_pos_emb = False,
                        rotary_pos_emb=True,
                        ff_swish = True,
                        ff_glu = True,
                        **kwargs
                    )
                )

            elif self.transformer_type == "continuous_transformer":

                global_dim = None

                if self.global_cond_type == "adaLN":
                    # The global conditioning is projected to the embed_dim already at this point
                    global_dim = embed_dim

                self.transformer = ContinuousTransformer(
                    dim=embed_dim,
                    depth=depth,
                    dim_heads=embed_dim // num_heads,
                    dim_in=dim_in * patch_size,
                    dim_out=io_channels * patch_size,
                    cross_attend = cond_token_dim > 0,
                    cond_token_dim = cond_embed_dim,
                    global_cond_dim=global_dim,
                    **kwargs
                )
            ### Roi ####
            elif self.transformer_type == "controled_continuous_transformer":

                global_dim = None

                if self.global_cond_type == "adaLN":
                        # The global conditioning is projected to the embed_dim already at this point
                        global_dim = embed_dim

                self.transformer = ControlledContinuousTransformer(
                    dim=embed_dim,
                    depth=depth,
                    dim_heads=embed_dim // num_heads,
                    dim_in=dim_in * patch_size,
                    dim_out=io_channels * patch_size,
                    cross_attend = cond_token_dim > 0,
                    cond_token_dim = cond_embed_dim,
                    global_cond_dim=global_dim,
                    **kwargs
                )
            #######

            else:
                raise ValueError(f"Unknown transformer type: {self.transformer_type}")

            self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
            nn.init.zeros_(self.preprocess_conv.weight)
            self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
            nn.init.zeros_(self.postprocess_conv.weight)


            self.controlnet = ControlNet(    
                    dim=embed_dim,
                    depth=depth,
                    dim_heads=embed_dim // num_heads,
                    dim_in=dim_in * patch_size,
                    dim_out=io_channels * patch_size,
                    cross_attend = cond_token_dim > 0,
                    cond_token_dim = cond_embed_dim,
                    global_cond_dim=global_dim,
                    **controlnet_config.get('config', None), 
                    **kwargs )

    def _forward(
        self, 
        x, 
        t, 
        conotrol_signal=None,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        **kwargs):

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:

            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists
        if global_embed is not None:
            global_embed = global_embed + timestep_embed
        else:
            global_embed = timestep_embed

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend":
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)


        control_outs = self.controlnet(x,conotrol_signal, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **extra_args, **kwargs)


        if self.transformer_type == "x-transformers":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, **extra_args, **kwargs)
        elif self.transformer_type == "continuous_transformer":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **extra_args, **kwargs)

            if return_info:
                output, info = output
        elif self.transformer_type == "mm_transformer":
            output = self.transformer(x, context=cross_attn_cond, mask=mask, context_mask=cross_attn_cond_mask, **extra_args, **kwargs)

        elif self.transformer_type == "controled_continuous_transformer":
            output = self.transformer(x, control_outs, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **extra_args, **kwargs)


        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    def forward(
        self, 
        x, 
        t, 
        control_signal,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        causal=False,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        **kwargs):

        assert causal == False, "Causal mode is not supported for DiffusionTransformer"

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # CFG dropout
        if cfg_dropout_prob > 0.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)


        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None):
            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if input_concat_cond is not None:
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:

                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                        negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                    
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                           
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
         

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None
            
            batch_output = self._forward(
                batch_inputs, 
                batch_timestep, 
                cross_attn_cond=batch_cond, 
                cross_attn_cond_mask=batch_cond_masks, 
                mask = batch_masks, 
                input_concat_cond=batch_input_concat_cond, 
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                return_info = return_info,
                **kwargs)

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            else:
                output = cfg_output
            
            if return_info:
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                control_signal,
                cross_attn_cond=cross_attn_cond, 
                cross_attn_cond_mask=cross_attn_cond_mask, 
                input_concat_cond=input_concat_cond, 
                global_embed=global_embed, 
                prepend_cond=prepend_cond, 
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )







    