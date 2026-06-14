"""
for ControlNet
"""
import einops
import torch
import torch.nn as nn
from lvdm.models.ddpm3d import LatentCustomDiffusion
from lvdm.modules.networks.custom import UNetModel
from lvdm.basics import zero_module, conv_nd, linear, avg_pool_nd, normalization, disabled_train
from lvdm.models.utils_diffusion import timestep_embedding
from utils.utils import instantiate_from_config
from einops import rearrange
from lvdm.modules.attention import SpatialTransformer, TemporalTransformer
from lvdm.modules.networks.custom import TimestepEmbedSequential, ResBlock, Downsample
import logging
mainlogger = logging.getLogger('mainlogger')

class ControlNet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: in_channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(self,
                 in_channels,
                 model_channels,
                 hint_channels, # instead out_channels
                 num_res_blocks,
                 attention_resolutions,
                 dropout=0.0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=2,
                 context_dim=None,
                 use_scale_shift_norm=False,
                 resblock_updown=False,
                 num_heads=-1,
                 num_head_channels=-1,
                 transformer_depth=1,
                 use_linear=False,
                 use_checkpoint=False,
                 temporal_conv=False,
                 tempspatial_aware=False,
                 temporal_attention=True,
                 use_relative_position=True,
                 use_causal_attention=False,
                 temporal_length=None,
                 use_fp16=False,
                 addition_attention=False,
                 temporal_selfatt_only=True,
                 image_cross_attention=False,
                 image_cross_attention_scale_learnable=False,
                 default_fs=4,
                 fs_condition=False,
                 camera_pose_condition=False,
                 use_guidance=None,
                 guidance_config=None
                ):
        super(ControlNet, self).__init__()
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.hint_channels = hint_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        temporal_self_att_only = True
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.default_fs = default_fs
        self.fs_condition = fs_condition
        # CUSTOM
        self.camera_pose_condition = camera_pose_condition
        self.use_guidance = use_guidance

        if self.use_guidance is not None:
            if self.use_guidance == "adapter": # naive adapter
                self.guidance_model = instantiate_from_config(guidance_config)
            elif self.use_guidance == "ca-adapter": # C.A adapter # TODO:
                self.guidance_model = instantiate_from_config(guidance_config)
            elif self.use_guidance == "gating-attn": # gating attn 
                self.guidance_model = instantiate_from_config(guidance_config)
            elif self.use_guidance == "gating-attn-add": # gating attn add
                self.guidance_model = instantiate_from_config(guidance_config)
            else:
                self.guidance_model = None
                raise ValueError(f"Invalid guidance type: {self.use_guidance}")
            print(f"[DEBUG] : C.A : self.guidance_model {self.guidance_model}")

        ## Time embedding blocks
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if fs_condition or camera_pose_condition:
            self.fps_embedding = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)
        if self.camera_pose_condition:
            self.camera_pose_embedding = nn.Sequential(
                linear(temporal_length*12, model_channels),
                nn.SiLU()
            )
            nn.init.zeros_(self.camera_pose_embedding[0].weight)
            nn.init.zeros_(self.camera_pose_embedding[0].bias)
            # linear layer -> fps embedding
        ## Input Block
        self.dims = dims
        
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
            ]
        )
        
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        
        # FIXME: shape 검토. DINO feature -> 더 작은 size 임
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        if self.addition_attention:
            self.init_attn=TimestepEmbedSequential(
                TemporalTransformer(
                    model_channels,
                    n_heads=8,
                    d_head=num_head_channels,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint, only_self_att=temporal_selfatt_only, 
                    causal_attention=False, relative_position=use_relative_position, 
                    temporal_length=temporal_length))

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, 
                            depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                            use_checkpoint=use_checkpoint, disable_self_attn=False, 
                            video_length=temporal_length, image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                            use_guidance=self.use_guidance
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(ch, num_heads, dim_head,
                                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                                use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only, 
                                causal_attention=use_causal_attention, relative_position=use_relative_position, 
                                temporal_length=temporal_length
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout, 
                            out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2

        print("[DEBUG] : ControlNet : len(self.input_blocks) = ", len(self.input_blocks))
        print("[DEBUG] : ControlNet : len(self.zero_convs) = ", len(self.zero_convs))
        # exit()
        
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        layers = [
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv
            ),
            SpatialTransformer(ch, num_heads, dim_head, 
                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                use_checkpoint=use_checkpoint, disable_self_attn=False, video_length=temporal_length, 
                image_cross_attention=self.image_cross_attention,image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                use_guidance=self.use_guidance
            )
        ]
        if self.temporal_attention:
            layers.append(
                TemporalTransformer(ch, num_heads, dim_head,
                    depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                    use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only, 
                    causal_attention=use_causal_attention, relative_position=use_relative_position, 
                    temporal_length=temporal_length
                )
            )
        layers.append(
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware, 
                use_temporal_conv=temporal_conv
                )
        )

        ## Middle Block
        self.middle_block = TimestepEmbedSequential(*layers)
        self.middle_zero_conv = self.make_zero_conv(ch)

    def forward(self, x, hint, timesteps, context=None, features_adapter=None, fs=None, camera_pose=None, **kwargs):
        b,_,t,_,_ = x.shape
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
        emb = self.time_embed(t_emb)
        # if isinstance(features_adapter, list):
        #     for i, feat in enumerate(features_adapter):
        #         print(f"[DEBUG] : features_adapter[{i}].shape", feat.shape)
                # (B*T, C = [320, 640, 1280, 1280], H, W) 
        # exit()
        # import pdb; pdb.set_trace()
        if len(hint.shape) == 5 and hint.shape[0] == b:
            hint = rearrange(hint, "b c t h w -> (b t) c h w")
        guided_hint = self.input_hint_block(hint, emb, context)
        # guided_hint.shape torch.Size([16, 320, 40, 64]) torch.Size([16, 3, 320, 512]
        # print("[DEBUG] : guided_hint.shape", guided_hint.shape, hint.shape)
        
        # print("[DEBUG] : features_adapter.shape", features_adapter.shape)
        ## repeat t times for context [(b t) 77 768] & time embedding
        ## check if we use per-frame image conditioning
        _, l_context, _ = context.shape
        if l_context == 77 + t*16: ## !!! HARD CODE here
            context_text, context_img = context[:,:77,:], context[:,77:,:]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)
        emb = emb.repeat_interleave(repeats=t, dim=0)
        
        ## always in shape (b t) c h w, except for temporal layer
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        ## combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor(
                    [self.default_fs] * b, dtype=torch.long, device=x.device)
            fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)
            # print("[DEBUG] : fs_emb.shape", fs_emb.shape, "fs.shape", fs.shape) # fs_emb.shape torch.Size([1, 320]) fs.shape torch.Size([1])  
            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        if self.camera_pose_condition:
            if camera_pose is None:
                raise ValueError('camera_pose is required for camera_pose_condition')
            # print("[DEBUG] : camera_pose.shape", camera_pose.shape) # (B, temporal length, 12)
            camera_pose = rearrange(camera_pose, 'b t l -> b (t l)', t=self.temporal_length, l=12)
            camera_pose_emb = self.camera_pose_embedding(camera_pose)
            camera_pose_embed = self.fps_embedding(camera_pose_emb)
            camera_pose_embed = camera_pose_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + camera_pose_embed
 
        h = x.type(self.dtype)
        adapter_idx = 0
        hs = []
        additional_context = None
        for id, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            # ------------------------------- #
            # if "gating-attn" in self.use_guidance and id > 0: # id > 0 -> is it necessary?
            #     additional_context = features_adapter[id // 3]
            # ------------------------------- #
            h = module(h, emb, context=context, batch_size=b, additional_context=additional_context)
            
            # ControlNet
            if guided_hint is not None:
                h = h + guided_hint
                guided_hint = None
            
            if id ==0 and self.addition_attention:
                h = self.init_attn(h, emb, context=context, batch_size=b, additional_context=additional_context)
            ## plug-in adapter features
            # print("[DEBUG] : id = ", id, " ; h.shape = ", h.shape)
            # ------------------------------- #
            # if self.use_guidance == "adapter":
            #     if ((id+1)%3 == 0) and features_adapter is not None:
            #         # print("[DEBUG] : id = ", id, " ; h.shape = ", h.shape, " ; features_adapter[idx].shape = ", features_adapter[adapter_idx].shape)
            #         h = h + features_adapter[adapter_idx]
            #         adapter_idx += 1
            # ------------------------------- #
            # hs.append(h) # original
            hs.append(zero_conv(h, emb, context=context, batch_size=b))
            
        # original code
        # if features_adapter is not None:
        #     assert len(features_adapter)==adapter_idx, 'Wrong features_adapter'

        h = self.middle_block(h, emb, context=context, batch_size=b)
        hs.append(self.middle_zero_conv(h, emb, context=context, batch_size=b))
        
        return hs

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))
    
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        b,_,t,_,_ = x.shape
        camera_pose = kwargs.get('camera_pose', None)
        
        # with torch.no_grad():
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
        emb = self.time_embed(t_emb)
        _, l_context, _ = context.shape
        if l_context == 77 + t*16: ## !!! HARD CODE here
            context_text, context_img = context[:,:77,:], context[:,77:,:]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)
        emb = emb.repeat_interleave(repeats=t, dim=0)
        
        ## always in shape (b t) c h w, except for temporal layer
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        ## combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor(
                    [self.default_fs] * b, dtype=torch.long, device=x.device)
            fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)
            # print("[DEBUG] : fs_emb.shape", fs_emb.shape, "fs.shape", fs.shape) # fs_emb.shape torch.Size([1, 320]) fs.shape torch.Size([1])  
            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        if self.camera_pose_condition:
            if camera_pose is None:
                raise ValueError('camera_pose is required for camera_pose_condition')
            # print("[DEBUG] : camera_pose.shape", camera_pose.shape) # (B, temporal length, 12)
            camera_pose = rearrange(camera_pose, 'b t l -> b (t l)', t=self.temporal_length, l=12)
            camera_pose_emb = self.camera_pose_embedding(camera_pose)
            camera_pose_embed = self.fps_embedding(camera_pose_emb)
            camera_pose_embed = camera_pose_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + camera_pose_embed

        h_state = x.type(self.dtype)
        adapter_idx = 0
        hs = []
        additional_context = None
        for id, module in enumerate(self.input_blocks):
            # ------------------------------- #
            # if "gating-attn" in self.use_guidance and id > 0: # id > 0 -> is it necessary?
            #     additional_context = features_adapter[id // 3]
            # ------------------------------- #
            h_state = module(h_state, emb, context=context, batch_size=b, additional_context=additional_context)
            if id ==0 and self.addition_attention:
                h_state = self.init_attn(h_state, emb, context=context, batch_size=b, additional_context=additional_context)
            ## plug-in adapter features
            # print("[DEBUG] : id = ", id, " ; h.shape = ", h.shape)
            # ------------------------------- #
            # if self.use_guidance == "adapter":
            #     if ((id+1)%3 == 0) and features_adapter is not None:
            #         # print("[DEBUG] : id = ", id, " ; h.shape = ", h.shape, " ; features_adapter[idx].shape = ", features_adapter[adapter_idx].shape)
            #         h = h + features_adapter[adapter_idx]
            #         adapter_idx += 1
            # ------------------------------- #
            hs.append(h_state)
        # original code
        # if features_adapter is not None:
        #     assert len(features_adapter)==adapter_idx, 'Wrong features_adapter'

        h_state = self.middle_block(h_state, emb, context=context, batch_size=b)
        # import pdb; pdb.set_trace()

        if control is not None:
            h_state += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h_state = torch.cat([h_state, hs.pop()], dim=1)
            else:
                h_state = torch.cat([h_state, hs.pop() + control.pop()], dim=1)
            h_state = module(h_state, emb, context, batch_size=b)

        h = h_state.type(x.dtype)
        y = self.out(h)
        
        # reshape back to (b c t h w)
        y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
        return y

class ControlVideoDiffusion(LatentCustomDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model.requires_grad = False
        
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        # Freeze original U-Net
        self.model.eval()
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False
        
    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs, camera_pose, control = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        kwargs.update({"control": control})
        loss, loss_dict = self(x, c, camera_pose=camera_pose, **kwargs)
        return loss, loss_dict
    
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        ## encode video frames x to z via a 2D encoder        
        z = self.encode_first_stage(x)
        
        render = super().get_input(batch, 'renders')
        z_render = self.encode_first_stage(render)
        
        ## get caption condition
        cond_input = batch[self.cond_stage_key]

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
                
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if self.cond_select_all: # B x T
            if random_uncond:
                random_num_input = torch.rand(x.size(0) * x.size(2), device=x.device)
            else:
                random_num_input = torch.ones(x.size(0) * x.size(2), device=x.device)
        else: # B # original
            if random_uncond:
                random_num_input = torch.rand(x.size(0), device=x.device)
            else:
                random_num_input = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        if random_uncond:
            random_num_prompt = torch.rand(x.size(0), device=x.device)
        else:
            random_num_prompt = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference

        prompt_mask = rearrange(random_num_prompt < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num_input >= self.uncond_prob).float() * (random_num_input < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        # cond_frame_index = 0
        # if self.rand_cond_frame:
        #     cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)
        # img = x[:,:,cond_frame_index,...]

        # WARNING : enhance mode
        if self.cond_select_all:
            img = render
            img = rearrange(img, 'b c t h w -> (b t) c h w')
        else: # original
            cond_frame_index = batch['cond_frame_idx']  # 1D tensor
            batch_indices = torch.arange(x.shape[0], device=x.device)
            img = x[batch_indices, :, cond_frame_index]
            # print("[DEBUG] : img.shape = ", img.shape)
            # print("[DEBUG] : batch[cond_frame_idx] = ", batch['cond_frame_idx'], batch['cond_frame_idx'].shape)
            # print("[DEBUG] : batch[video] = ", batch['video'].shape) 
            # print("[DEBUG] : cond_frame_index = ", cond_frame_index, "input_mask.shape = ", input_mask.shape)
            # [DEBUG] : batch[cond_frame_idx] =  tensor([1, 3], device='cuda:2') torch.Size([2])                                                                                                                    
            # [DEBUG] : batch[video] =  torch.Size([2, 3, 16, 320, 512])                                                                                                                                            
            # [DEBUG] : cond_frame_index =  tensor([1, 3], device='cuda:2') input_mask.shape =  torch.Size([2, 1, 1, 1])                                                                                            
            # [DEBUG] : img.shape =  torch.Size([2, 3, 320, 512])                                                       

        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

        if 'hybrid' in self.model.conditioning_key:
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            elif self.enhance_mode:
                ## enhance the cond_frame
                img_cat_cond = z_render
                # img_cat_cond = img_cat_cond.unsqueeze(2)
                # img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = z[:,:,cond_frame_index,:,:]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

            cond["c_concat"] = [img_cat_cond] # b c t h w
        if self.cond_select_all:
            prompt_imb = repeat(prompt_imb, 'b c d -> (b t) c d', t=z.shape[2])
            print("[DEBUG] : prompt_imb.shape = ", prompt_imb.shape, "img_emb.shape = ", img_emb.shape) 
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        if return_cond_frame:
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        camera_pose = super().get_input(batch, 'camera_pose')
        out.append(camera_pose)
        
        if 'guidance' in batch.keys():
            guidances = super().get_input(batch, 'guidance')
            # print("[DEBUG] : guidances.shape = ", guidances.shape) # (B, 1, T, H, W)
            z_guidance = self.encode_guidance(guidances)
            cond["c_guidance"] = z_guidance
        
        control = super().get_input(batch, 'control').to(self.device)
        control = einops.rearrange(control, 'b t c h w -> b c t h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        out.append(control)
        
        return out

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        # if cond['c_concat'] is None:
        #     eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        # else:
        # from DiffusionWrapper
        xc = torch.cat([x_noisy] + cond['c_concat'], dim=1)
        cc = torch.cat(cond['c_crossattn'], 1)
        # import pdb; pdb.set_trace()
        
        control = self.control_model(x=xc, hint=kwargs["control"], timesteps=t, context=cc, camera_pose=kwargs["camera_pose"])
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        kwargs.pop('control')
        eps = diffusion_model(x=xc, timesteps=t, context=cc, control=control, only_mid_control=self.only_mid_control, **kwargs)
        return eps

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=50, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, mask=None, **kwargs):
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 1
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

        z, c, xrec, xc, fs, cond_x, camera_pose, control = self.get_batch_input(batch, random_uncond=False,
                                                return_first_stage_outputs=True,
                                                return_original_cond=True,
                                                return_fs=True,
                                                return_cond_frame=True)

        N = xrec.shape[0]
        log["image_condition"] = cond_x
        log["reconst"] = xrec
        xc_with_fs = []
        for idx, content in enumerate(xc):
            xc_with_fs.append(content + '_fs=' + str(fs[idx].item()))
        log["condition"] = xc_with_fs
        kwargs.update({"fs": fs.long()})
        kwargs.update({"camera_pose": camera_pose})
        kwargs.update({"control": control})
        
        c_cat = None
        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_emb = c["c_crossattn"][0]
                    if 'c_concat' in c.keys():
                        c_cat = c["c_concat"][0]
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc_prompt = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc_prompt = torch.zeros_like(c_emb)
                
                img = torch.zeros_like(xrec[:,:,0]) ## b c h w
                ## img: b c h w
                img_emb = self.embedder(img) ## b l c
                uc_img = self.image_proj_model(img_emb)

                uc = torch.cat([uc_prompt, uc_img], dim=1)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            if self.use_guidance is not None:
                if "gating-attn" in self.use_guidance:
                    c_guidance = c["c_guidance"]
                    if "features_adapter" not in kwargs.keys():
                        # print("[DEBUG] : log_images : update features_adapter")
                        kwargs.update({"features_adapter": c_guidance})
                    else:
                        # print("[DEBUG] : log_images : change features_adapter")
                        kwargs["features_adapter"] = c_guidance
                    
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
        return log

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        # params = list(self.model.parameters())
        params = list(self.control_model.parameters())
        mainlogger.info(f"@Training ControlModel [{len(params)}] Full Paramters.")

        # if self.cond_stage_trainable:
        #     params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
        #     mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
        #     params.extend(params_cond_stage)
        
        # if self.image_proj_model_trainable:
        #     mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
        #     params.extend(list(self.image_proj_model.parameters()))   

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer
