"""Construct local random Diffusers and Transformers checkpoints for offline integration."""

import json

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from transformers import (
    BertTokenizerFast,
    BitImageProcessor,
    BlipConfig,
    BlipForQuestionAnswering,
    BlipImageProcessor,
    BlipProcessor,
    CLIPConfig,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    Dinov2Config,
    Dinov2Model,
    T5Config,
    T5EncoderModel,
    T5TokenizerFast,
)


def make_local_models(root):
    """Small random local checkpoints can exercise model APIs without downloading weights."""
    import sentencepiece as spm

    vocab = (
        ["<|startoftext|>", "<|endoftext|>"]
        + list("abcdefghijklmnopqrstuvwxyz")
        + [c + "</w>" for c in "abcdefghijklmnopqrstuvwxyz"]
    )
    (root / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}))
    (root / "merges.txt").write_text("#version: 0.2\n")
    tok = CLIPTokenizer(str(root / "vocab.json"), str(root / "merges.txt"), model_max_length=77)
    text_cfg = CLIPTextConfig(
        vocab_size=len(tok),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=77,
        projection_dim=8,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
    )
    corpus = root / "corpus.txt"
    corpus.write_text(
        "Exactly two red cups to the left of one box. A green bottle above a basket.\n"
    )
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(root / "spiece"),
        vocab_size=48,
        hard_vocab_limit=False,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        minloglevel=2,
    )
    t5tok = T5TokenizerFast(vocab_file=str(root / "spiece.model"), extra_ids=0)
    t5 = T5EncoderModel(
        T5Config(
            vocab_size=len(t5tok),
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_heads=2,
            d_kv=8,
            dropout_rate=0.0,
        )
    )
    vae = AutoencoderKL(
        block_out_channels=(8, 8),
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        norm_num_groups=4,
        latent_channels=4,
        sample_size=32,
        scaling_factor=1.0,
        shift_factor=0.1,
    )
    net = SD3Transformer2DModel(
        sample_size=16,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        joint_attention_dim=16,
        caption_projection_dim=16,
        pooled_projection_dim=16,
        pos_embed_max_size=16,
    )
    pipeline = StableDiffusion3Pipeline(
        net,
        FlowMatchEulerDiscreteScheduler(),
        vae,
        CLIPTextModelWithProjection(text_cfg),
        tok,
        CLIPTextModelWithProjection(text_cfg),
        tok,
        t5,
        t5tok,
    )
    pipeline.save_pretrained(root / "sd3")

    bt = root / "bert.txt"
    bt.write_text(
        "\n".join(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "no",
                "yes",
                "red",
                "blue",
                "cup",
                "is",
                "there",
                "the",
                "image",
                "?",
            ]
        )
    )
    blip_processor = BlipProcessor(
        BlipImageProcessor(size={"height": 16, "width": 16}), BertTokenizerFast(vocab_file=str(bt))
    )
    blip_cfg = BlipConfig(
        text_config=dict(
            vocab_size=len(blip_processor.tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
            bos_token_id=2,
        ),
        vision_config=dict(
            image_size=16,
            patch_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
        ),
    )
    BlipForQuestionAnswering(blip_cfg).save_pretrained(root / "vqa")
    blip_processor.save_pretrained(root / "vqa")
    BlipForQuestionAnswering(blip_cfg).save_pretrained(root / "eval_vqa")
    blip_processor.save_pretrained(root / "eval_vqa")
    size = {"shortest_edge": 16}
    crop = {"height": 16, "width": 16}
    Dinov2Model(
        Dinov2Config(
            image_size=16, patch_size=4, hidden_size=16, num_hidden_layers=1, num_attention_heads=2
        )
    ).save_pretrained(root / "dino")
    BitImageProcessor(size=size, crop_size=crop).save_pretrained(root / "dino")
    clip_cfg = CLIPConfig(
        text_config=text_cfg.to_dict(),
        vision_config=dict(
            image_size=16,
            patch_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
        ),
        projection_dim=8,
    )
    # Independently initialized evaluation checkpoints prevent accidental reward reuse in the smoke.
    Dinov2Model(Dinov2Config.from_pretrained(root / "dino")).save_pretrained(root / "eval_dino")
    BitImageProcessor.from_pretrained(root / "dino").save_pretrained(root / "eval_dino")
    CLIPModel(clip_cfg).save_pretrained(root / "eval_pick")
    CLIPProcessor(CLIPImageProcessor(size=size, crop_size=crop), tok).save_pretrained(
        root / "eval_pick"
    )
    CLIPModel(clip_cfg).save_pretrained(root / "pick")
    CLIPProcessor(CLIPImageProcessor(size=size, crop_size=crop), tok).save_pretrained(root / "pick")
