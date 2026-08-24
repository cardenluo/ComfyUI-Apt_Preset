import logging

import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.utils
import node_helpers

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_motion_context")

MC_KEY = "motion_context_index"
MC_AUDIO_KEY = "motion_context_audio_end_frame"

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)
FPS = 24
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0

ENCODE_MODE = "video"
ANCHOR_MODE = "head"
AUDIO_MODE = "timeline"
CROP = "disabled"


class AptMiniMaxH3NativeAudioLock:
    """Lock exact user audio into an H3 AV latent and denoise video only."""

    def lock_audio(self, model, av_latent, audio_vae, audio):
        samples = av_latent.get("samples")
        if samples is None or not getattr(samples, "is_nested", False):
            raise ValueError("AptMiniMaxH3NativeAudioLock requires a joint MiniMax H3 AV latent")

        video_latent, target_audio_template = samples.unbind()[:2]
        waveform = audio["waveform"][:1]
        sample_rate = int(audio["sample_rate"])
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sample_rate != vae_rate:
            if torchaudio is None:
                raise RuntimeError("AptMiniMaxH3NativeAudioLock needs torchaudio to resample audio")
            waveform = torchaudio.functional.resample(waveform, sample_rate, vae_rate)

        exact_audio_latent = audio_vae.encode(waveform.movedim(1, -1))
        target_t = target_audio_template.shape[-1]
        if exact_audio_latent.shape[-1] > target_t:
            exact_audio_latent = exact_audio_latent[..., :target_t]
        elif exact_audio_latent.shape[-1] < target_t:
            exact_audio_latent = F.pad(exact_audio_latent, (0, target_t - exact_audio_latent.shape[-1]))

        locked = dict(av_latent)
        locked["samples"] = comfy.nested_tensor.NestedTensor((video_latent, exact_audio_latent))
        locked["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video_latent), torch.zeros_like(exact_audio_latent))
        )

        patched_model = model.clone()
        transformer_options = patched_model.model_options["transformer_options"] = (
            patched_model.model_options.get("transformer_options", {}).copy()
        )
        transformer_options["minimax_h3_lock_audio_clean"] = True
        return patched_model, locked, audio


def _ensure_layout_patch():
    if _layout_patch_applied():
        return
    if not _apply_layout_patch():
        raise RuntimeError(
            "h3_motion_context: the layout patch could not be applied, so "
            "interior anchors would be rejected by ComfyUI. The reason was "
            "logged just above this error.")


def _ensure_payload_patch():
    if _payload_patch_applied():
        return
    if not _apply_payload_patch():
        raise RuntimeError(
            "h3_motion_context: the payload patch could not be applied. "
            "Without it the audio ref would overwrite the pinned video "
            "latents and the motion context would be lost. The reason was "
            "logged just above this error.")


def h3_keyframe_anchor(position):
    """Return a first/last anchor compatible with native and patched layouts."""
    _ensure_layout_patch()
    position = int(position)
    if _layout_native:
        return {"resolved_frame_index": position}
    return {"resolved_frame_index": 0, MC_KEY: position}


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_tail_audio(audio_vae, audio, seconds):
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                "h3_motion_context: context_audio is %d Hz but the VAE wants %d Hz "
                "and torchaudio is not available to resample." % (sr, vae_sr))
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have < want:
        _LOG.warning("h3_motion_context: context_audio is %.3fs, shorter than the "
                     "%.3fs of pinned video. Pinning what there is.",
                     have / vae_sr, seconds)
    else:
        waveform = waveform[..., have - want:]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, int(z.shape[-1])


def _streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_motion_context: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3_motion_context: AV latent contains no streams")
    return parts


def _video_from_latent(latent):
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3_motion_context: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


def _steps_for_frames(n):
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def _video_tail_from_latent(latent, n):
    video = _video_from_latent(latent)
    total = int(video.shape[2])
    steps = _steps_for_frames(n)
    if steps is None:
        raise ValueError(
            "h3_motion_context: a %d frame window is not a whole number of "
            "latent steps, so it cannot be sliced from a latent. Use 5, 22, "
            "39 or 56, or unwire context_latent to encode pixels." % n)
    if steps > total:
        raise ValueError(
            "h3_motion_context: asked for %d latent steps, context_latent "
            "has %d." % (steps, total))
    start = total - steps
    if start % 5 != 0:
        raise RuntimeError(
            "h3_motion_context: the %d step tail of a %d step latent starts "
            "at cycle position %d, not 0, so its frame spans would not match "
            "the positions written for them. Clip lengths are meant to make "
            "this impossible; refusing rather than rendering a shifted join."
            % (steps, total, start % 5))
    covered = _pixel_frames(steps)
    if covered != n:
        raise RuntimeError(
            "h3_motion_context: %d steps cover %d frames, expected %d."
            % (steps, covered, n))
    blocks = [video[:1, :, start + k:start + k + 1].clone()
              for k in range(steps)]
    return blocks, _step_offsets(steps), covered


def _audio_tail_from_latent(latent, a_frames):
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context: context_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("h3_motion_context: expected audio latent [B,C,2,T], "
                         "got shape %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    # H3 rounds the audio grid to the nearest step. Depending on the clip
    # length the latent may overrun or stop short by one third of a step.
    if not (-0.5 < overhang < 0.5):
        _LOG.warning(
            "h3_motion_context: context_latent audio grid is unexpected "
            "(%d steps for %d frames); assuming no overhang.", total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        _LOG.warning("h3_motion_context: asked for %d audio steps, the latent "
                     "has %d. Pinning all of it.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("h3_motion_context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


class AptMiniMaxH3MotionContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "trim_frames": ("INT", {
                    "default": 22,
                    "min": 5,
                    "max": 56,
                    "step": 17}),
            },
            "optional": {
                "context_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = "Apt_Preset/MiniMax H3"

    def apply(self, conditioning, latent, trim_frames=22,
              context_latent=None, audio_context_length=24):
        if context_latent is None:
            return (conditioning, 0)

        context_length = int(trim_frames)
        # Video and audio use independent windows. 22 video frames gives a
        # whole H3 video-latent run; 24 audio frames is exactly one second
        # and lands exactly on the 40 Hz audio grid.
        audio_context_length = int(audio_context_length)
        vae = None
        context_frames = None
        audio_vae = None
        context_audio = None
        encode_mode, anchor_mode = ENCODE_MODE, ANCHOR_MODE
        audio_mode, crop = AUDIO_MODE, CROP
        _ensure_layout_patch()

        video = _video_from_latent(latent)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        if context_latent is not None:
            src_video = _video_from_latent(context_latent)
            src_w = int(src_video.shape[4]) * 16
            src_h = int(src_video.shape[3]) * 16
            if src_w != width or src_h != height:
                raise ValueError(
                    "h3_motion_context: context_latent is %dx%d but this "
                    "clip is %dx%d. A latent cannot be resized, so the "
                    "previous clip has to be regenerated at this "
                    "resolution, or the chain restarted here."
                    % (src_w, src_h, width, height))
            if int(src_video.shape[1]) != int(video.shape[1]):
                raise ValueError(
                    "h3_motion_context: context_latent has %d channels, "
                    "this clip has %d. That is not an H3 video latent from "
                    "the same model."
                    % (int(src_video.shape[1]), int(video.shape[1])))
            available = _pixel_frames(int(src_video.shape[2]))
            video_src = "latent"
        else:
            if context_frames is None:
                raise ValueError(
                    "h3_motion_context: nothing to pin. Wire context_latent "
                    "(preferred) or context_frames.")
            available = int(context_frames.shape[0])
            video_src = "pixels"

        n = min(int(context_length), available)
        if n < 1:
            raise ValueError("h3_motion_context: no frames available to pin")
        if n < context_length:
            _LOG.warning("h3_motion_context: only %d frames available, pinning %d",
                         available, n)

        if encode_mode == "video":
            run = next(g for g in VIDEO_RUN_GRID if g <= n)
            if run != n:
                _LOG.warning(
                    "h3_motion_context: %d frames is off the VAE grid; pinning "
                    "the last %d instead (usable runs: 1, 5, 22, 39, 56)", n, run)
            n = run

        if n >= frame_count:
            raise ValueError(
                "h3_motion_context: asked to pin %d frames into a %d frame clip. "
                "The pinned run must be a small fraction of the timeline."
                % (n, frame_count))

        if video_src == "latent" and _steps_for_frames(n) is None:
            raise RuntimeError(
                "h3_motion_context: a %d frame window is not a whole number "
                "of latent steps. VIDEO_RUN_GRID no longer matches the "
                "VAE; refusing rather than rendering a shifted join." % n)

        if video_src == "latent":
            blocks, offsets, covered = _video_tail_from_latent(
                context_latent, n)
            span = covered
        else:
            tail = _resize(context_frames[available - n:], width, height, crop)

        if video_src == "pixels" and encode_mode == "video":
            enc = vae.encode(tail)
            if getattr(enc, "ndim", 0) != 5:
                raise ValueError(
                    "h3_motion_context: video-mode encode returned shape %s, "
                    "expected [B,C,T,H,W]. Try encode_mode=frames."
                    % (tuple(getattr(enc, "shape", ())),))
            steps = int(enc.shape[2])
            offsets = _step_offsets(steps)
            covered = _pixel_frames(steps)
            if covered != n:
                raise RuntimeError(
                    "h3_motion_context: %d frames encoded to %d latent steps "
                    "covering %d frames; the VAE grid no longer matches "
                    "VIDEO_RUN_GRID. Upstream VAE change, refusing to run."
                    % (n, steps, covered))
            blocks = [enc[:, :, k:k + 1] for k in range(steps)]
            span = covered
        elif video_src == "pixels":
            blocks, offsets = [], []
            for i in range(n):
                blocks.append(vae.encode(tail[i:i + 1]))
                offsets.append(i)
            span = n

        if anchor_mode == "before":
            indices = [o - span for o in offsets]
        else:
            indices = list(offsets)

        keyframes = []
        for p, blk in zip(indices, blocks):
            kf = {"latent": blk}
            if _layout_native:
                kf["resolved_frame_index"] = p
            else:
                kf["resolved_frame_index"] = 0
                kf[MC_KEY] = p
            keyframes.append(kf)

        ref_audio_t = 0
        audio_ref = None
        a_frames = 0
        audio_src = "off"
        if context_latent is not None or context_audio is not None:
            _ensure_payload_patch()
            a_frames = int(audio_context_length) or span
            if context_latent is not None:
                if context_audio is not None:
                    _LOG.info("h3_motion_context: both context_latent and "
                              "context_audio wired; using the latent (skips "
                              "one VAE round trip).")
                audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                    context_latent, a_frames)
                audio_src = "latent"
            else:
                if audio_vae is None:
                    raise ValueError(
                        "h3_motion_context: context_audio supplied without "
                        "audio_vae. Wire the H3 audio VAE, or wire "
                        "context_latent instead.")
                audio_latent, ref_audio_t = _encode_tail_audio(
                    audio_vae, context_audio, a_frames / float(FPS))
                overhang = 0.0
                audio_src = "vae"
            ref = {
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent,
            }
            if audio_mode == "timeline":
                if _layout_native:
                    _LOG.info("h3_motion_context: audio_mode=timeline requires "
                              "the wrapper patch; stock ref placement used "
                              "instead (audio still pinned, just on the ref path)")
                else:
                    end_frame = float(span if anchor_mode == "head" else 0)
                    end_frame += overhang / FRAME_RESCALE
                    end_coord = round(FRAME_RESCALE * end_frame)
                    end_frame = end_coord / FRAME_RESCALE
                    ref[MC_AUDIO_KEY] = end_frame
            audio_ref = ref

        # Merge with upstream keyframes instead of replacing them. In
        # particular, keep a valid last-frame target from the H3 guide while
        # dropping anchors that conflict with the pinned head.
        head_end = span if anchor_mode == "head" else 0
        out = []
        dropped = []
        for emb, extra in conditioning:
            d = extra.copy()
            prior = d.get("minimax_keyframes") or []
            prior_frame_count = d.get("minimax_frame_count")
            if (prior and prior_frame_count is not None
                    and int(prior_frame_count) != frame_count):
                raise ValueError(
                    "h3_motion_context: the conditioning carries keyframes "
                    "resolved for a %d frame clip, but the latent is %d "
                    "frames. Wire the conditioning and the latent from the "
                    "same node." % (int(prior_frame_count), frame_count))
            kept = []
            for keyframe in prior:
                position = int(keyframe.get(
                    MC_KEY, keyframe.get("resolved_frame_index", 0)))
                if position < head_end:
                    dropped.append(position)
                    continue
                keyframe = dict(keyframe)
                if _layout_native:
                    keyframe["resolved_frame_index"] = position
                    keyframe.pop(MC_KEY, None)
                else:
                    keyframe[MC_KEY] = position
                kept.append(keyframe)
            d["minimax_keyframes"] = kept + keyframes
            d["minimax_frame_count"] = frame_count
            out.append([emb, d])
        if dropped:
            _LOG.warning(
                "h3_motion_context: dropped %d keyframe anchor(s) at "
                "frame(s) %s: the pinned head already decides frames "
                "0..%d. A last_frame anchor is kept.",
                len(dropped), sorted(set(dropped)), head_end - 1)

        if audio_ref is not None:
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": [audio_ref]}, append=True)

        trim = span if anchor_mode == "head" else 0
        audio_end_frame = (audio_ref.get(MC_AUDIO_KEY)
                           if ref_audio_t and audio_ref is not None else None)
        if audio_end_frame is not None:
            audio_place = "on the timeline ending at frame %.3f" % float(audio_end_frame)
        elif audio_mode == "timeline" and _layout_native:
            audio_place = "timeline mode requested but stock ref placement used (native interior anchors)"
        else:
            audio_place = "stock ref placement"
        _LOG.info("h3_motion_context: video from %s, %s/%s, %d frames -> %d "
                  "cond blocks at indices %d..%d, %d frame clip at %dx%d, "
                  "trim %d, audio %s",
                  video_src, encode_mode, anchor_mode, n, len(blocks),
                  indices[0], indices[-1], frame_count, width, height, trim,
                  ("%d frames -> %d latent steps (%.3fs) from %s, %s"
                   % (a_frames, ref_audio_t, ref_audio_t / AUDIO_HZ, audio_src,
                      audio_place))
                  if ref_audio_t else "off")
        return (out, trim)





import torch

import comfy.ldm.minimax.model as mm

_LAYOUT_PATCH_MARKER = "_h3_motion_context_layout_patch"

_layout_orig_init = None
_layout_applied = False
_layout_native = False

REF_SEGMENT_KINDS = ("ref_img", "ref_audio")


def _target_origin(layout):
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            "h3_motion_context: expected the target video rows to be the "
            "last layout segment, found %r spanning %d rows. Upstream "
            "layout change; refusing to rewrite positions." % (kind, b - a))
    return float(layout.position_ids[a, 0])


def _expected_ref_segments(blk):
    kind = blk.get("kind")
    if kind == "image":
        return ("ref_img",)
    if kind == "audio":
        return ("ref_audio",) if int(blk.get("ref_audio_t", 0)) > 0 else ()
    if kind in ("video", "video_audio"):
        if int(blk.get("ref_audio_t", 0)) > 0:
            return ("ref_audio", "ref_img")
        return ("ref_img",)
    raise RuntimeError(
        "h3_motion_context: unknown reference kind %r; cannot tell which "
        "layout rows belong to it." % (kind,))


def _ref_segment_map(layout, refs):
    ref_segs = [(a, b, k) for a, b, k in layout.segments
                if k in REF_SEGMENT_KINDS]
    want = [(i, k) for i, blk in enumerate(refs or [])
            for k in _expected_ref_segments(blk)]
    if len(want) != len(ref_segs):
        raise RuntimeError(
            "h3_motion_context: %d reference blocks should have produced %d "
            "layout segments, the layout has %d. Upstream layout change; "
            "refusing to move rows." % (len(refs or []), len(want),
                                        len(ref_segs)))
    out = {}
    for (i, kind), (a, b, got) in zip(want, ref_segs):
        if got != kind:
            raise RuntimeError(
                "h3_motion_context: reference block %d (%r) should have "
                "emitted a %s segment, the layout has %s. Upstream layout "
                "change; refusing to move rows."
                % (i, refs[i].get("kind"), kind, got))
        out.setdefault(i, {})[kind] = (a, b)
    return out


def _cond_t(text_len, latent_t, frame_count, p):
    if p == 0:
        return float(text_len)
    if frame_count is not None and p == frame_count - 1:
        return float(text_len) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
    return float(text_len) + mm.FRAME_RESCALE * float(p)


def _fixup(layout, text_len, latent_t, frame_count, keyframes, refs=None):
    offset = _target_origin(layout) - float(text_len)
    if offset and any(kf.get(MC_KEY) is None for kf in keyframes):
        raise RuntimeError(
            "h3_motion_context: stock and motion-context keyframes mixed in "
            "one graph alongside a ref; their coordinates would disagree. "
            "Give every keyframe a %s entry or remove the refs." % MC_KEY)
    cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            "h3_motion_context: expected %d cond segments, layout has %d. "
            "Refusing to rewrite positions."
            % (len(keyframes), len(cond_spans)))
    for (a, b), kf in zip(cond_spans, keyframes):
        p = kf.get(MC_KEY)
        if p is None:
            continue
        layout.position_ids[a:b, 0] = _cond_t(text_len, latent_t, frame_count, p) + offset


def _fixup_audio(layout, text_len, refs):
    marked = [i for i, r in enumerate(refs or [])
              if r.get(MC_AUDIO_KEY) is not None]
    if len(marked) != 1:
        raise RuntimeError(
            "h3_motion_context: audio timeline placement needs exactly one "
            "reference marked with %s; the layout has %d references and %d "
            "marked. If this appeared during startup, check for more than "
            "one H3 Motion Context folder in custom_nodes."
            % (MC_AUDIO_KEY, len(refs or []), len(marked)))
    idx = marked[0]
    blk = refs[idx]
    if blk.get("kind") != "audio":
        raise RuntimeError(
            "h3_motion_context: %s set on a %r ref; only audio refs can be "
            "moved onto the timeline." % (MC_AUDIO_KEY, blk.get("kind")))
    rt = int(blk.get("ref_audio_t", 0))
    if rt <= 0:
        return

    seg = _ref_segment_map(layout, refs).get(idx, {}).get("ref_audio")
    if seg is None:
        raise RuntimeError(
            "h3_motion_context: the marked audio reference produced no "
            "ref_audio segment. Upstream layout change; refusing to move "
            "rows.")
    a, b = seg
    if b - a != 2 * rt:
        raise RuntimeError(
            "h3_motion_context: the marked audio reference has %d rows for "
            "%d latent steps, expected %d (stereo, channel-major). Upstream "
            "layout change; refusing to move rows." % (b - a, rt, 2 * rt))

    target_origin = _target_origin(layout)
    slot_start = float(layout.position_ids[a, 0])
    end_frame = float(blk[MC_AUDIO_KEY])
    desired_start = target_origin + mm.FRAME_RESCALE * end_frame - float(rt)
    layout.position_ids[a:b, 0] = (layout.position_ids[a:b, 0]
                                   + (desired_start - slot_start))


def _patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                  keyframes=None, refs=None, frame_count=None):
    _call_layout_init(_layout_orig_init, self, text_len, latent_t, latent_h,
                      latent_w, audio_t, keyframes=keyframes, refs=refs,
                      frame_count=frame_count)
    has_mc_kf = bool(keyframes) and any(
        kf.get(MC_KEY) is not None for kf in keyframes)
    has_mc_audio = bool(refs) and any(
        r.get(MC_AUDIO_KEY) is not None for r in refs)
    if has_mc_kf:
        _fixup(self, text_len, latent_t, frame_count, keyframes, refs)
    if has_mc_audio:
        _fixup_audio(self, text_len, refs)


def _layout_self_test():
    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    frame_count = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(latent_t))

    def build(keyframes=None, refs=None, fix=False, move=False):
        lay = mm.PackedLayout.__new__(mm.PackedLayout)
        _call_layout_init(_layout_orig_init, lay, text_len, latent_t, lh, lw,
                          audio_t, keyframes=keyframes, refs=refs,
                          frame_count=frame_count)
        if fix:
            _fixup(lay, text_len, latent_t, frame_count, keyframes, refs)
        if move:
            _fixup_audio(lay, text_len, refs)
        return lay

    def cond_ts(lay):
        return [float(lay.position_ids[a, 0])
                for a, _, k in lay.segments if k == "cond"]

    stock_kf = [{"resolved_frame_index": 0},
                {"resolved_frame_index": frame_count - 1}]
    ours_kf = [{"resolved_frame_index": 0, MC_KEY: 0},
               {"resolved_frame_index": 0, MC_KEY: frame_count - 1}]
    a = build(keyframes=stock_kf)
    b = build(keyframes=ours_kf, fix=True)
    if a.position_ids.shape != b.position_ids.shape:
        raise RuntimeError("position_ids shape mismatch in self-test")
    if not torch.equal(a.position_ids, b.position_ids):
        bad = (a.position_ids != b.position_ids).any(dim=1).nonzero().flatten()
        raise RuntimeError("position mismatch at rows %s" % bad[:8].tolist())

    run = [{"resolved_frame_index": 0, MC_KEY: i} for i in range(4)]
    c = build(keyframes=run, fix=True)
    ts = cond_ts(c)
    if len(ts) != len(run):
        raise RuntimeError("expected %d cond segments, got %d" % (len(run), len(ts)))
    if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
        raise RuntimeError("consecutive anchors not strictly increasing: %s" % ts)
    t_last = float(text_len) + mm.FRAME_RESCALE * (frame_count - 1)
    if not (ts[0] == float(text_len) and ts[-1] < t_last):
        raise RuntimeError("run %s escapes the [%.4f, %.4f] span"
                           % (ts, float(text_len), t_last))

    ref = [{"kind": "audio", "ref_audio_t": 8}]
    d = build(keyframes=run, refs=ref, fix=True)
    ts_ref = cond_ts(d)
    if len(ts_ref) != len(ts):
        raise RuntimeError("cond segment count changed when a ref was added")
    tol = 1e-3
    gap = float(c.position_ids[:, 0].max()) - ts[0]
    gap_ref = float(d.position_ids[:, 0].max()) - ts_ref[0]
    if abs(gap - gap_ref) > tol:
        raise RuntimeError(
            "ref compensation off by %.6f: anchor-to-target gap %.6f without "
            "ref, %.6f with. The target origin read back from the layout no "
            "longer matches its cursor arithmetic." % (gap_ref - gap, gap, gap_ref))
    shifts = [y - x for x, y in zip(ts, ts_ref)]
    if any(abs(sh - shifts[0]) > tol for sh in shifts):
        raise RuntimeError("ref shifted anchors unevenly: %s" % shifts)

    end_frame, rt = 4, 8
    ref_mc = [{"kind": "audio", "ref_audio_t": rt, MC_AUDIO_KEY: end_frame}]
    e = build(keyframes=run, refs=ref_mc, fix=True, move=True)
    _check_move(d, e, ref_mc, 0, "single-ref")

    r_lh, r_lw, r_vt = 8, 12, 3
    others = [
        {"kind": "image", "latent_h": r_lh, "latent_w": r_lw},
        {"kind": "video_audio", "latent_h": r_lh, "latent_w": r_lw,
         "latent_t": r_vt, "ref_audio_t": 5},
        {"kind": "audio", "ref_audio_t": 3},
    ]
    marked = {"kind": "audio", "ref_audio_t": rt, MC_AUDIO_KEY: end_frame}
    plain = {"kind": "audio", "ref_audio_t": rt}
    multi_plain = others[:2] + [plain] + others[2:]
    multi_marked = others[:2] + [marked] + others[2:]
    f = build(keyframes=run, refs=multi_plain, fix=True)
    g = build(keyframes=run, refs=multi_marked, fix=True, move=True)
    _check_move(f, g, multi_marked, 2, "multi-ref")

    smap = _ref_segment_map(f, multi_plain)
    prev_hi = float(text_len) - 1e-9
    origin = _target_origin(f)
    for i in range(len(multi_plain)):
        spans = smap.get(i)
        if not spans:
            continue
        rows = [r for a0, b0 in spans.values() for r in range(a0, b0)]
        lo = min(float(f.position_ids[r, 0]) for r in rows)
        hi = max(float(f.position_ids[r, 0]) for r in rows)
        if lo < prev_hi - 1e-9:
            raise RuntimeError(
                "reference block %d starts at %.6f, before block %d ended "
                "at %.6f. Reference blocks are not laid out in list order."
                % (i, lo, i - 1, prev_hi))
        if hi >= origin - 1e-9:
            raise RuntimeError(
                "reference block %d reaches %.6f, at or past the target "
                "origin %.6f. Reference rows should sit before the target."
                % (i, hi, origin))
        prev_hi = hi


def _check_move(before, after, refs, idx, label):
    if after.position_ids.shape != before.position_ids.shape:
        raise RuntimeError("%s: audio move changed the layout shape" % label)
    if not torch.equal(before.position_ids[:, 1:], after.position_ids[:, 1:]):
        raise RuntimeError(
            "%s: audio move touched a non-time coordinate column" % label)
    a, b = _ref_segment_map(before, refs)[idx]["ref_audio"]
    expect_moved = set(range(a, b))
    tb, ta = before.position_ids[:, 0], after.position_ids[:, 0]
    moved = set(i for i in range(len(tb)) if float(tb[i]) != float(ta[i]))
    if not moved:
        raise RuntimeError("%s: audio move moved no rows" % label)
    if moved != expect_moved:
        raise RuntimeError(
            "%s: audio move touched the wrong rows: %d moved, %d expected, "
            "e.g. %s" % (label, len(moved), len(expect_moved),
                         sorted(moved ^ expect_moved)[:8]))
    deltas = [float(ta[i]) - float(tb[i]) for i in sorted(moved)]
    if any(abs(dd - deltas[0]) > 1e-9 for dd in deltas):
        raise RuntimeError("%s: audio rows shifted non-uniformly: %s"
                           % (label, deltas[:4]))

    blk = refs[idx]
    rt = int(blk["ref_audio_t"])
    want_end = (_target_origin(after)
                + mm.FRAME_RESCALE * float(blk[MC_AUDIO_KEY]))
    got_end = float(after.position_ids[a, 0]) + float(rt)
    if abs(got_end - want_end) > 1e-9:
        raise RuntimeError(
            "%s: audio window ends at %.6f, should end at %.6f"
            % (label, got_end, want_end))


setattr(_patched_init, _LAYOUT_PATCH_MARKER, True)


def _call_layout_init(init, self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=None, refs=None, frame_count=None):
    """Call PackedLayout.__init__, forwarding frame_count only when the
    stock constructor accepts it.

    ComfyUI 0.33.0+ generalised the keyframe position formula and removed
    the frame_count parameter. Passing it on those builds raises
    TypeError on an unexpected keyword argument, so probe the signature
    (with a fallback test call) and strip it when unsupported.
    """
    import inspect
    kwargs = dict(keyframes=keyframes, refs=refs)
    send_fc = False
    try:
        sig = inspect.signature(init)
        send_fc = "frame_count" in sig.parameters
    except (ValueError, TypeError):
        pass
    if not send_fc:
        try:
            init(self, text_len, latent_t, latent_h, latent_w, audio_t, **kwargs)
            return
        except TypeError:
            send_fc = True
            # fall through to retry with frame_count (older build)
    if frame_count is not None:
        kwargs["frame_count"] = frame_count
    init(self, text_len, latent_t, latent_h, latent_w, audio_t, **kwargs)


_H3_INTERIOR_SOURCE_MARK = "FRAME_RESCALE * float(pixel_index)"
_H3_INTERIOR_CLASS_MARK = "_h3_interior_keyframes_patched"


def _layout_source_has_interior_form(cls):
    """Detect source-level (exec/monkey-patch) interior-anchor rewrites.

    Packs like H3-Multishot do ``inspect.getsource`` + string replace +
    ``exec`` instead of wrapping the constructor. Their output is invisible
    to wrapper markers but leaves a clear fingerprint in the source text.
    """
    init = getattr(cls, "__init__", None)
    if init is None:
        return False
    try:
        import inspect
        raw = inspect.getsource(init)
    except Exception:
        # frozen / built-in / bytecode-only: fall back to attribute marks
        return bool(getattr(cls, _H3_INTERIOR_CLASS_MARK, False))
    if _H3_INTERIOR_SOURCE_MARK in raw:
        return True
    if "only first/last keyframe anchors are supported" not in raw:
        # No stock rejection branch → someone already generalised it
        # (hand-patched core build, exec patch, different pack)
        return bool(getattr(cls, _H3_INTERIOR_CLASS_MARK, False))
    return False


def _layout_already_patched():
    cls = getattr(mm, "PackedLayout", None)
    init = getattr(cls, "__init__", None)
    if init is None:
        return None
    if getattr(init, _LAYOUT_PATCH_MARKER, False):
        return "same"
    if getattr(init, "__name__", "") == "_patched_init":
        return "other"
    if hasattr(init, "__wrapped__"):
        return "foreign"
    home = getattr(cls, "__module__", None)
    where = getattr(init, "__module__", None)
    if home and where and where != home:
        return "foreign"
    if _layout_source_has_interior_form(cls):
        return "foreign"
    return None


def _apply_layout_patch():
    global _layout_orig_init, _layout_applied, _layout_native
    if _layout_applied:
        return True
    who = _layout_already_patched()

    def _class_has_native_interior():
        """Probe whether mm.PackedLayout actually supports interior anchors.

        Two paths count as ``native``: the stock constructor accepts an
        interior ``resolved_frame_index`` and does not require
        ``frame_count``, OR another pack has already installed a wrapper
        that our caller-side MC_KEY/MC_AUDIO_KEY paths can rely on.
        """
        if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
            return False
        import inspect
        stock_has_fc = False
        try:
            stock_sig = inspect.signature(mm.PackedLayout.__init__)
            stock_has_fc = "frame_count" in stock_sig.parameters
        except (ValueError, TypeError):
            try:
                probe = mm.PackedLayout.__new__(mm.PackedLayout)
                mm.PackedLayout.__init__(probe, 4, 3, 8, 8, 4,
                                         keyframes=None, refs=None)
                stock_has_fc = False
            except TypeError:
                try:
                    probe2 = mm.PackedLayout.__new__(mm.PackedLayout)
                    mm.PackedLayout.__init__(probe2, 4, 3, 8, 8, 4,
                                             keyframes=None, refs=None,
                                             frame_count=9)
                    stock_has_fc = True
                except TypeError:
                    stock_has_fc = False
        if not stock_has_fc:
            import torch
            try:
                probe = mm.PackedLayout(
                    4, 3, 8, 8, 4,
                    keyframes=[{"resolved_frame_index": 1,
                                "latent": torch.zeros(1, 16, 1, 8, 8)}],
                    refs=None)
                return True
            except Exception:
                return False
        return False

    if who == "foreign":
        _layout_applied = True
        # Source-level / exec-style patches generalise the keyframe formula
        # but still accept ``resolved_frame_index`` natively; they do NOT
        # consume MC_KEY markers, so our side must emit native anchors.
        _layout_native = _class_has_native_interior()
        _LOG.info(
            "h3_motion_context: another pack has already generalised H3's "
            "first/last keyframe restriction (source-level patch from %r). "
            "Apt_Preset is not installing its own wrapper; native-support "
            "probe: %s.",
            getattr(getattr(getattr(mm, "PackedLayout", None), "__init__",
                            None), "__module__", "?"),
            _layout_native)
        return True
    if who:
        _layout_applied = True
        # A wrapper-style patch is already present. The wrapper is expected
        # to consume MC_KEY/MC_AUDIO_KEY markers, so ``_layout_native`` must
        # stay False. But if the probe says native support is also present
        # (wrapper + native coexist because it fell back to a no-op), prefer
        # the native path on our side so keyframes and audio refs align.
        if _class_has_native_interior() and (
                who != "same" or not getattr(
                    getattr(mm.PackedLayout, "__init__", None),
                    "moves_audio_refs", False)):
            _layout_native = True
        if who == "same":
            _LOG.info("h3_motion_context: interior keyframe anchors already "
                      "enabled by another pack, standing down (native=%s)",
                      _layout_native)
        else:
            _LOG.warning(
                "h3_motion_context: the H3 layout patch is already installed "
                "by a DIFFERENT copy of this code (another version, or a "
                "fork). Standing down; that copy decides what the patch "
                "supports, so features added since it may be unavailable. "
                "If you have more than one H3 Motion Context folder in "
                "custom_nodes, keep one and remove the rest. Renaming a "
                "folder does not stop ComfyUI loading it. (native=%s)",
                _layout_native)
        return True
    if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
        _LOG.warning("h3_motion_context: MiniMax H3 model module missing expected "
                     "attributes, patch not applied")
        return False

    # Newer ComfyUI builds already ship the general keyframe position formula
    # (cursor + FRAME_RESCALE * resolved_frame_index) and dropped the
    # frame_count constructor argument. Our wrapper and self-test both pass
    # frame_count=... so they cannot run against that stock. Detect it and
    # skip the wrapper entirely: stock already does what the wrapper did.
    import inspect
    stock_has_fc = False  # default: assume recent ComfyUI (post 0.33.0)
    try:
        stock_sig = inspect.signature(mm.PackedLayout.__init__)
        stock_has_fc = "frame_count" in stock_sig.parameters
    except (ValueError, TypeError):
        # Signature probe failed; fall back to a real call probe. Try to
        # construct a PackedLayout WITHOUT frame_count first: if it works the
        # build is recent and native-supporting.
        try:
            probe = mm.PackedLayout.__new__(mm.PackedLayout)
            mm.PackedLayout.__init__(probe, 4, 3, 8, 8, 4, keyframes=None, refs=None)
            stock_has_fc = False
        except TypeError:
            # Rejected because refs/kwargs shape, or because frame_count is
            # required as positional. Try the other way to disambiguate.
            try:
                probe2 = mm.PackedLayout.__new__(mm.PackedLayout)
                mm.PackedLayout.__init__(probe2, 4, 3, 8, 8, 4, keyframes=None,
                                         refs=None, frame_count=9)
                stock_has_fc = True
            except TypeError:
                stock_has_fc = False
    if not stock_has_fc:
        _layout_native = True
        _layout_applied = True
        _LOG.info("h3_motion_context: stock ComfyUI already supports interior "
                  "keyframe anchors; skipping wrapper, using resolved_frame_index "
                  "directly")
        return True

    _layout_orig_init = mm.PackedLayout.__init__
    try:
        _layout_self_test()
    except Exception as exc:
        _layout_orig_init = None
        # Self-test failure is not always terminal: newer ComfyUI builds
        # might have changed the layout internals while still supporting
        # resolved_frame_index natively. Before giving up, probe one anchor
        # at an interior position directly through stock.
        try:
            probe = mm.PackedLayout(4, 3, 8, 8, 4,
                                    keyframes=[{"resolved_frame_index": 1,
                                                "latent": torch.zeros(1, 16, 1, 8, 8)}],
                                    refs=None)
        except Exception as probe_exc:
            _LOG.warning("h3_motion_context: native interior probe also "
                         "failed (%s), patch not applied. Interior keyframe "
                         "anchors unavailable.", probe_exc)
            _LOG.warning("h3_motion_context: self-test failure was (%s)", exc)
            _LOG.warning(
                "h3_motion_context: if you have more than one H3 Motion Context "
                "folder in custom_nodes (a fork, a backup, a manual clone "
                "alongside a Manager install), that is the usual cause: each "
                "copy self-tests against whichever one loaded first. Keep one "
                "and remove the rest. Renaming a folder does not stop ComfyUI "
                "loading it. Otherwise this is an upstream ComfyUI change and "
                "the message above says what moved.")
            return False
        _layout_native = True
        _layout_applied = True
        _LOG.warning("h3_motion_context: wrapper self-test failed (%s) but "
                     "the stock constructor accepted an interior "
                     "resolved_frame_index; switching to the native path.",
                     exc)
        return True
    mm.PackedLayout.__init__ = _patched_init
    _layout_applied = True
    _LOG.info("h3_motion_context: interior keyframe anchors enabled")
    return True


def _layout_patch_applied():
    return _layout_applied


import comfy.model_base as model_base

_PAYLOAD_PATCH_MARKER = "_h3_motion_context_payload_patch"

_payload_orig_extra_conds = None
_payload_applied = False


def _patched_extra_conds(self, **kwargs):
    out = _payload_orig_extra_conds(self, **kwargs)

    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out
    if not (any(MC_KEY in kf for kf in keyframes)
            or any(MC_AUDIO_KEY in r for r in refs)):
        return out

    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        _LOG.warning("h3_motion_context: could not reach the H3 payload, "
                     "keyframe latents may have been overwritten by refs")
        return out

    kf_video = [kf["latent"] for kf in keyframes if "latent" in kf]
    ref_video = [r["latent"] for r in refs if "latent" in r]
    payload["cond_video_latents"] = kf_video + ref_video
    payload["cond_audio_latents"] = [r["audio_latent"] for r in refs
                                     if r.get("audio_latent") is not None]

    fc = kwargs.get("minimax_frame_count", None)
    if fc is not None:
        payload["frame_count"] = fc
    return out


setattr(_patched_extra_conds, _PAYLOAD_PATCH_MARKER, True)


def _payload_already_patched(cls):
    fn = getattr(cls, "extra_conds", None)
    if fn is None:
        return None
    if getattr(fn, _PAYLOAD_PATCH_MARKER, False):
        return "same"
    if getattr(fn, "__name__", "") == "_patched_extra_conds":
        return "other"
    if hasattr(fn, "__wrapped__"):
        return "foreign"
    home = getattr(cls, "__module__", None)
    where = getattr(fn, "__module__", None)
    if home and where and where != home:
        return "foreign"
    return None


def _apply_payload_patch():
    global _payload_orig_extra_conds, _payload_applied
    if _payload_applied:
        return True
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _LOG.warning("h3_motion_context: MiniMaxH3.extra_conds not found, "
                     "keyframes and refs cannot be combined")
        return False
    who = _payload_already_patched(cls)
    if who == "foreign":
        _payload_applied = True
        _LOG.info(
            "h3_motion_context: another pack has already patched "
            "MiniMaxH3.extra_conds (it now comes from %r). Apt_Preset is "
            "standing down instead of wrapping the wrapper; the installed "
            "patch should already handle keyframe+ref coexistence.",
            getattr(getattr(cls, "extra_conds", None), "__module__", "?"))
        return True
    if who:
        _payload_applied = True
        if who == "same":
            _LOG.info("h3_motion_context: keyframe/ref coexistence already "
                      "enabled by another pack, standing down")
        else:
            _LOG.warning(
                "h3_motion_context: the H3 payload patch is already "
                "installed by a DIFFERENT copy of this code (another "
                "version, or a fork). Standing down. If you have more than "
                "one H3 Motion Context folder in custom_nodes, keep one and "
                "remove the rest.")
        return True
    _payload_orig_extra_conds = cls.extra_conds
    cls.extra_conds = _patched_extra_conds
    _payload_applied = True
    _LOG.info("h3_motion_context: keyframe/ref coexistence enabled")
    return True


def _payload_patch_applied():
    return _payload_applied
