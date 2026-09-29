"""Append-only Responses action dialogue, configurable horizons and real feedback.

No local access to provider KV state. A stable prefix makes caching eligible;
only returned usage can establish hits. All paid requests retain smoke budgets.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from .astra_policy import validate_actions, CAMERAS
from .bounded_policy import Model as BoundedModel, BudgetExceeded, RequestTransportError
from .provider_compat import provider_cache_usage, to_wire_payload
from .rolling_cache import annotate_history_breakpoints
from .anchored_cache import annotate_anchored_breakpoints, observation_end
from .train_tool_reference import tool_reference_messages
from .trajectory_memory import LiveTrajectoryMemory, omitted_interval
from .image_budget import assert_image_budget, reference_image_count
from .observation_images import camera_input_parts, image_encoding, image_layout_content, prompt_image, validate_prompt_image


ARM_WORKSPACE_GUIDANCE = (
    "Use the left arm in the robot's left workspace and the right arm in its right workspace. "
    'Do not cross the arms.'
)
ARM_POSTURE_GUIDANCE = (
    'Keep both arms in the selected elbow-up configuration and avoid downward elbow folding. Maintain configuration continuity throughout motion and leave a margin from full arm extension.'
)
GRIPPER_GUIDANCE = (
    'Grippers use normalized opening targets: 0 fully closed, 1 fully open; keep 0 while holding. '
    'This is position control, not a force command. Reported gripper state is the previous '
    'command, not measured aperture or grasp success; judge retention from RGB.'
)


# A single xhigh multimodal turn can legitimately take several minutes, but a
# connection that never produces HTTP headers must not hold an episode forever.
# Keep this separate from max_seconds: the latter is an optional whole-run
# budget, while every provider request needs a transport deadline.
DEFAULT_REQUEST_TIMEOUT_S = 600


@dataclass(frozen=True)
class DialogueConfig:
    predict_horizon: int = 24
    fixed_predict_horizon: bool = False
    execute_horizon: int = 24
    feedback_frames: int = 4
    include_endpoints: bool = True
    feedback_cameras: tuple = CAMERAS
    endpoint_cameras: tuple = ()
    max_turns: int | None = None
    action_space: str = 'ee_delta'
    delta_frame: str = 'world'
    max_translation_m: float = .02
    max_rotation_rad: float = .15
    max_joint_increment_rad: float = .15
    ik_continuity_guard: bool = False  # Shared launcher enables execution-side IK validation.
    max_output_tokens: int | None = None
    # Budget accounting only; this value is never sent as an API output limit.
    output_token_reservation: int = 16384
    reasoning_effort: str = "xhigh"
    max_total_tokens: int | None = None
    max_seconds: int | None = None
    max_context_text_bytes: int | None = None
    max_context_images: int | None = None
    max_context_wire_bytes: int | None = None
    max_context_reserved_tokens: int | None = None
    cache_mode: str = 'implicit'
    rolling_history_cache: bool = False  # Persistent pruned-history cache frontiers.
    preserve_reasoning: bool = True
    # Exact LIVE instructions that may consume this reference despite using a
    # variant-specific caption. The launcher supplies these per task/variant;
    # an empty tuple preserves strict reference/LIVE instruction equality.
    reference_instruction_aliases: tuple = ()
    official_task_documentation_version: str = ''
    official_task_documentation_source: str = ''
    official_task_prompt: str = ''
    demo_message_format: str = 'tool_calls'
    demo_cameras: tuple = ('cam_head',)
    demo_frame0_cameras: tuple = ()  # Optional camera override for the frame-zero TRAIN observation only.
    image_profile: str = 'thumbnail_jpeg'
    image_detail: str = 'low'
    image_layout: str = 'separate'
    include_head_calibration: bool = False  # Requires independently checked TRAIN/runtime calibration.
    visual_history_batches: int = 0  # 0: append-only RGB; >0: recent live batches, fixed TRAIN unchanged.
    max_transport_attempts: int = 1  # Opt-in retransmission may incur duplicate inference cost.
    # Paid repairs allowed at one observation. Exhaustion uses one observed-state
    # joint hold, so progress and total request cost remain bounded by step_lim.
    max_action_corrections: int = 0
    visual_history_prune_every: int = 1  # Batch removals to preserve append-only inputs between prunes.
    request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S
    reasoning_summary: str = 'omit'
    train_result_observations: bool = False
    live_anchor_count: int = 0  # 0 retains the historical RGB-window implementation.
    live_anchor_interval: int = 210  # Fallback only when private environment duration is unavailable.
    live_retain_latest: bool = False  # Reserve the final key slot for the latest complete chunk.

    def __post_init__(self):
        for field, low, high in (
            ('predict_horizon', 1, 64), ('execute_horizon', 1, 64),
            ('feedback_frames', 1, 8), ('max_turns', 1, 80),
            ('max_output_tokens', 256, 65536), ('output_token_reservation', 256, 65536), ('max_total_tokens', 1, 20000000),
            ('max_seconds', 1, 7200), ('max_context_text_bytes', 1024, 2048000),
            ('max_context_images', 3, 512), ('max_context_wire_bytes', 100000, 32000000),
            # Allow the maximum text cap plus the permitted images and output.
            ('max_context_reserved_tokens', 1024, 2048000 + 512 * 1024 + 65536),
            ('visual_history_batches', 0, 50),
            ('max_transport_attempts', 1, 3),
            ('max_action_corrections', 0, 3),
            ('visual_history_prune_every', 1, 10),
            ('request_timeout_s', 1, 900),
            ('live_anchor_count', 0, 50), ('live_anchor_interval', 1, 100000),
        ):
            value = getattr(self, field)
            if field in ('max_turns', 'max_total_tokens', 'max_seconds', 'max_context_text_bytes', 'max_context_images', 'max_context_wire_bytes', 'max_context_reserved_tokens', 'max_output_tokens') and value is None:
                continue
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{field} must be an integer in [{low},{high}]')
        if self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported reasoning_effort")
        if self.execute_horizon > self.predict_horizon:
            raise ValueError('Cannot execute more steps than predicted')
        if self.action_space not in ('ee_delta', 'joint') or self.delta_frame != 'world':
            raise ValueError('Supported: world ee_delta, or absolute joint. No implicit frame conversion.')
        if self.cache_mode not in ('implicit', 'omit'):
            raise ValueError('cache_mode must be implicit or omit for unsupported gateways')
        if self.demo_message_format != 'tool_calls':
            raise ValueError('Only the formal tool-call dialogue policy is supported')
        image_encoding(self.image_profile, self.image_detail, image_layout=self.image_layout)
        if self.image_layout == 'triptych' and (
                self.demo_message_format != 'tool_calls' or self.demo_cameras != CAMERAS
                or self.feedback_cameras != CAMERAS or self.feedback_frames != 1
                or self.endpoint_cameras or self.demo_frame0_cameras not in ((), CAMERAS)):
            raise ValueError('Triptych requires all three cameras, tool-call demos, and one current observation per feedback')
        if ((self.image_profile, self.image_detail) != ('thumbnail_jpeg', 'low')
                and self.demo_message_format != 'tool_calls'):
            raise ValueError('Non-default image settings require tool-call demonstrations')
        if (self.demo_cameras not in (('cam_head',), CAMERAS)
                or self.demo_cameras != ('cam_head',) and self.demo_message_format != 'tool_calls'):
            raise ValueError('demo_cameras requires head-only or all three cameras in tool_calls format')
        if (self.demo_frame0_cameras not in ((), ('cam_head',), CAMERAS)
                or self.demo_frame0_cameras and self.demo_message_format != 'tool_calls'):
            raise ValueError('demo_frame0_cameras requires head-only or all three cameras in tool_calls format')
        if self.reasoning_summary not in ('omit','auto'):
            raise ValueError('reasoning_summary must be omit or auto')
        if self.train_result_observations and self.demo_message_format != 'tool_calls':
            raise ValueError('TRAIN result observations require tool-call demonstrations')
        if self.live_anchor_count and (self.feedback_frames != 1 or self.endpoint_cameras):
            raise ValueError('Anchored history requires one current observation per feedback')
        if self.live_retain_latest and self.live_anchor_count < 2:
            raise ValueError('Latest-chunk retention needs an initial fixed chunk and a rolling last slot')
        if (not self.feedback_cameras or len(set(self.feedback_cameras)) != len(self.feedback_cameras)
                or any(c not in CAMERAS for c in self.feedback_cameras)):
            raise ValueError('Invalid feedback camera selection')
        if len(set(self.endpoint_cameras)) != len(self.endpoint_cameras) or any(c not in CAMERAS for c in self.endpoint_cameras):
            raise ValueError('Invalid endpoint camera selection')
        for field in ('max_translation_m', 'max_rotation_rad', 'max_joint_increment_rad'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < value <= 1:
                raise ValueError(f'Invalid {field}')
        if any(type(value) is not bool for value in (self.include_endpoints, self.preserve_reasoning, self.rolling_history_cache, self.include_head_calibration, self.fixed_predict_horizon, self.train_result_observations, self.live_retain_latest, self.ik_continuity_guard)):
            raise ValueError('Boolean fields require boolean values')
        if (not isinstance(self.reference_instruction_aliases, tuple)
                or any(not isinstance(value, str) or not value
                       for value in self.reference_instruction_aliases)
                or len(set(self.reference_instruction_aliases)) != len(self.reference_instruction_aliases)):
            raise ValueError('reference_instruction_aliases must contain unique nonempty strings')
        for field in ('official_task_documentation_version',
                      'official_task_documentation_source', 'official_task_prompt'):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f'{field} must be a string')
        if bool(self.official_task_prompt) != bool(self.official_task_documentation_version):
            raise ValueError('Official task prompt and documentation version must be set together')
        if bool(self.official_task_prompt) != bool(self.official_task_documentation_source):
            raise ValueError('Official task prompt and documentation source must be set together')


def load_config(path=None):
    raw = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    for field in ('feedback_cameras', 'endpoint_cameras', 'demo_cameras', 'demo_frame0_cameras',
                  'reference_instruction_aliases'):
        if field in raw:
            raw[field] = tuple(raw[field])
    return DialogueConfig(**raw)


def sample_steps(available, count, include_endpoints=True):
    steps = sorted(set(available))
    if not steps:
        raise ValueError('No observed frames in executed interval')
    count = min(count, len(steps))
    if count == 1:
        return [steps[-1]]
    if include_endpoints:
        indices = np.linspace(0, len(steps)-1, count).round().astype(int)
    else:
        indices = np.linspace(0, len(steps)-1, count+2)[1:-1].round().astype(int)
    return [steps[i] for i in sorted(set(indices.tolist()))]


def canonical_tool_history(outputs):
    """Replay portable input fields, retaining opaque reasoning items untouched.

    Portable continuation avoids optional response metadata. A replay passed with
    these fields omitted, but later first-turn TLS failures ruled out a proven
    causal diagnosis of gateway incompatibility.
    call_id is the action/result linkage; the output item id is not that linkage.
    Original response items remain in exchanges.jsonl for audit.
    """
    items = deepcopy(outputs)
    for item in items:
        if item.get('type') == 'function_call':
            item.pop('id', None)
            item.pop('status', None)
    return items


def without_reasoning_replay(payload):
    """Drop rejected opaque reasoning while preserving observable dialogue state.

    Function calls, execution receipts, observations, and the request to return
    encrypted reasoning on future responses remain intact. This is deliberately
    a rebuilt payload, not an identical retry of provider-rejected ciphertext.
    """
    recovered = deepcopy(payload)
    inputs = recovered.get('input')
    if not isinstance(inputs, list):
        return recovered, 0
    kept = [item for item in inputs
            if not isinstance(item, dict) or item.get('type') != 'reasoning']
    removed = len(inputs) - len(kept)
    recovered['input'] = kept
    return recovered, removed


class Model(BoundedModel):
    def __init__(self, cfg):
        self.dialogue_config = load_config(os.environ.get('ASTRA_HARNESS_CONFIG'))
        super().__init__(cfg)
        c = self.dialogue_config
        self.image_budget_plan = None
        if c.live_anchor_count and c.max_context_images is not None:
            train_images = reference_image_count(self.demo, cameras=c.demo_cameras,
                frame0_cameras=c.demo_frame0_cameras, image_layout=c.image_layout,
                include_result_observations=c.train_result_observations)
            self.image_budget_plan = assert_image_budget(train_images, c.live_anchor_count,
                c.feedback_cameras, c.image_layout, max_images=c.max_context_images,
                retain_latest=c.live_retain_latest)
            self._log('image_budget.jsonl', {'phase': 'manifest_admission', **self.image_budget_plan})
        self.max_chunk = c.predict_horizon
        self.max_output = c.max_output_tokens or c.output_token_reservation
        self.effort = os.environ.get("ASTRA_REASONING_EFFORT", c.reasoning_effort)
        self.max_requests = None if c.max_turns is None else c.max_turns * c.max_transport_attempts
        self.max_total_tokens = c.max_total_tokens
        self.max_seconds = c.max_seconds
        self.request_timeout_s = c.request_timeout_s
        prediction_contract = (
            f'Predict exactly {c.predict_horizon} steps per act; execute at most {c.execute_horizon}. '
            if c.fixed_predict_horizon else
            f'Predict 1..{c.predict_horizon} steps per act; execute at most {c.execute_horizon}. '
        )
        action_contract = (
            'action_space=ee_delta: each arm6=[dx,dy,dz,rx,ry,rz], world translation in meters '
            'and world rotation-vector increments in radians, not Euler angles. Start from the '
            'latest observed EE pose; within a chunk accumulate from the previous requested '
            'target: p_next=p_previous+dxyz, q_next=exp(rotvec)*q_previous. '
            f'Per step, translation norm <= {c.max_translation_m}m and rotation norm <= {c.max_rotation_rad}rad. '
            if c.action_space == 'ee_delta' else
            'action_space=joint: each arm6 is an absolute joint target in radians. '
            f'Per step, joint change <= {c.max_joint_increment_rad}rad. '
        )
        self.prompt = (
            'Control dual ARX X5 from RGB, task instruction and proprioception using exactly one act call per turn. '
            'Reason internally; execution_note is a brief intent/uncertainty note. '
            'Do not access simulator object poses, rewards, task code or layouts.\n\n'
            'EE means link6, not fingertip/TCP. EE poses use environment-origin positions with '
            'world-aligned axes, meters, and quaternions [w,x,y,z]. '
            + ARM_WORKSPACE_GUIDANCE + ' ' + ARM_POSTURE_GUIDANCE + ' ' +
            GRIPPER_GUIDANCE + '\n\n'
            'Each action row has 14 numbers: [left6,left_gripper,right6,right_gripper]. '
            'Use at most six decimal places. '
            + action_contract + prediction_contract +
            'Commands run at 25Hz. After a tracking-guard interruption, replan from the latest observation. '
            'Tool receipts identify executed steps and discarded suffixes; rejected acts execute zero steps. '
            'Small tracking residuals do not prove grasping or completion. '
            + ('Tool images show the current observation after the executed prefix. '
               if c.feedback_frames == 1 else 'Tool images show sampled, already executed observations. ') +
            'Continue until the environment ends the episode.\n\n'
            'TRAIN examples are separate trajectories. Learn progression and action scale, then adapt to current RGB; '
            'do not blindly replay coordinates or assume matching object layouts. '
            'Plans are not evidence of execution or success.'
        )
        if c.train_result_observations or c.live_anchor_count:
            self.prompt += (
                '\n\n' + ('TRAIN and retained LIVE chunks' if c.train_result_observations and c.live_anchor_count
                           else 'TRAIN chunks' if c.train_result_observations else 'Retained LIVE chunks')
                + ' show a starting observation, an action call, '
                'an execution receipt, and the immediate ending observation. '
                '<TRAJECTORY_GAP> marks omitted history, not a continuous displayed transition. '
                'Summaries describe recorded execution and stated intent, not proof of task success. '
                'execution_note is retained verbatim as stated intent when a LIVE chunk is compressed. '
                'OBSERVATION_END closes the displayed camera views only; it never means the episode ended. '
                'Use the latest LIVE observation for current geometry.'
            )
        if c.ik_continuity_guard:
            self.prompt += (
                f'\n\nBefore each LIVE EE action, the controller rejects IK solutions with a joint '
                f'change greater than {c.max_joint_increment_rad} rad from the observed joints '
                'or the preceding accepted IK target. A rejected action moves neither arm nor gripper; '
                'the remaining chunk is discarded. Use the receipt to distinguish the executed prefix '
                'from rejected actions, and replan with smaller increments or another approach.'
            )
        self.base_prompt = self.prompt
        # Keep serialization/config immutable throughout a conversation.
        self.frozen_tools = self._tools()

    def reset(self):
        super().reset()
        self.messages = []
        self.fixed_prefix_length = 1
        self.pending = None
        self.turns = 0
        self.stable_prefix_hash = None
        self.cache_input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.cache_reported_responses = 0
        self.cache_hit_responses = 0
        self.seen_call_ids = set()
        self.last_active_payload = None
        self.returned_reasoning_items = []
        self.reasoning_replay_disabled = False
        c = self.dialogue_config
        self.live_memory = LiveTrajectoryMemory(c.live_anchor_count, c.live_anchor_interval,
            retain_latest=c.live_retain_latest) if c.live_anchor_count else None
        self.live_history_tail_start = None
        self.controller_rejection_streak = 0
        self.local_safety_holds = 0

    def _tools(self):
        c = self.dialogue_config
        return [{'type': 'function', 'name': 'act', 'strict': True,
                 'description': ('Submit predicted action tensor [T,14]. Receive actual executed range and current RGB views.'
                                 if c.feedback_frames == 1 else
                                 'Submit predicted action tensor [T,14]. Receive actual executed range and RGB samples.'),
                 'parameters': {'type': 'object', 'additionalProperties': False,
                                'properties': {
                                    'actions': {'type': 'array', 'minItems': c.predict_horizon if c.fixed_predict_horizon else 1, 'maxItems': c.predict_horizon,
                                                'items': {'type': 'array', 'minItems': 14, 'maxItems': 14,
                                                          'items': {'type': 'number'}}},
                                    'execution_note': {'type': 'string', 'maxLength': 400}},
                                'required': ['actions', 'execution_note']}}]

    def _observations(self, steps, cameras=CAMERAS):
        content = []
        for step in steps:
            entry = self.archive[step]
            state = {'observation_step': step, 'robot_state': entry['state']}
            content.append({'type': 'input_text', 'text': json.dumps(state, ensure_ascii=False)})
            c = self.dialogue_config
            images = {camera: prompt_image(entry['images'][camera], c.image_profile) for camera in cameras}
            content.extend(camera_input_parts(step, images, cameras, profile=c.image_profile, detail=c.image_detail,
                                              image_layout=c.image_layout))
            if c.live_anchor_count:
                content.append(observation_end(step))
        return content

    def _initialize(self):
        from .train_reference_bundle import is_reference_bundle, validate_bundle
        bundle = bool(self.demo and is_reference_bundle(self.demo))
        if bundle:
            validate_bundle(self.demo)
            if not self.train_reference_checks:
                from .train_reference_checks import verify_training_manifest
                self.train_reference_checks = verify_training_manifest(self.demo)
            if (not self.train_reference_checks.get('passed')
                    or self.train_reference_checks.get('source_sha256') != self.demo['sha256']):
                raise ValueError('Bundle requires verified independent TRAIN sources')
        demos = self.demo['episodes'] if bundle else ([self.demo] if self.demo else [])
        reports = (self.train_reference_checks['episodes'] if bundle else [self.train_reference_checks])
        if bundle and (len(reports) != len(demos) or any(
                not report or not report.get('passed') or report.get('source_sha256') != demo['sha256']
                for demo, report in zip(demos, reports))):
            raise ValueError('Bundle requires a matching checked report for every TRAIN episode')
        train_calibrations = []
        system_parts = [self.base_prompt]
        instruction = self.archive[self.step]['instruction']
        task_prompt = 'TASK:\n' + instruction
        if self.dialogue_config.official_task_prompt:
            task_prompt += '\n\n' + self.dialogue_config.official_task_prompt
        prefix = [{'type': 'input_text', 'text': task_prompt}]
        system_parts.extend(part['text'] for part in image_layout_content(self.dialogue_config.image_layout))
        instruction_matches_reference = (
            instruction in self.dialogue_config.reference_instruction_aliases)
        if (self.demo and self.demo['instruction'] != instruction
                and not instruction_matches_reference):
            raise ValueError('Demo task mismatch')
        if self.demo and self.dialogue_config.fixed_predict_horizon:
            if any(len(example.get('action_tensor', [])) != self.dialogue_config.predict_horizon
                   for example in self.demo['examples']):
                raise ValueError('TRAIN action count must equal the fixed LIVE prediction horizon')
        if self.demo and (self.dialogue_config.demo_cameras == CAMERAS or
                self.dialogue_config.demo_frame0_cameras == CAMERAS or
                (self.dialogue_config.image_profile, self.dialogue_config.image_detail) != ('thumbnail_jpeg', 'low')):
            c = self.dialogue_config
            archive_encoding = self.demo.get('prompt_image_encoding')
            # Bundles store the individual native source JPEGs; the triptych is
            # composed only for presentation. Reuse those verified source bytes
            # for separate views without rewriting the immutable TRAIN bundle.
            separate_bundle_views = (bundle and c.image_profile == 'native_jpeg'
                and c.image_layout == 'separate'
                and archive_encoding == image_encoding('native_jpeg', c.image_detail, 'triptych'))
            if (archive_encoding != image_encoding(c.image_profile, c.image_detail, c.image_layout)
                    and not separate_bundle_views):
                raise ValueError('TRAIN source images must match the requested encoding contract')
            observations = [example for demo in demos for example in
                            demo['examples'] + ([demo['terminal_observation']] if demo.get('terminal_observation') else [])]
            for example in observations:
                available = set(example.get('prompt_images', {}))
                cameras = (self.dialogue_config.demo_frame0_cameras
                           if example['frame'] == 0 and self.dialogue_config.demo_frame0_cameras
                           else self.dialogue_config.demo_cameras)
                if (not set(cameras).issubset(available)
                        or not available.issubset(CAMERAS)):
                    raise ValueError('TRAIN example requires the selected prompt camera images')
                if example['head_image'] != example['prompt_images']['cam_head']:
                    raise ValueError('TRAIN head calibration image differs from prompt head image')
        live_calibration = None
        if self.dialogue_config.include_head_calibration:
            from .head_camera_prompt import calibration_content, calibration_conventions, image_size
            system_parts.append(calibration_conventions()['text'])
            context = self.head_calibration_context
            if not context or context.get('source') != 'TEST_RUNTIME':
                raise ValueError('Head calibration requires live TEST_RUNTIME camera context')
            initial = context['initial_sample']
            if (self.step != 0 or initial['step'] != 0 or initial['state'] != self.archive[0]['state']
                    or initial['head'] != context['head_reference']):
                raise ValueError('Head calibration must match the initial live observation')
            head = context['head_reference']
            live_image = self.archive[0]['images']['cam_head']
            if image_size(live_image) != (head['shape'][1], head['shape'][0]):
                raise ValueError('Native live head image differs from calibrated resolution')
            live_calibration = calibration_content(head, prompt_image(live_image, self.dialogue_config.image_profile), 'TEST_RUNTIME',
                                                  context['camera_source'], image_profile=self.dialogue_config.image_profile,
                                                  image_layout=self.dialogue_config.image_layout)
            for demo, report in zip(demos, reports):
                if (not report or not report.get('passed') or report.get('source_sha256') != demo['sha256']
                        or report.get('profile', {}).get('frame') != 'environment_origin'
                        or report.get('profile', {}).get('train_camera_convention') != 'camera_to_world_usd'
                        or demo['examples'][0]['frame'] != 0):
                    raise ValueError('Head calibration requires a checked frame-zero TRAIN reference')
                train_calibrations.append(calibration_content(report['head_reference'], demo['examples'][0]['head_image'],
                    'TRAIN', 'Source HDF5 calibration; camera-to-world USD convention and environment origin supplied by the explicit TRAIN geometry profile. Historical writer calibration is not independently certified.',
                    image_profile=self.dialogue_config.image_profile, image_layout=self.dialogue_config.image_layout))
        shared_train_calibration = bool(bundle and train_calibrations and all(
            item == train_calibrations[0] for item in train_calibrations))
        if train_calibrations:
            if not bundle or shared_train_calibration:
                value = json.loads(train_calibrations[0]['text'])
                value['applies_to'] = 'all TRAIN episodes'
                prefix.append({'type': 'input_text', 'text': json.dumps(value, ensure_ascii=False, separators=(',', ':'))})
            else:
                for demo, block in zip(demos, train_calibrations):
                    value = json.loads(block['text'])
                    source_id = hashlib.sha256(demo['sha256'].encode()).hexdigest()[:16]
                    value['applies_to_call_id_prefix'] = f'train_{source_id}_'
                    prefix.append({'type': 'input_text', 'text': json.dumps(value, ensure_ascii=False, separators=(',', ':'))})
        # Rebuild from the base, never append to a previous episode's system prompt.
        self.prompt = '\n\n'.join(system_parts)
        tool_demo = bool(self.demo)
        if tool_demo:
            if not bundle and self.demo.get('variant') != 'train_action_chunks_world_ee_delta_v1':
                raise ValueError('Tool-call few-shot requires action chunk labels')
            if self.dialogue_config.action_space != self.demo['action_space'] or self.demo['delta_frame'] != 'world':
                raise ValueError('TRAIN action space differs from online action space')
            if self.demo['frequency'] != 25:
                raise ValueError('TRAIN action rate differs from online action rate')
            options = dict(cameras=self.dialogue_config.demo_cameras,
                           frame0_cameras=self.dialogue_config.demo_frame0_cameras,
                           image_profile=self.dialogue_config.image_profile,
                           image_detail=self.dialogue_config.image_detail,
                           image_layout=self.dialogue_config.image_layout,
                           include_result_observations=self.dialogue_config.train_result_observations)
            if bundle:
                self.messages = []
                for index, (demo, report) in enumerate(zip(demos, reports)):
                    header = deepcopy(prefix) if index == 0 else []
                    if index and self.dialogue_config.train_result_observations:
                        self.messages.append(omitted_interval('TRAIN', None, None, reason='episode_boundary'))
                    self.messages.extend(tool_reference_messages(demo, header, self._convert_tensor, report,
                        include_end_marker=False, **options))
                self.messages.append({'role': 'user', 'content': [{'type': 'input_text',
                    'text': 'END OF FIXED TASK REFERENCE. Subsequent observations belong to the active episode.'}]})
            else:
                self.messages = tool_reference_messages(self.demo, prefix, self._convert_tensor, self.train_reference_checks, **options)
            self.seen_call_ids.update(item['call_id'] for item in self.messages
                                      if item.get('type') == 'function_call')
        if not tool_demo:
            self.messages = [{'role': 'user', 'content': prefix}]
        if live_calibration is not None:
            # Keep runtime matrices at the TRAIN/LIVE boundary, before the first
            # LIVE observation, and outside the rolling observation history.
            self.messages.append({'role': 'user', 'content': [live_calibration]})
        if self.dialogue_config.cache_mode == 'implicit':
            self.messages[-1]['content'][-1]['prompt_cache_breakpoint'] = {'mode': 'explicit'}
        self.fixed_prefix_length = len(self.messages)
        c = self.dialogue_config
        if self.live_memory and c.max_context_images is not None:
            fixed_images = sum(part.get('type') == 'input_image' for item in self.messages
                               for part in item.get('content', []))
            self.image_budget_plan = assert_image_budget(fixed_images, c.live_anchor_count,
                c.feedback_cameras, c.image_layout, max_images=c.max_context_images,
                retain_latest=c.live_retain_latest)
            self._log('image_budget.jsonl', {'phase': 'assembled_prefix', **self.image_budget_plan})
        digest_payload = {'model': self.model_name, 'instructions': self.prompt,
                          'tools': self.frozen_tools, 'prefix': self.messages, 'effort': self.effort}
        self.stable_prefix_hash = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest()
        self.messages.append({'role': 'user', 'content': self._observations([self.step])})
        if self.live_memory:
            self.live_memory.configure(self.archive[self.step].get('max_episode_steps'))
        self.live_history_tail_start = len(self.messages)

    def notify_action_rejection(self, obs):
        """Execution-side rejection after IK, before any physical action step.

        This is not an observation update: duplicate-step observations are
        intentionally ignored by the archive and must not invent execution.
        """
        step, event = obs['step'], obs['event']
        if self.pending is None or step != self.step:
            raise ValueError('Controller rejection requires a pending action at the latest step')
        if (event.get('type') != 'action_rejected_ik_continuity'
                or event.get('rejected_action_executed') is not False
                or event.get('action_index') != step - self.pending['start']
                or event.get('discarded_commands') != self.pending['returned'] - (step - self.pending['start'])
                or self.pending.get('local_safety_hold')
                or 'controller_rejection' in self.pending):
            raise ValueError('Invalid or duplicate controller rejection')
        self.pending['controller_rejection'] = deepcopy(event)
        self.controller_rejection_streak = (
            self.controller_rejection_streak + 1 if step == self.pending['start'] else 0)
        self._log('controller_rejections.jsonl', {'step': step, 'call_id': self.pending['call_id'], **event})

    def _commit_feedback(self, terminal=False):
        if self.pending is None:
            return
        start = self.pending['start']
        records = [r for r in self.ledger if start < r['resulting_step'] <= self.step]
        expected_ids = list(range(start+1, self.step+1))
        if [r['resulting_step'] for r in records] != expected_ids:
            raise ValueError('Missing execution receipts; will not infer execution from elapsed time')
        rejection = self.pending.get('controller_rejection')
        if not records and not terminal and rejection is None:
            raise ValueError('Pending act has no executed observation; duplicate inference prohibited')
        if len(records) > self.pending['returned']:
            raise ValueError('Observed more actions than dispatched')
        if records:
            self.controller_rejection_streak = 0
        steps = sample_steps([s for s in self.archive if start <= s <= self.step],
                             self.dialogue_config.feedback_frames,
                             self.dialogue_config.include_endpoints)
        command_ledger = [{'step': r['resulting_step'], 'command': r['executed_action']} for r in records]
        receipt = {'executed_step_interval': [start, self.step],
                   'predicted_steps': self.pending['predicted'],
                   'dispatched_steps': self.pending['returned'], 'executed_steps': len(records),
                   'unexecuted_steps': self.pending['predicted'] - len(records),
                   'executed_prediction_indices_inclusive': [0, len(records)-1] if records else [],
                   'executed_command_ledger_sha256': hashlib.sha256(json.dumps(command_ledger, sort_keys=True).encode()).hexdigest(),
                   'last_submitted_command': records[-1]['executed_action'] if records else {},
                   'latest_tracking': records[-1].get('tracking', {}) if records else {},
                   'controller_events': ([r['controller_event'] for r in records if 'controller_event' in r]
                                         + ([rejection] if rejection is not None else [])),
                   'sampled_steps': steps, 'terminal': terminal,
                   'success': 'not supplied to policy; infer progress from RGB only'}
        if self.pending.get('local_safety_hold'):
            if rejection is not None or len(records) != 1 or self.step != start + 1:
                raise ValueError('Local safety hold requires exactly one executed observation')
            notice = {
                'type': 'local_safety_hold_result',
                'executed': True,
                'executed_step_interval': [start, self.step],
                'arm_targets': 'copied from the latest observed joint state',
                'gripper_targets': 'copied from the latest observed commanded state',
                'reason': self.pending['local_safety_hold']['reason'],
                'recovery': ('The rejected proposal was not executed. A new observation follows; '
                             'replan from it without compensating for rejected motion.'),
            }
            self.messages.append({'role': 'user', 'content': [{
                'type': 'input_text', 'text': json.dumps(notice, ensure_ascii=False),
            }]})
            if self.live_memory:
                self.live_memory.capture(
                    start, self.step, self.messages[self.live_history_tail_start:],
                    summary={
                        'executed_steps': 1, 'predicted_steps': 1,
                        'discarded_prediction_steps': 0,
                        'execution_note': 'Local observed-state safety hold.',
                        'controller_interrupted': False,
                        'local_safety_hold': True,
                    },
                )
            visual = self._observations(steps, self.dialogue_config.feedback_cameras)
            extra_cameras = tuple(
                camera for camera in self.dialogue_config.endpoint_cameras
                if camera not in self.dialogue_config.feedback_cameras or self.step not in steps)
            if extra_cameras:
                visual.extend(self._observations([self.step], extra_cameras))
            self.messages.append({'role': 'user', 'content': visual})
            self.live_history_tail_start = len(self.messages)
            self._log('safety_holds.jsonl', {
                'turn': self.turns, **notice,
                'full_executed_commands_local_only': command_ledger,
            })
            self.pending = None
            return
        self.messages.append({'type': 'function_call_output', 'call_id': self.pending['call_id'],
                              'output': json.dumps(receipt, ensure_ascii=False)})
        if self.live_memory and records:
            self.live_memory.capture(start, self.step,
                self.messages[self.live_history_tail_start:], summary={
                    'executed_steps': len(records), 'predicted_steps': self.pending['predicted'],
                    'discarded_prediction_steps': self.pending['predicted'] - len(records),
                    'execution_note': self.pending.get('execution_note', ''),
                    'controller_interrupted': bool(receipt['controller_events'])})
        if not records:
            # Retain the call/rejection in the current tail for correction. No
            # new RGB, ledger entry or anchor is created for zero execution.
            self._log('tool_results.jsonl', {'turn': self.turns, 'call_id': self.pending['call_id'],
                                           **receipt, 'full_executed_commands_local_only': []})
            self.pending = None
            return
        # Archive each observation unchanged; the request view may omit old RGB.
        visual = self._observations(steps, self.dialogue_config.feedback_cameras)
        extra_cameras = tuple(camera for camera in self.dialogue_config.endpoint_cameras
                              if camera not in self.dialogue_config.feedback_cameras or self.step not in steps)
        if extra_cameras:
            visual.extend(self._observations([self.step], extra_cameras))
        self.messages.append({'role': 'user', 'content': visual})
        self.live_history_tail_start = len(self.messages)
        self._log('tool_results.jsonl', {'turn': self.turns, 'call_id': self.pending['call_id'],
                                        **receipt, 'full_executed_commands_local_only': command_ledger})
        self.pending = None

    def _request_inputs(self):
        if self.live_memory:
            observation = lambda step: {'role': 'user', 'content': self._observations(
                [step], self.dialogue_config.feedback_cameras)}
            live, stable_live_length = self.live_memory.render_with_frontier(
                observation(0), self.step, observation, tail=self.messages[self.live_history_tail_start:])
            inputs = deepcopy(self.messages[:self.fixed_prefix_length]) + live
            boundaries = []
            if self.dialogue_config.rolling_history_cache and self.dialogue_config.cache_mode == 'implicit':
                boundaries = annotate_anchored_breakpoints(inputs, self.fixed_prefix_length, stable_live_length)
            self._log('cache_breakpoints.jsonl', {'turn': self.turns+1, 'mode': 'anchored_chunks',
                'history_boundaries': boundaries, 'fixed_prefix_length': self.fixed_prefix_length,
                'note': 'Explicit text boundaries end at the latest fixed anchor. Growing merged gaps and the rolling latest chunk remain unmarked until an anchor freezes them. Eligibility only; hits require provider usage.'})
            self._log('context_window.jsonl', {'turn': self.turns+1, 'mode': 'anchored_chunks',
                **self.live_memory.stats, 'anchors_local_only': list(self.live_memory.anchors),
                'fixed_train_prefix_unchanged': True, 'local_archive_unchanged': True})
            return inputs
        inputs = deepcopy(self.messages)
        window = self.dialogue_config.visual_history_batches
        if not window:
            return inputs
        visual = [index for index, item in enumerate(inputs) if index >= self.fixed_prefix_length
                  and item.get('role') == 'user' and isinstance(item.get('content'), list)
                  and any(part.get('type') == 'input_image' for part in item['content'])]
        every = self.dialogue_config.visual_history_prune_every
        # At most window+every-1 live batches. Never mutate a wire prefix each turn
        # when batching is enabled; callers must size this against their image cap.
        trim_count = max(0, (len(visual)-window)//every*every)
        trimmed = visual[:trim_count]
        omitted = 0
        for index in trimmed:
            content = inputs[index]['content']
            omitted += sum(part.get('type') == 'input_image' for part in content)
            placeholder = {'type': 'input_text', 'text':
                'Earlier RGB omitted by the configured visual-history window; text/state/action receipts are retained. '
                'Use the newer RGB below for current geometry.'}
            inputs[index]['content'] = [deepcopy(placeholder) if part.get('type') == 'input_image' else part
                                        for part in content]
        boundaries = []
        if self.dialogue_config.rolling_history_cache and self.dialogue_config.cache_mode == 'implicit':
            boundaries = annotate_history_breakpoints(
                inputs, self.messages, visual, trim_count, every, self.fixed_prefix_length)
            self._log('cache_breakpoints.jsonl', {'turn': self.turns+1,
                'trimmed_live_batches': trim_count, 'history_boundaries': boundaries,
                'fixed_prefix_length': self.fixed_prefix_length,
                'note': 'Boundary placement only; cache hits require provider usage evidence.'})
        self._log('context_window.jsonl', {'turn': self.turns+1, 'visual_history_batches': window,
                  'omitted_live_images': omitted, 'retained_live_batches': len(visual)-trim_count,
                  'prune_every': every,
                  'fixed_train_prefix_unchanged': True, 'local_archive_unchanged': True})
        return inputs

    def build_payload(self):
        if self.step is None:
            raise ValueError('Observation required')
        if not self.messages:
            self._initialize()
        self._commit_feedback()
        payload = {'model': self.model_name, 'instructions': self.prompt,
                   'tools': deepcopy(self.frozen_tools), 'input': self._request_inputs(),
                   'tool_choice': {'type': 'function', 'name': 'act'}, 'parallel_tool_calls': False,
                   'store': False, 'reasoning': {'effort': self.effort}}
        if self.dialogue_config.max_output_tokens is not None and os.environ.get('ASTRA_OMIT_MAX_OUTPUT_TOKENS') != '1':
            payload['max_output_tokens'] = self.max_output
        if self.dialogue_config.preserve_reasoning:
            payload['include'] = ['reasoning.encrypted_content']
            payload['reasoning']['context'] = 'all_turns'
        if self.reasoning_replay_disabled:
            payload, _ = without_reasoning_replay(payload)
        if self.dialogue_config.reasoning_summary != 'omit':
            payload['reasoning']['summary'] = self.dialogue_config.reasoning_summary
        if self.dialogue_config.cache_mode == 'implicit':
            payload['prompt_cache_options'] = {'mode': 'implicit', 'ttl': '30m'}
            payload['prompt_cache_key'] = 'robodojo-'+self.stable_prefix_hash[:32]
        return payload

    def payload_metrics(self, payload):
        c = self.dialogue_config
        images = []
        def scrub(value):
            if isinstance(value, dict):
                if value.get('type') == 'input_image':
                    if value.get('detail') != c.image_detail:
                        raise ValueError('Image detail differs from the configured TRAIN/LIVE contract')
                    validate_prompt_image(value['image_url'], profile=c.image_profile, detail=c.image_detail,
                                          image_layout=c.image_layout)
                    images.append(value)
                    return {'type': 'input_image', 'detail': value['detail']}
                return {k: scrub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [scrub(v) for v in value]
            return value
        text_bytes = len(json.dumps(scrub(payload), ensure_ascii=False).encode())
        wire_bytes = len(json.dumps(payload).encode())
        reserve = text_bytes + len(images)*1024 + self.max_output
        if ((c.max_context_text_bytes is not None and text_bytes > c.max_context_text_bytes) or
                (c.max_context_images is not None and len(images) > c.max_context_images) or
                (c.max_context_wire_bytes is not None and wire_bytes > c.max_context_wire_bytes) or
                (c.max_context_reserved_tokens is not None and reserve > c.max_context_reserved_tokens)):
            raise BudgetExceeded('Dialogue context cap reached; no truncation, no paid compaction, no request')
        return {'text_bytes': text_bytes, 'image_count': len(images), 'wire_bytes': wire_bytes,
                'reserved_tokens': reserve, 'reservation_note': 'Estimate assumes no cache discount; not monetary guarantee'}

    def _request(self, payload):
        outgoing_reasoning = [item for item in payload['input'] if item.get('type')=='reasoning']
        expected_reasoning = self.returned_reasoning_items
        # Whole-chunk compression intentionally removes its opaque reasoning too.
        # Retained items must remain an exact, ordered subsequence of originals.
        remaining = iter(expected_reasoning)
        exact_subsequence = all(any(candidate == item for candidate in remaining) for item in outgoing_reasoning)
        replay_status = ('disabled_after_provider_rejection'
                         if self.reasoning_replay_disabled and not outgoing_reasoning
                         else 'no_reasoning_items_provided' if not expected_reasoning and not outgoing_reasoning
                         else 'exact_replay' if outgoing_reasoning == expected_reasoning
                         else 'intentional_chunk_compaction' if self.live_memory and exact_subsequence
                         else 'replay_mismatch')
        self._log('reasoning_replay.jsonl', {'turn':self.turns+1,
            'expected_items':len(self.returned_reasoning_items),'outgoing_items':len(outgoing_reasoning),
            'status': replay_status,
            'expected_item_sha256':[hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
                                    for item in self.returned_reasoning_items],
            'outgoing_item_sha256':[hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
                                    for item in outgoing_reasoning],
            'note':'Opaque items compared without decryption; no reasoning item is not evidence of reasoning replay.'})
        active_payload = payload
        bounded_failure_counts = {}
        for attempt in range(1, self.dialogue_config.max_transport_attempts+1):
            try:
                response = super()._request(active_payload)
                break
            except RequestTransportError as exc:
                if not exc.retryable or attempt >= self.dialogue_config.max_transport_attempts:
                    raise
                if exc.max_attempts is not None:
                    failure_key = (exc.error_code, exc.recovery)
                    bounded_failure_counts[failure_key] = (
                        bounded_failure_counts.get(failure_key, 0) + 1)
                    if bounded_failure_counts[failure_key] >= exc.max_attempts:
                        raise
                recovery = exc.recovery
                removed_reasoning_items = 0
                if recovery == 'drop_reasoning_items':
                    active_payload, removed_reasoning_items = without_reasoning_replay(active_payload)
                    if not removed_reasoning_items:
                        # Never turn a deterministic bad-request response into
                        # an identical paid retry.
                        raise
                    self.reasoning_replay_disabled = True
                delay = (0.0 if recovery is not None else
                         exc.retry_after_s if exc.retry_after_s is not None else min(2 ** (attempt-1), 8))
                if self.max_seconds is not None:
                    run_remaining = self.max_seconds - (time.monotonic() - self.started)
                    if delay >= run_remaining:
                        self._log('network_recovery.jsonl', {
                            'turn': self.turns+1, 'failed_attempt': attempt,
                            'retry_skipped': 'run_deadline', 'retry_delay_s': delay,
                            'run_remaining_s': max(run_remaining, 0.0),
                            'error_code': exc.error_code, 'robot_action_replayed': False,
                        })
                        raise BudgetExceeded(
                            'harness_timeout: insufficient time for transport retry') from exc
                next_wire_payload = to_wire_payload(active_payload, self.api_mode)
                if os.environ.get('ASTRA_STREAM', '0') == '1':
                    next_wire_payload['stream'] = True
                # No response was accepted and no robot actions were dispatched.
                # Conservatively retain the failed attempt's reservation: some
                # terminal/provider failures may still be billed.
                # Transient transport/provider errors and explicitly rebuilt
                # invalid-encrypted-content payloads follow this bounded path.
                self._log('network_recovery.jsonl', {'turn': self.turns+1, 'failed_attempt': attempt,
                    'next_attempt': attempt+1,
                    'retained_failed_reservation_tokens': self.pending_reserved_tokens,
                    'error_code': exc.error_code, 'recovery': recovery or 'same_payload',
                    'removed_reasoning_items': removed_reasoning_items,
                    'retry_delay_s': delay,
                    'next_payload_sha256': hashlib.sha256(
                        json.dumps(next_wire_payload).encode()).hexdigest(),
                    'possible_duplicate_inference_cost': recovery is None,
                    'robot_action_replayed': False})
                if recovery is None:
                    time.sleep(delay)
                self.budget_poisoned = False
        self.last_active_payload = deepcopy(active_payload)
        usage = response.get('usage', {})
        cache_usage = provider_cache_usage(usage)
        total = cache_usage['input_tokens']
        cached = cache_usage['cached_tokens']
        written = cache_usage['cache_write_tokens']
        valid = cache_usage['cache_reporting_valid']
        if valid:
            self.cache_reported_responses += 1
            self.cache_hit_responses += cached > 0
            self.cache_input_tokens += total
            self.cache_read_tokens += cached
            if type(written) is int and written >= 0:
                self.cache_write_tokens += written
        self._log('cache.jsonl', {'turn': self.turns+1, 'stable_prefix_sha256': self.stable_prefix_hash,
                  **cache_usage,
                  'server_diagnostics':response.get('prompt_cache_diagnostics'),
                  'request_has_cache_hit': cached > 0 if valid else None,
                  'cached_input_fraction': cached/total if valid and total else None,
                  'cumulative_request_hit_rate': self.cache_hit_responses/self.cache_reported_responses
                  if self.cache_reported_responses else None,
                  'cumulative_cached_input_fraction': self.cache_read_tokens/self.cache_input_tokens
                  if self.cache_input_tokens else None,
                  'note': 'Provider report only; not direct KV inspection or verified invoice'})
        return response

    def get_action(self):
        for correction in range(self.dialogue_config.max_action_corrections+1):
            returned = self._get_action_once(allow_correction=correction < self.dialogue_config.max_action_corrections)
            if returned is not None:
                return returned
        raise RuntimeError('Unreachable action correction state')

    def _observed_state_safety_hold(self, reason):
        """Return one bounded, local no-motion step or ``None`` without a bound.

        The action is reconstructed exclusively from the latest whitelisted
        proprioception. In particular, gripper targets are copied verbatim; no
        default/open/closed value is guessed. One accepted hold creates a new
        observation. The two nested correction paths and transport retry count
        therefore have a finite product with the episode's native step limit.
        """
        entry = self.archive[self.step]
        maximum = entry.get('max_episode_steps')
        if type(maximum) is not int or not 0 <= self.step < maximum:
            return None
        state = entry['state']
        try:
            raw = [{
                'mode': 'joint',
                'left': deepcopy(state['left_arm_joint_state']),
                'left_gripper': state['left_ee_joint_state'][0],
                'right': deepcopy(state['right_arm_joint_state']),
                'right_gripper': state['right_ee_joint_state'][0],
            }]
            hold = validate_actions(raw, 1, state)
        except (KeyError, IndexError, TypeError, ValueError):
            # Missing/invalid proprioception must never be replaced by guessed
            # arm or gripper commands.
            return None
        self.local_safety_holds += 1
        remaining_steps = maximum - self.step
        correction_width = self.dialogue_config.max_action_corrections + 1
        self.pending = {
            'start': self.step, 'predicted': 1, 'returned': 1,
            'execution_note': 'Local observed-state safety hold.',
            'local_safety_hold': {
                'reason': str(reason)[:500],
                'index': self.local_safety_holds,
            },
        }
        self._log('safety_holds.jsonl', {
            'turn': self.turns, 'step': self.step,
            'event': 'dispatched', 'reason': str(reason)[:500],
            'rejected_action_executed': False,
            'arm_targets': 'latest_observed_joint_state',
            'gripper_targets': 'latest_observed_commanded_state',
            'remaining_episode_steps': remaining_steps,
            'max_additional_provider_attempts': (
                remaining_steps * correction_width * correction_width
                * self.dialogue_config.max_transport_attempts),
        })
        return hold

    def _get_action_once(self, allow_correction=False):
        if self.dialogue_config.max_turns is not None and self.turns >= self.dialogue_config.max_turns:
            self.finalize()
            raise BudgetExceeded('harness_timeout: max_dialogue_turns reached')
        payload = self.build_payload()
        if self.controller_rejection_streak > self.dialogue_config.max_action_corrections:
            hold = self._observed_state_safety_hold(
                'Consecutive zero-step IK rejection correction limit reached')
            if hold is not None:
                return hold
            raise ValueError(
                'Action correction limit reached; bounded observed-state hold unavailable')
        metrics = self.payload_metrics(payload)
        self.last_active_payload = None
        response = self._request(payload)
        if self._time_limit_reached():
            raise BudgetExceeded('harness_timeout: response arrived after deadline; no action dispatched')
        self.turns += 1
        accepted_payload = self.last_active_payload or payload
        accepted_metrics = self.payload_metrics(accepted_payload)
        self._log('exchanges.jsonl', {'turn': self.turns, 'step': self.step,
                                     'request': accepted_payload, 'response': response,
                                     'payload_metrics': accepted_metrics})
        outputs = response.get('output', [])
        self.returned_reasoning_items.extend(deepcopy([item for item in outputs if item.get('type')=='reasoning']))
        calls = [item for item in outputs if item.get('type') == 'function_call']
        if len(calls) != 1 or calls[0]['name'] != 'act':
            raise ValueError('Expected one act tensor, no automatic retry')
        call_id = calls[0].get('call_id')
        if not isinstance(call_id, str) or not call_id.strip() or call_id in self.seen_call_ids:
            raise ValueError('Missing, empty or reused action call_id')
        self.messages.extend(canonical_tool_history(outputs))
        self.seen_call_ids.add(call_id)
        try:
            args = json.loads(calls[0]['arguments'])
            note = args['execution_note']
            if not isinstance(note, str) or len(note) > 400:
                raise ValueError('Invalid execution_note')
            rows = args['actions']
            converted = self._convert_tensor(rows, self.archive[self.step]['state'])
        except (ValueError, KeyError, TypeError) as exc:
            error={'executed':False,'executed_steps':0,'observation_step':self.step,
                   'error':'invalid_action_tensor','detail':str(exc)[:500],
                   'recovery':'No motion occurred. Correct the tensor from the SAME latest observation; do not compensate for unexecuted motion.'}
            self.messages.append({'type':'function_call_output','call_id':call_id,'output':json.dumps(error)})
            self._log('action_rejections.jsonl',{'turn':self.turns,'call_id':call_id,**error,
                       'correction_allowed':allow_correction})
            if allow_correction:
                return None
            hold = self._observed_state_safety_hold(
                'Action tensor correction limit reached: ' + str(exc))
            if hold is not None:
                return hold
            raise ValueError('Action correction limit reached: '+str(exc)) from exc
        c = self.dialogue_config
        returned = converted[:c.execute_horizon]
        self.pending = {'call_id': call_id, 'start': self.step,
                        'predicted': len(converted), 'returned': len(returned), 'execution_note': note}
        self.decisions.append({'step': self.step, 'execution_note': note,
                               'requested_tensor': rows, 'dispatched_steps': len(returned)})
        self._log('decisions.jsonl', self.decisions[-1])
        return returned

    def _convert_tensor(self, rows, state):
        # Booleans are not numeric robot actions, even though numpy can cast them.
        if self.dialogue_config.fixed_predict_horizon and (
                not isinstance(rows, list) or len(rows) != self.dialogue_config.predict_horizon):
            raise ValueError(f'actions must contain exactly {self.dialogue_config.predict_horizon} control steps; no padding or truncation')
        if not isinstance(rows, list) or not 1 <= len(rows) <= self.max_chunk:
            raise ValueError('Invalid prediction horizon')
        if any(not isinstance(r, list) or len(r) != 14 or
               any(type(v) not in (int, float) for v in r) for r in rows):
            raise ValueError('actions must have shape [T,14] with numeric values')
        tensor = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(tensor).all():
            raise ValueError('Nonfinite actions')
        c = self.dialogue_config
        raw = []
        prior_joint = {arm: np.asarray(state.get(f'{arm}_arm_joint_state', [])) for arm in ('left','right')}
        for index, row in enumerate(tensor):
            item = {'mode': c.action_space, 'left': row[:6], 'left_gripper': float(row[6]),
                    'right': row[7:13], 'right_gripper': float(row[13])}
            for arm in ('left', 'right'):
                vector = item[arm]
                if c.action_space == 'ee_delta':
                    translation, rotation = float(np.linalg.norm(vector[:3])), float(np.linalg.norm(vector[3:]))
                    if translation > c.max_translation_m+1e-6 or rotation > c.max_rotation_rad+1e-6:
                        raise ValueError(f'actions[{index}].{arm}: translation L2={translation:.8f}m '
                            f'(limit {c.max_translation_m}m), rotation L2={rotation:.8f}rad '
                            f'(limit {c.max_rotation_rad}rad); bounds are norms, not per-axis limits')
                else:
                    if prior_joint[arm].shape != (6,) or np.max(np.abs(vector-prior_joint[arm])) > c.max_joint_increment_rad+1e-6:
                        raise ValueError('Absolute joint target change exceeds bound')
                    prior_joint[arm] = vector
            raw.append(item)
        return validate_actions(raw, c.predict_horizon, state)

    def finalize(self):
        self._commit_feedback(terminal=True)
        self._log('dialogue_final.jsonl', {'turns': self.turns, 'last_step': self.step,
                  'stable_prefix_sha256': self.stable_prefix_hash, 'config': asdict(self.dialogue_config)})
        return {'turns': self.turns, 'step': self.step}
