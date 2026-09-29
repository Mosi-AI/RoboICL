"""Serialize sampled TRAIN labels as tool history, without executing them."""
from copy import deepcopy
import hashlib
import json

from .action_feedback import tracking_receipt
from .astra_policy import STATE_KEYS, plain
from .observation_images import camera_input_parts
from .anchored_cache import observation_end


def tool_reference_messages(demo, header, convert, consistency=None, *, cameras=('cam_head',),
                            frame0_cameras=(), image_profile='thumbnail_jpeg', image_detail='low',
                            image_layout='separate', include_end_marker=True,
                            include_result_observations=False):
    if consistency is None and demo.get('geometry_profile'):
        from .train_reference_checks import verify_training_manifest
        consistency = verify_training_manifest(demo)
    examples = demo['examples']
    if not examples or examples[0]['frame'] != 0:
        raise ValueError('Tool-call few-shot requires frame 0 and its following action chunk')
    messages = []
    source_id = hashlib.sha256(demo['sha256'].encode()).hexdigest()[:16]
    terminal = demo.get('terminal_observation')
    endpoints = {}
    if include_result_observations:
        from .train_endpoint_observations import result_observations
        from .trajectory_memory import omitted_interval
        endpoints = result_observations(demo, cameras=cameras, image_profile=image_profile)
    previous_result = 0
    for index, example in enumerate(examples):
        start, end = example['frame'], example['result_frame']
        tensor = example['action_tensor']
        count = len(tensor)
        if (type(start) is not int or type(end) is not int or start < previous_result or
                count < 1 or end != start + count or
                example['action_step_interval_inclusive'] != [start, end - 1]):
            raise ValueError('TRAIN chunks require ordered, non-overlapping observation/action/result intervals')
        # Validate every row using the same converter and limits as live actions.
        commands = plain(convert(tensor, example['state']))
        if len(commands) != count:
            raise ValueError('TRAIN conversion must not truncate an action chunk')
        state = {k: deepcopy(example['state'][k]) for k in STATE_KEYS if k in example['state']}
        result_state = {k: deepcopy(example['result_state'][k]) for k in STATE_KEYS if k in example['result_state']}
        observation = {'source': 'TRAIN', 'observation_step': start, 'robot_state': state}
        if include_result_observations and start > previous_result:
            messages.append(omitted_interval('TRAIN', previous_result, start, episode_id=source_id))
        content = deepcopy(header) if index == 0 else []
        content.append({'type': 'input_text', 'text': json.dumps(observation, ensure_ascii=False)})
        selected_cameras = frame0_cameras if start == 0 and frame0_cameras else cameras
        if selected_cameras == ('cam_head',) and (image_profile, image_detail, image_layout) == ('thumbnail_jpeg', 'low', 'separate'):
            content.append({'type': 'input_image', 'detail': 'low', 'image_url': example['head_image']})
        else:
            content.extend(camera_input_parts(start, example['prompt_images'], selected_cameras,
                                              profile=image_profile, detail=image_detail, image_layout=image_layout))
        if include_result_observations:
            content.append(observation_end(start))
        messages.append({'role': 'user', 'content': content})
        call_id = f'train_{source_id}_{index}_{start}'
        messages.append({'type': 'function_call', 'name': 'act', 'call_id': call_id,
            'arguments': json.dumps({'actions': tensor, 'execution_note': ''},
                                    ensure_ascii=False, separators=(',', ':'))})
        ledger = [{'step': start + i + 1, 'command': command} for i, command in enumerate(commands)]
        receipt = {
            'source': 'TRAIN',
            'executed_step_interval': [start, end],
            'predicted_steps': count, 'dispatched_steps': count, 'executed_steps': count,
            'unexecuted_steps': 0, 'executed_prediction_indices_inclusive': [0, count - 1],
            'executed_command_ledger_sha256': hashlib.sha256(json.dumps(ledger, sort_keys=True).encode()).hexdigest(),
            'last_submitted_command': commands[-1],
            'latest_tracking': tracking_receipt(commands[-1], result_state),
            'result_state': result_state,
            # The dataset supplies targets and next states, not controller events
            # or episode outcomes. Null preserves that distinction from live logs.
            'controller_events': None, 'sampled_steps': [start, end],
            'terminal': None, 'success': None,
        }
        messages.append({'type': 'function_call_output', 'call_id': call_id,
                         'output': json.dumps(receipt, ensure_ascii=False)})
        if include_result_observations:
            endpoint = endpoints[end]
            observation = {'source': 'TRAIN', 'observation_step': end,
                           'robot_state': endpoint['state']}
            if terminal is not None and index == len(examples) - 1 and terminal['frame'] == end:
                observation['end_of_example'] = True
            content = [{'type': 'input_text', 'text': json.dumps(observation, ensure_ascii=False)}]
            content.extend(camera_input_parts(end, endpoint['prompt_images'], cameras,
                profile=image_profile, detail=image_detail, image_layout=image_layout))
            content.append(observation_end(end))
            messages.append({'role': 'user', 'content': content})
        previous_result = end
    if terminal is not None:
        if terminal['frame'] != demo['total_frames'] - 1 or terminal['frame'] < previous_result:
            raise ValueError('Final TRAIN observation must be the true final source frame after selected chunks')
        if include_result_observations and terminal['frame'] > previous_result:
            messages.append(omitted_interval('TRAIN', previous_result, terminal['frame'], episode_id=source_id))
        if not include_result_observations or terminal['frame'] != previous_result:
            content = [{'type': 'input_text', 'text': json.dumps({
                'source': 'TRAIN', 'observation_step': terminal['frame'],
                'robot_state': terminal['state'], 'end_of_example': True}, ensure_ascii=False)}]
            content.extend(camera_input_parts(terminal['frame'], terminal['prompt_images'], cameras,
                profile=image_profile, detail=image_detail, image_layout=image_layout))
            if include_result_observations:
                content.append(observation_end(terminal['frame']))
            messages.append({'role': 'user', 'content': content})
    if include_end_marker:
        messages.append({'role': 'user', 'content': [{'type': 'input_text',
            'text': 'END OF FIXED TASK REFERENCE. Subsequent observations belong to the active episode.'}]})
    return messages
