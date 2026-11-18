import time

import torch
import inspect
import threading
import execution
import server


class ExecutionTime:
    CATEGORY = "Apt_Preset/chx_tools"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "process"

    def process(self):
        return ()


CURRENT_START_EXECUTION_DATA = None


def get_peak_memory():
    if not torch.cuda.is_available():
        return 0
    device = torch.device('cuda')
    return torch.cuda.max_memory_allocated(device)


def reset_peak_memory_record():
    if not torch.cuda.is_available():
        return
    device = torch.device('cuda')
    torch.cuda.reset_max_memory_allocated(device)


def handle_execute(class_type, last_node_id, prompt_id, server, unique_id, node_inputs=None):
    if not CURRENT_START_EXECUTION_DATA:
        return
    start_time = CURRENT_START_EXECUTION_DATA['nodes_start_perf_time'].get(unique_id)
    start_vram = CURRENT_START_EXECUTION_DATA['nodes_start_vram'].get(unique_id)
    if start_time:
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        end_vram = get_peak_memory()
        vram_used = end_vram - start_vram
        print(f"end_vram - start_vram: {end_vram} - {start_vram} = {vram_used}")
        if server.client_id is not None and last_node_id != server.last_node_id:
            data = {"node": unique_id, "prompt_id": prompt_id, "execution_time": int(execution_time * 1000),
                    "vram_used": vram_used, "class_type": class_type}
            if class_type == "flow_stage_begin" and isinstance(node_inputs, dict):
                data["flow_stage_index"] = node_inputs.get("stage_index")
                data["flow_stage_total"] = node_inputs.get("total")
            server.send_sync(
                "TyDev-Utils.ExecutionTime.executed",
                data,
                server.client_id
            )
        print(f"#{unique_id} [{class_type}]: {execution_time:.2f}s - vram {vram_used}b")


# We avoid monkeypatching at import time because new ComfyUI versions
# may not have the same attributes available during import, which can
# break startup. Instead we poll in a background thread and apply the
# patches once the target attributes are present.

_PATCHED_EXECUTE = False
_PATCHED_RECURSIVE = False
_PATCHED_SEND_SYNC = False


def _apply_execute_patch():
    global _PATCHED_EXECUTE, _PATCHED_RECURSIVE
    if _PATCHED_EXECUTE:
        return

    try:
        origin_execute = getattr(execution, 'execute', None)
        if origin_execute is None:
            return False

        if inspect.iscoroutinefunction(origin_execute):
            async def dev_utils_execute(server_obj, dynprompt, caches, current_item, extra_data, executed, prompt_id,
                                        execution_list, pending_subgraph_results, pending_async_nodes, *args, **kwargs):
                unique_id = current_item
                node = dynprompt.get_node(unique_id)
                class_type = node['class_type']
                last_node_id = server_obj.last_node_id
                result = await origin_execute(server_obj, dynprompt, caches, current_item, extra_data, executed, prompt_id,
                                              execution_list, pending_subgraph_results, pending_async_nodes, *args, **kwargs)
                handle_execute(class_type, last_node_id, prompt_id, server_obj, unique_id, node.get('inputs'))
                return result
            execution.execute = dev_utils_execute
        else:
            def dev_utils_execute(server_obj, dynprompt, caches, current_item, extra_data, executed, prompt_id,
                                  execution_list, pending_subgraph_results, *args, **kwargs):
                unique_id = current_item
                node = dynprompt.get_node(unique_id)
                class_type = node['class_type']
                last_node_id = server_obj.last_node_id
                result = origin_execute(server_obj, dynprompt, caches, current_item, extra_data, executed, prompt_id,
                                        execution_list, pending_subgraph_results, *args, **kwargs)
                handle_execute(class_type, last_node_id, prompt_id, server_obj, unique_id, node.get('inputs'))
                return result
            execution.execute = dev_utils_execute

        _PATCHED_EXECUTE = True
        # try patching recursive_execute if present (deprecated hook)
        origin_recursive = getattr(execution, 'recursive_execute', None)
        if origin_recursive and not _PATCHED_RECURSIVE:
            def dev_utils_origin_recursive_execute(server_obj, prompt, outputs, current_item, extra_data, executed, prompt_id,
                                                   outputs_ui, object_storage):
                unique_id = current_item
                class_type = prompt[unique_id]['class_type']
                last_node_id = server_obj.last_node_id
                result = origin_recursive(server_obj, prompt, outputs, current_item, extra_data, executed, prompt_id,
                                          outputs_ui, object_storage)
                handle_execute(class_type, last_node_id, prompt_id, server_obj, unique_id, prompt[unique_id].get('inputs'))
                return result

            execution.recursive_execute = dev_utils_origin_recursive_execute
            _PATCHED_RECURSIVE = True

        return True
    except Exception:
        return False


def _apply_send_sync_patch():
    global _PATCHED_SEND_SYNC
    if _PATCHED_SEND_SYNC:
        return True
    try:
        ps = getattr(server, 'PromptServer', None)
        if ps is None:
            return False
        origin_func = getattr(ps, 'send_sync', None)
        if origin_func is None:
            return False

        def dev_utils_send_sync(self, event, data, sid=None):
            global CURRENT_START_EXECUTION_DATA
            if event == "execution_start":
                CURRENT_START_EXECUTION_DATA = dict(
                    start_perf_time=time.perf_counter(),
                    nodes_start_perf_time={},
                    nodes_start_vram={}
                )

            origin_func(self, event=event, data=data, sid=sid)

            if event == "executing" and data and CURRENT_START_EXECUTION_DATA:
                if data.get("node") is None:
                    if sid is not None:
                        start_perf_time = CURRENT_START_EXECUTION_DATA.get('start_perf_time')
                        new_data = data.copy()
                        if start_perf_time is not None:
                            execution_time = time.perf_counter() - start_perf_time
                            new_data['execution_time'] = int(execution_time * 1000)
                        origin_func(self, event="TyDev-Utils.ExecutionTime.execution_end", data=new_data, sid=sid)
                else:
                    node_id = data.get("node")
                    CURRENT_START_EXECUTION_DATA['nodes_start_perf_time'][node_id] = time.perf_counter()
                    reset_peak_memory_record()
                    CURRENT_START_EXECUTION_DATA['nodes_start_vram'][node_id] = get_peak_memory()

        setattr(ps, 'send_sync', dev_utils_send_sync)
        _PATCHED_SEND_SYNC = True
        return True
    except Exception:
        return False


def _poll_and_patch():
    # Run in background thread to avoid blocking import
    for _ in range(0, 60):
        try:
            done_exec = _apply_execute_patch()
            done_send = _apply_send_sync_patch()
            if done_exec and done_send:
                break
        except Exception:
            pass
        time.sleep(0.5)


# start the patcher thread
threading.Thread(target=_poll_and_patch, daemon=True).start()
