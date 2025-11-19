custom_heuristics = {}


def custom_heuristic(project_id, label_name):
    def decorator(f):
        custom_heuristics[f.__name__] = {
            "project_id": project_id,
            "label_name": label_name,
            "apply_fn": f,
        }
        return f

    return decorator


@custom_heuristic("docket-entry", "Motion")
def startswith_motion_caps(text, **kwargs):
    return text.startswith("MOTION")
