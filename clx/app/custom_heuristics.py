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


def within_first(text, term, n):
    first_n = " ".join(text.split()[:n])
    return term in first_n


@custom_heuristic("docket-entry", "Motion")
def first_3_motion(text, **kwargs):
    return within_first(text.lower(), "motion", 3)
