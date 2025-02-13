def parse_log(file_path):
    params = {}
    metrics = {}
    current_model = None
    parsing_models = True

    with open(file_path, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        line = lines[i]
        if parsing_models:
            if line.startswith('>>'):
                model_part = line[2:].split(':', 1)[0].strip()
                train_time = line[2:].split(':', 1)[1].strip().rstrip(" s")
                current_model = model_part
                params[current_model] = {}
                metrics[current_model] = {}
                metrics[current_model]["Time Cost"] = float(train_time)
                i += 1
            elif line.startswith(' *'):
                if current_model is None:
                    i += 1
                    continue
                param_line = line[1:].strip()
                if ':' in param_line:
                    param_name, param_value = param_line.split(':', 1)
                    param_name = param_name.strip()
                    param_name = param_name.lstrip("* ")
                    param_value = param_value.strip()
                    params[current_model][param_name] = param_value
                i += 1
            else:
                if line.startswith('Model') and '|' in line and 'Accuracy' in line:
                    parsing_models = False
                    i += 2  # Skip header and separator line
                    while i < len(lines):
                        data_line = lines[i]
                        if not data_line:
                            i += 1
                            continue
                        if '|' not in data_line:
                            break
                        parts = [p.strip() for p in data_line.split('|')]
                        if len(parts) < 4:
                            i += 1
                            continue
                        model_name = parts[0]
                        try:
                            accuracy = float(parts[1])
                            macro_f1 = float(parts[2])
                            micro_f1 = float(parts[3])
                            metrics[model_name].update({
                                'Accuracy': accuracy,
                                'Macro F1': macro_f1,
                                'Micro F1': micro_f1
                            })
                        except ValueError:
                            pass
                        i += 1
                else:
                    i += 1
        else:
            i += 1
    return params, metrics


def compare_params(old_params, new_params):
    changes = {}
    all_models = set(old_params.keys()).union(new_params.keys())
    for model in all_models:
        old = old_params.get(model, {})
        new = new_params.get(model, {})
        model_changes = {}
        all_params = set(old.keys()).union(new.keys())
        for param in all_params:
            old_val = old.get(param)
            new_val = new.get(param)
            if old_val != new_val:
                model_changes[param] = (old_val, new_val)
        if model_changes:
            changes[model] = model_changes
    return changes


class Color:
    def red(text):
        return f"\033[41;30m{text}\033[0m"

    def green(text):
        return f"\033[42;30m{text}\033[0m"


def main(old_log, new_log):
    old_params, old_metrics = parse_log(old_log)
    new_params, new_metrics = parse_log(new_log)
    # import json
    # json_str1 = json.dumps(old_params, sort_keys=True, indent=2)
    # json_str2 = json.dumps(new_params, sort_keys=True, indent=2)
    # print(json_str1)
    # print(json_str2)
    # exit(0)

    param_changes = compare_params(old_params, new_params)

    print("% 参数变化：")
    for model, changes in param_changes.items():
        print(f"  >> {model}:")
        for param, (old_val, new_val) in changes.items():
            old_str = str(old_val) if old_val is not None else "None"
            new_str = str(new_val) if new_val is not None else "None"
            print(f"    * {param}: {old_str} → {new_str}")

    print("\n% 指标变化：")
    table = f"{'Model':<20} | {'Accuracy':>27} | {'Macro F1':>27} | {'Micro F1':>27} | {'Time Cost':>27}\n"
    table += "-" * 140 + '\n'
    for model in param_changes.keys():
        old_metrics_row = old_metrics.get(model, {})
        new_metrics_row = new_metrics.get(model, {})
        accuracy_old = old_metrics_row.get('Accuracy', 0)
        accuracy_new = new_metrics_row.get('Accuracy', 0)
        accuracy_delta = (accuracy_new - accuracy_old) * 100
        macro_f1_old = old_metrics_row.get('Macro F1', 0)
        macro_f1_new = new_metrics_row.get('Macro F1', 0)
        macro_f1_delta = (macro_f1_new - macro_f1_old) * 100
        micro_f1_old = old_metrics_row.get('Micro F1', 0)
        micro_f1_new = new_metrics_row.get('Micro F1', 0)
        micro_f1_delta = (micro_f1_new - micro_f1_old) * 100
        time_old = old_metrics_row.get('Time Cost', 0)
        time_new = new_metrics_row.get('Time Cost', 0)
        time_delta = (time_new - time_old) / time_old

        accuracy_str = f"{accuracy_old:.5f} -> {accuracy_new:.5f} " + (Color.green(f"({accuracy_delta:+.2f}%)") if accuracy_delta > 0 else Color.red(f"({accuracy_delta:+.2f}%)"))
        macro_f1_str = f"{macro_f1_old:.5f} -> {macro_f1_new:.5f} " + (Color.green(f"({macro_f1_delta:+.2f}%)") if macro_f1_delta > 0 else Color.red(f"({macro_f1_delta:+.2f}%)"))
        micro_f1_str = f"{micro_f1_old:.5f} -> {micro_f1_new:.5f} " + (Color.green(f"({micro_f1_delta:+.2f}%)") if micro_f1_delta > 0 else Color.red(f"({micro_f1_delta:+.2f}%)"))
        time_str = f"{time_old:.5f} s -> {time_new:.5f} s " + (Color.green(f"({time_delta:+.2f}%)") if time_delta > 0 else Color.red(f"({time_delta:+.2f}%)"))

        table += f"{model:<20} | {accuracy_str:>20} | {macro_f1_str:>20} | {micro_f1_str:>20} | {time_str:>20}" + '\n'

    print(table)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python log.py <old_log> <new_log>")
        sys.exit(1)
    else:
        print(f"# {sys.argv[1]} -> {sys.argv[2]}\n\n")
        main(sys.argv[1], sys.argv[2])
