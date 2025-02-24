import os
import toml
from flask import Flask, render_template, jsonify

app = Flask(__name__)

LOG_DIR = './log'


def read_toml(file_path):
    try:
        with open(file_path, 'r') as f:
            return toml.load(f)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}


def compare_toml(toml1, toml2):
    differences = []

    for section in toml1:
        if section in toml2:
            for key in toml1[section]:
                if key in toml2[section]:
                    value1 = toml1[section][key]
                    value2 = toml2[section][key]
                    if value1 != value2:
                        differences.append(f"[{section}] {key}: {value1} -> {value2}")
    return differences


def compare_hyper_toml(log1, log2):
    toml1 = read_toml(f'./log/{log1}/hyper.toml')
    toml2 = read_toml(f'./log/{log2}/hyper.toml')
    differences = compare_toml(toml1, toml2)
    return differences


@app.route('/')
def index():
    return render_template('index.html')

# 新增的路由，列出所有日志文件夹


@app.route('/api/logs', methods=['GET'])
def list_logs():
    logs = [folder for folder in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, folder))]
    return jsonify({"logs": logs})


@app.route('/api/compare/<log1>/<log2>', methods=['GET'])
def compare_logs(log1, log2):
    differences = compare_hyper_toml(log1, log2)
    if differences:
        return jsonify({"differences": differences})
    else:
        return jsonify({"message": "没有发现参数变化"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
