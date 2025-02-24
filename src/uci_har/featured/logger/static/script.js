document.addEventListener('DOMContentLoaded', function () {
    // 获取日志文件夹列表并填充到下拉框中
    fetch('/api/logs')
        .then(response => response.json())
        .then(data => {
            const log1Select = document.getElementById('log1');
            const log2Select = document.getElementById('log2');

            // 清空现有选项
            log1Select.innerHTML = '';
            log2Select.innerHTML = '';

            // 动态填充日志文件夹选项
            data.logs.forEach(log => {
                const option1 = document.createElement('option');
                option1.value = log;
                option1.textContent = log;
                log1Select.appendChild(option1);

                const option2 = document.createElement('option');
                option2.value = log;
                option2.textContent = log;
                log2Select.appendChild(option2);
            });
        })
        .catch(error => console.error('Error fetching logs:', error));

    // 比较按钮点击事件
    document.getElementById('compareBtn').addEventListener('click', function () {
        const log1 = document.getElementById('log1').value;
        const log2 = document.getElementById('log2').value;

        fetch(`/api/compare/${log1}/${log2}`)
            .then(response => response.json())
            .then(data => {
                const differenceList = document.getElementById('differenceList');
                differenceList.innerHTML = '';  // 清空之前的结果

                if (data.differences && data.differences.length > 0) {
                    data.differences.forEach(difference => {
                        const li = document.createElement('li');
                        li.textContent = difference;
                        differenceList.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.textContent = data.message;
                    differenceList.appendChild(li);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });
});
