# web/app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "植物识别系统 Web 界面示例"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
# 运行后访问 http://127.0.0.1:5000/ 即可看到 JSON 响应
# 这是一个简单的 Flask Web 应用示例，可以扩展为更复杂的界面
# 你可以添加更多路由和功能来展示模型预测结果等
# 例如添加一个 /predict 路由来处理图片上传和返回预测结果
# 需要安装 Flask：pip install Flask
# 运行命令：python src/web/app.py
# 这是一个简单的示例，实际项目中你可能需要更复杂的前端框架（如 React/Vue）和后端逻辑
# 以及与模型推理的集成
# 你可以根据需要修改和扩展这个文件
# 这个文件位于 plant_disease_model/src/web/app.py
# 你可以把它作为项目的 Web 界面入口 
# 方便展示和交互
# 当然，你也可以使用其他 Web 框架如 FastAPI、Django 等

#安装 Flask 后就可以直接运行: python web/app.py