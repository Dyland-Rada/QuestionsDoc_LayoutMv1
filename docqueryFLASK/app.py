from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from transformers import pipeline
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

models_checkpoint = {
    "LayoutLMv1 🦉": "impira/layoutlm-document-qa",
    "LayoutLMv1 for Invoices": "impira/layoutlm-invoices",
    "Donut": "naver-clova-ix/donut-base-finetuned-docvqa",
}

pipe = pipeline("document-question-answering", model=models_checkpoint["LayoutLMv1 for Invoices"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['file']
    question = request.form['question']
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Cargar la imagen usando PIL
        image = Image.open(file_path)

        # Ejecutar el pipeline
        result = pipe(image=image, question=question)
        answer = result[0]['answer'] if result else "No answer found."

        return render_template('result.html', question=question, result=answer, file_path=filename)
    else:
        return redirect(url_for('home'))

@app.route('/ask_question', methods=['POST'])
def ask_question():
    file_path = request.form['file_path']
    question = request.form['question']
    
    # Cargar la imagen usando PIL
    image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], file_path))

    # Ejecutar el pipeline
    result = pipe(image=image, question=question)
    answer = result[0]['answer'] if result else "No answer found."

    return render_template('result.html', question=question, result=answer, file_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
