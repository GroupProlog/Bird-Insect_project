from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)


#this will have to be changed once we have our api and hosting service
app.config["UPLOAD FOLDER"] = "static/"

@app.route("/")
def upload_file():
    return render_template('index.html')


@app.route('/display', methods = ['GET', 'POST'])
def display_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        f.save(app.config['UPLOAD_FOLDER'] + filename)

        file = open(app.config['UPLOAD_FOLDER'] + filename,"r")
        content = file.read()   
        
    return render_template('content.html', content=content) 


if __name__ == '__main__':
    app.run(port=5000, debug = True)
